"""
Retrieval Tools V2 — 6 个工具实现

V2 改动（相对 v1）：
  search_memory: 单路 unified_index（LTMemory 式，不用 RRF 融合）
  search_by_speaker / search_personal: 改查 unified_index（原查 text_index）
  新增 search_by_caption: 查 caption_index（route 专项工具，用于图像内容问题）
  删除 _rrf_fuse（不再需要，search_memory 单路）

search_memory / search_by_speaker / search_personal / search_conflict / fetch_raw / search_by_caption
"""
import numpy as np

import config
from stores.semantic_store import SemanticStore
from stores.episodic_store import EpisodicStore
from stores.raw_store import RawStore
from stores.registry import SpeakerRegistry
from stores.bm25_index import BM25Index
from encoders.qwen3vl_encoder import encode_text

text_encode = encode_text


class RetrievalTools:
    def __init__(self):
        self.sem_store = SemanticStore()
        self.epi_store = EpisodicStore()
        self.raw_store = RawStore()
        self.registry = SpeakerRegistry()
        self.bm25 = BM25Index()
        self._bm25_built = False

    def _ensure_bm25(self):
        """懒加载 BM25 索引（首次调用时从 sem_store 构建）。"""
        if not self._bm25_built:
            n = self.bm25.build_from_store(self.sem_store)
            print(f"  [BM25] index built: {n} docs", flush=True)
            self._bm25_built = True

    def _to_retrieved(self, mem: dict) -> dict:
        """SemanticMemory → RetrievedMemory（整体返回，带 mem_id + image_paths）。"""
        return {
            "mem_id": mem["mem_id"],
            "text": mem.get("text"),
            "caption": mem.get("image_captions", {}),
            "image_paths": mem.get("image_paths", []),
            "speaker_id": mem.get("speaker_id"),
            "speaker": mem.get("speaker"),
            "addressee": mem.get("addressee"),
            "timestamp": mem.get("timestamp"),
            "date": mem.get("date", ""),              # V2: 日期字符串（对齐 LTMemory）
            "is_superseded": mem.get("is_superseded", 0),
            "supersede_reason": mem.get("supersede_reason"),
            "score": mem.get("score", 0.0),
        }

    # ── Tool 1: search_memory ───────────────────────────────

    def search_memory(self, query: str,
                      session_filter: str = None,
                      time_range: tuple = None,
                      include_history: bool = False,
                      top_k: int = None) -> list[dict]:
        """非个性化语义检索。V2: 单路 unified_index（和 LTMemory 一样）。

        unified_emb = text+caption 拼接编码，query 原文直接查 cosine。
        不再用 text+caption 两路 RRF。
        """
        top_k = top_k or config.TOP_K_DEFAULT
        query_emb = text_encode(query)

        results = self.sem_store.search_by_emb(
            self.sem_store.unified_index, self.sem_store.unified_ids, query_emb,
            top_k=top_k * 2,
            filter_kwargs={"session_id": session_filter, "time_range": time_range,
                           "include_history": include_history}
        )
        for m in results[:top_k]:
            self.sem_store.increment_access(m["mem_id"])
        return [self._to_retrieved(m) for m in results[:top_k]]

    # ── Tool 2: search_by_speaker ───────────────────────────

    def search_by_speaker(self, query: str, speaker: str,
                          session_filter: str = None,
                          time_range: tuple = None,
                          top_k: int = None) -> list[dict]:
        """speaker_id 过滤（speaker 名经 Registry 解析）。V2: 查 unified_index。"""
        top_k = top_k or config.TOP_K_DEFAULT
        speaker_id = self.registry.resolve_speaker_id(speaker) or speaker
        query_emb = text_encode(query)

        results = self.sem_store.search_by_emb(
            self.sem_store.unified_index, self.sem_store.unified_ids, query_emb,
            top_k=top_k * 2,
            filter_kwargs={"speaker_id": speaker_id, "session_id": session_filter,
                           "time_range": time_range}
        )
        for m in results[:top_k]:
            self.sem_store.increment_access(m["mem_id"])
        return [self._to_retrieved(m) for m in results[:top_k]]

    # ── Tool 3: search_personal ─────────────────────────────

    def search_personal(self, query: str,
                        current_speaker: str = None,
                        session_filter: str = None,
                        time_range: tuple = None,
                        top_k: int = None) -> dict:
        """个性化检索。speaker 作 access key。V2: 查 unified_index。"""
        top_k = top_k or config.TOP_K_DEFAULT
        degraded = False
        degrade_reason = None

        if not current_speaker:
            # 声纹识别后续加，先退化
            degraded = True
            degrade_reason = "voiceprint not available, current_speaker not provided"

        query_emb = text_encode(query)
        if current_speaker:
            # addressee 过滤（当前说话人作为受众）
            speaker_id = self.registry.resolve_speaker_id(current_speaker) or current_speaker
            results = self.sem_store.search_by_emb(
                self.sem_store.unified_index, self.sem_store.unified_ids, query_emb,
                top_k=top_k * 2,
                filter_kwargs={"addressee": speaker_id, "session_id": session_filter,
                               "time_range": time_range}
            )
        else:
            # 退化：不过滤 addressee
            results = self.sem_store.search_by_emb(
                self.sem_store.unified_index, self.sem_store.unified_ids, query_emb,
                top_k=top_k * 2,
                filter_kwargs={"session_id": session_filter, "time_range": time_range}
            )
        for m in results[:top_k]:
            self.sem_store.increment_access(m["mem_id"])
        return {
            "results": [self._to_retrieved(m) for m in results[:top_k]],
            "current_speaker": current_speaker,
            "degraded": degraded,
            "degrade_reason": degrade_reason,
        }

    # ── Tool 4: search_conflict ─────────────────────────────

    def search_conflict(self, entity: str,
                        relation: str = None,
                        include_history: bool = True) -> dict:
        """走 Episodic 三级回退：Conflict 边 → Temporal 边 → Relation 一致性。"""
        # 1. Conflict 边（两人矛盾）
        conflict_edges = self.epi_store.get_conflict_edges(entity)
        if conflict_edges:
            history = []
            for e in conflict_edges:
                # 通过 refer_ids → Semantic 拿内容
                for mid in e.get("refer_ids", []):
                    mem = self.sem_store.get_memory(mid)
                    if mem:
                        history.append({
                            "speaker": mem.get("speaker") or mem.get("speaker_id"),
                            "value": mem.get("text"),
                            "timestamp": mem.get("timestamp"),
                            "mem_id": mid,
                        })
            latest = history[-1]["value"] if history else None
            return {
                "verdict": "conflict",
                "history": history,
                "latest_value": latest,
                "edges": conflict_edges,
            }

        # 2. Temporal 边（同人改口）
        temporal_edges = self.epi_store.get_temporal_edges(entity)
        if temporal_edges:
            history = []
            for e in temporal_edges:
                for mid in e.get("refer_ids", []):
                    mem = self.sem_store.get_memory(mid)
                    if mem:
                        history.append({
                            "speaker": mem.get("speaker") or mem.get("speaker_id"),
                            "value": mem.get("text"),
                            "timestamp": mem.get("timestamp"),
                            "mem_id": mid,
                        })
            latest = history[-1]["value"] if history else None
            return {
                "verdict": "update",
                "history": history,
                "latest_value": latest,
                "edges": temporal_edges,
            }

        # 3. Relation 一致性
        rel_edges = self.epi_store.get_relation_edges(entity, relation)
        values = [e.get("relation") for e in rel_edges]
        if len(set(values)) <= 1:
            return {
                "verdict": "consistent",
                "history": [],
                "latest_value": values[0] if values else None,
                "edges": rel_edges,
            }
        # 有多个不同值但没标 conflict（理论上不该，兜底）
        return {
            "verdict": "conflict",
            "history": [],
            "latest_value": None,
            "edges": rel_edges,
        }

    # ── Tool 5: fetch_raw ───────────────────────────────────

    def fetch_raw(self, mem_id: str,
                  image: bool = False, audio: bool = False,
                  text: bool = False) -> dict:
        """按需拉原始证据。mem_id 带前缀判层，沿 refer_ids 下钻到 Raw。"""
        layer = mem_id.split(":")[0]

        # 沿 refer_ids 找到 Raw turn_id
        turn_ids = self._resolve_to_raw_turn_ids(mem_id, layer)
        if not turn_ids:
            return {"fetched": [], "not_found": [mem_id]}

        turns = self.raw_store.fetch_by_turn_ids(turn_ids)
        fetched = []
        for t in turns:
            item = {"turn_id": t["turn_id"]}
            if text:
                item["text"] = t.get("text")
            if audio:
                item["audio_path"] = t.get("audio_path")
            if image:
                item["image_paths"] = t.get("image_paths", [])
            if not (text or audio or image):
                # 默认全字段
                item.update({k: v for k, v in t.items() if k != "turn_id"})
            fetched.append(item)
        return {"fetched": fetched, "not_found": []}

    def _resolve_to_raw_turn_ids(self, mem_id: str, layer: str) -> list[str]:
        """沿 refer_ids 递归到 Raw turn_id。"""
        if layer == "raw":
            return [mem_id]
        if layer == "sem":
            return self.sem_store.get_refer_ids(mem_id)
        if layer == "epi":
            # episodic 边 → refer_ids → sem → raw
            # 从 edges 表取 refer_ids
            import sqlite3
            with sqlite3.connect(config.EPISODIC_DB_PATH) as c:
                c.row_factory = sqlite3.Row
                row = c.execute("SELECT refer_ids FROM edges WHERE mem_id=?", (mem_id,)).fetchone()
                if not row:
                    return []
                import json
                sem_ids = json.loads(row["refer_ids"] or "[]")
                turn_ids = []
                for sid in sem_ids:
                    turn_ids.extend(self.sem_store.get_refer_ids(sid))
                return list(set(turn_ids))
        return []

    # ── Tool 6: search_by_caption（V2 新增）─────────────────

    def search_by_caption(self, query: str,
                          session_filter: str = None,
                          time_range: tuple = None,
                          top_k: int = None) -> list[dict]:
        """V2 新增：只查 caption_index（图描述向量）。

        route 专项工具：当问题是纯图像内容问题（"图里有什么"、
        "图片中的物体"），用 query 文本查 caption 路，可能比 unified 路更精准
        （unified 里 caption 被对话文本稀释）。
        """
        top_k = top_k or config.TOP_K_DEFAULT
        query_emb = text_encode(query)

        results = self.sem_store.search_by_emb(
            self.sem_store.caption_index, self.sem_store.caption_ids, query_emb,
            top_k=top_k * 2,
            filter_kwargs={"session_id": session_filter, "time_range": time_range}
        )
        for m in results[:top_k]:
            self.sem_store.increment_access(m["mem_id"])
        return [self._to_retrieved(m) for m in results[:top_k]]

    # ── Tool 6b: search_by_image_id（V2 新增）──────────────────

    def search_by_image_id(self, image_id: str, top_k: int = 10) -> list[dict]:
        """按 image_id（如 'D2:IMG_001'）精确查含该图的整条 turn 记忆。

        SQL LIKE 查 image_captions 的 key，返回整条 semantic memory（user+
        assistant 合并的整 turn）。用于"问某张图是什么"的 VR 题——语义检索召不到
        时按 image_id 直接查表兜底，保证含该图的 turn 一定进 context。
        """
        if not image_id:
            return []
        results = self.sem_store.search_by_image_id(image_id, top_k=top_k)
        for m in results:
            self.sem_store.increment_access(m["mem_id"])
        return [self._to_retrieved(m) for m in results]

    # ── Tool 7: search_bm25（V2 新增）────────────────────────

    def search_bm25(self, query: str, top_k: int = None) -> list[dict]:
        """BM25 稀有词精确匹配检索。

        语义检索（unified embedding）对全库只出现1次的稀有标签召回失败
        （分数被通用词稀释）。BM25 基于精确词匹配 + IDF 加权，稀有词 IDF 高，
        能保证稀有标签命中。

        route 专项工具：当问题包含专有名词、技术术语、稀有标签时，
        BM25 可能比语义检索更精准。
        """
        top_k = top_k or config.TOP_K_DEFAULT
        self._ensure_bm25()
        hits = self.bm25.search(query, top_k=top_k)
        if not hits:
            return []
        # 按 BM25 score 取对应 mem，归一化 score 到 [0,1]
        max_score = hits[0][1] if hits else 1.0
        results = []
        for mem_id, score in hits:
            mem = self.sem_store.get_memory(mem_id)
            if mem:
                mem["score"] = min(score / max_score, 1.0) if max_score > 0 else 0.0
                results.append(mem)
        for m in results[:top_k]:
            self.sem_store.increment_access(m["mem_id"])
        return [self._to_retrieved(m) for m in results[:top_k]]
