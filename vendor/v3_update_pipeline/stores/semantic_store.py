"""
Semantic Store V2 — SQLite(元数据) + faiss(向量检索)

V2 改动（相对 v1）：
  4路 emb → 2路：
    unified_emb (Qwen3-VL-Embedding, 2048) — 对话文本 + caption 拼接（LTMemory 式单路）
    caption_emb (Qwen3-VL-Embedding, 2048) — 图描述单独（search_by_caption 用）

  删除：text_emb / image_emb / text_clip_emb
  - text_emb 被 unified_emb 替代（unified 含 text+caption 拼接）
  - image_emb 从不被 search_memory 读取（浪费）
  - text_clip_emb 和 text_emb 完全重复
"""
import json
import os
import sqlite3
import time
import numpy as np
from pathlib import Path

import config

try:
    import faiss
except ImportError:
    faiss = None


class SemanticStore:
    def __init__(self, db_path: str | Path = None):
        self.db_path = str(db_path or config.SEMANTIC_DB_PATH)
        self._init_db()
        self._init_faiss()

    # ── SQLite 元数据 ──────────────────────────────────────

    def _conn(self):
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS semantic_memory (
                    mem_id            TEXT PRIMARY KEY,
                    dataset           TEXT,
                    conv_id           TEXT,
                    speaker_id        TEXT,
                    speaker           TEXT,
                    addressee         TEXT,
                    memory_type       TEXT,
                    modalities        TEXT,
                    text              TEXT,
                    summary           TEXT,
                    entities          TEXT,
                    image_captions    TEXT,
                    image_paths       TEXT,
                    session_id        TEXT,
                    timestamp         INTEGER,
                    date              TEXT,
                    refer_ids         TEXT,
                    is_superseded     INTEGER DEFAULT 0,
                    superseded_by     TEXT,
                    supersede_reason  TEXT,
                    importance        REAL DEFAULT 0.5,
                    access_count      INTEGER DEFAULT 0,
                    turn_seq          INTEGER
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_sem_speaker ON semantic_memory(speaker_id)")
            # 旧库迁移：无 turn_seq 列则 ALTER 补上（_row_to_dict 对缺失列自然返回 None）
            cols = {r["name"] for r in c.execute("PRAGMA table_info(semantic_memory)")}
            if "turn_seq" not in cols:
                c.execute("ALTER TABLE semantic_memory ADD COLUMN turn_seq INTEGER")
            c.execute("CREATE INDEX IF NOT EXISTS idx_sem_session ON semantic_memory(session_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_sem_superseded ON semantic_memory(is_superseded)")

    # ── faiss 索引 ─────────────────────────────────────────

    def _init_faiss(self):
        if faiss is None:
            raise ImportError("faiss not installed. pip install faiss-cpu")
        self.unified_index = faiss.IndexFlatIP(config.UNIFIED_EMB_DIM)
        self.caption_index = faiss.IndexFlatIP(config.CAPTION_EMB_DIM)
        # V3: fact 层单独索引（原子事实，和 raw turn 分开，避免稀释）
        self.fact_index = faiss.IndexFlatIP(config.UNIFIED_EMB_DIM)
        # V3: profile 层单独索引（用户长期画像，和 raw/fact 分开，search_profile 用）
        self.profile_index = faiss.IndexFlatIP(config.UNIFIED_EMB_DIM)
        # mem_id 列表（和 faiss 索引顺序对齐）
        self.unified_ids: list[str] = []
        self.caption_ids: list[str] = []
        self.fact_ids: list[str] = []
        self.profile_ids: list[str] = []
        self._add_count = 0
        # 启动时从文件加载索引（持久化）
        self._load_indexes()

    def flush(self):
        """主动存盘（ingest 结束时调）。"""
        self._save_indexes()

    def _add_to_index(self, index, ids_list: list, mem_id: str, emb: np.ndarray):
        """把向量加入 faiss 索引 + 记录 mem_id（批量 save，不每次写文件）。"""
        if emb is None:
            return
        vec = np.array(emb, dtype=np.float32).reshape(1, -1)
        index.add(vec)
        ids_list.append(mem_id)
        self._add_count += 1
        if self._add_count % 10 == 0:  # 每10条存一次
            self._save_indexes()

    def _save_indexes(self):
        """把 faiss 索引 + mem_id 列表存到文件（持久化）。"""
        if faiss is None:
            return
        if self.unified_index.ntotal > 0:
            faiss.write_index(self.unified_index, str(config.UNIFIED_INDEX_PATH))
        if self.caption_index.ntotal > 0:
            faiss.write_index(self.caption_index, str(config.CAPTION_INDEX_PATH))
        if self.fact_index.ntotal > 0:
            faiss.write_index(self.fact_index, str(config.FACT_INDEX_PATH))
        if self.profile_index.ntotal > 0:
            faiss.write_index(self.profile_index, str(config.PROFILE_INDEX_PATH))
        # mem_id 列表存 SQLite（faiss 不存 id，要单独存）
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS faiss_ids ("
                      "idx_name TEXT PRIMARY KEY, mem_ids TEXT)")
            c.execute("INSERT OR REPLACE INTO faiss_ids VALUES (?,?)",
                      ("unified", json.dumps(self.unified_ids)))
            c.execute("INSERT OR REPLACE INTO faiss_ids VALUES (?,?)",
                      ("caption", json.dumps(self.caption_ids)))
            c.execute("INSERT OR REPLACE INTO faiss_ids VALUES (?,?)",
                      ("fact", json.dumps(self.fact_ids)))
            c.execute("INSERT OR REPLACE INTO faiss_ids VALUES (?,?)",
                      ("profile", json.dumps(self.profile_ids)))

    def _load_indexes(self):
        """从文件加载 faiss 索引 + mem_id 列表（重启不丢）。

        dolphinfs（网络FS）上 SQLite 读偶发瞬时失败。原代码 except:pass 会静默吞掉
        → ids 空 → 检索全0 → 产出垃圾分数无告警。改为：重试5次，仍失败或读到
        0行(但DB有数据)则报错退出，绝不静默产出垃圾。
        """
        if faiss is None:
            return
        # 加载 mem_id 列表（重试，应对 dolphinfs 瞬时读失败）
        id_map = {}
        last_err = None
        for attempt in range(5):
            try:
                with self._conn() as c:
                    c.execute("CREATE TABLE IF NOT EXISTS faiss_ids ("
                              "idx_name TEXT PRIMARY KEY, mem_ids TEXT)")
                    rows = c.execute("SELECT idx_name, mem_ids FROM faiss_ids").fetchall()
                    id_map = {r[0]: json.loads(r[1]) for r in rows}
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"  [warn] faiss_ids read attempt {attempt+1} failed: {e}", flush=True)
                time.sleep(0.5 * (attempt + 1))
        if last_err is not None:
            raise RuntimeError(
                f"[FATAL] faiss_ids 读取5次全失败 (dolphinfs?): {last_err}. "
                f"不静默产出垃圾, 整个程序退出.")
        self.unified_ids = id_map.get("unified", [])
        self.caption_ids = id_map.get("caption", [])
        self.fact_ids = id_map.get("fact", [])
        self.profile_ids = id_map.get("profile", [])
        # sanity: 若 unified+fact+profile 都空但 semantic_memory 有非 superseded 行 → load 异常，报错退出
        with self._conn() as c:
            n_rows = c.execute(
                "SELECT COUNT(*) FROM semantic_memory WHERE is_superseded=0"
            ).fetchone()[0]
        if n_rows > 0 and not self.unified_ids and not self.fact_ids and not self.profile_ids:
            raise RuntimeError(
                f"[FATAL] memory load 异常: semantic_memory 有 {n_rows} 行但 unified/fact/profile_ids 都空 "
                f"(dolphinfs 读失败 / faiss_ids 未持久化). 不静默产出垃圾, 整个程序退出.")
        # 加载 faiss 索引文件
        if config.UNIFIED_INDEX_PATH.exists():
            self.unified_index = faiss.read_index(str(config.UNIFIED_INDEX_PATH))
        if config.CAPTION_INDEX_PATH.exists():
            self.caption_index = faiss.read_index(str(config.CAPTION_INDEX_PATH))
        if config.FACT_INDEX_PATH.exists():
            self.fact_index = faiss.read_index(str(config.FACT_INDEX_PATH))
        if config.PROFILE_INDEX_PATH.exists():
            self.profile_index = faiss.read_index(str(config.PROFILE_INDEX_PATH))

    # ── 写入 ───────────────────────────────────────────────

    def add_memory(self, mem: dict,
                   unified_emb: np.ndarray = None,
                   caption_emb: np.ndarray = None) -> str:
        """添加一条 Semantic 记忆 + 各 emb。返回 mem_id。

        V2: 2路 emb（unified + caption），替代 v1 的 4路。
        """
        mem_id = mem["mem_id"]
        with self._conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO semantic_memory
                (mem_id, dataset, conv_id, speaker_id, speaker, addressee,
                 memory_type, modalities, text, summary, entities,
                 image_captions, image_paths, session_id, timestamp, date,
                 refer_ids, is_superseded, superseded_by, supersede_reason,
                 importance, access_count, turn_seq)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                mem_id, mem.get("dataset"), mem.get("conv_id"),
                mem.get("speaker_id"), mem.get("speaker"), mem.get("addressee"),
                mem.get("memory_type"), json.dumps(mem.get("modalities", [])),
                mem.get("text"), mem.get("summary"),
                json.dumps(mem.get("entities", [])),
                json.dumps(mem.get("image_captions", {})),
                json.dumps(mem.get("image_paths", [])),
                mem.get("session_id"), mem.get("timestamp"),
                mem.get("date", ""),
                json.dumps(mem.get("refer_ids", [])),
                mem.get("is_superseded", 0),
                mem.get("superseded_by"),
                mem.get("supersede_reason"),
                mem.get("importance", 0.5),
                mem.get("access_count", 0),
                mem.get("turn_seq"),
            ))
        # 加入 faiss（fact/profile 层单独进各自 index，不进 unified_index，避免稀释 raw turn）
        # 分开检索：fact 只进 fact_index（search_fact 用），raw 只进 unified_index（search_memory 用）。
        mt = mem.get("memory_type")
        if mt == "fact":
            self._add_to_index(self.fact_index, self.fact_ids, mem_id, unified_emb)
        elif mt == "profile":
            self._add_to_index(self.profile_index, self.profile_ids, mem_id, unified_emb)
        else:
            self._add_to_index(self.unified_index, self.unified_ids, mem_id, unified_emb)
        # caption_index 只放有图记忆（fact/profile 的 caption_emb 为 None，天然跳过）
        self._add_to_index(self.caption_index, self.caption_ids, mem_id, caption_emb)
        return mem_id

    def supersede(self, old_mem_id: str, new_mem_id: str, reason: str):
        """标记旧记忆被覆盖。"""
        with self._conn() as c:
            c.execute(
                "UPDATE semantic_memory SET is_superseded=1, superseded_by=?, supersede_reason=? "
                "WHERE mem_id=?",
                (new_mem_id, reason, old_mem_id)
            )

    def increment_access(self, mem_id: str):
        with self._conn() as c:
            c.execute(
                "UPDATE semantic_memory SET access_count=access_count+1 WHERE mem_id=?",
                (mem_id,)
            )

    # ── 查询 ────────────────────────────────────────────────

    def get_memory(self, mem_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM semantic_memory WHERE mem_id=?", (mem_id,)).fetchone()
            return self._row_to_dict(row)

    def get_memories_by_ids(self, mem_ids: list[str]) -> list[dict]:
        if not mem_ids:
            return []
        placeholders = ",".join("?" * len(mem_ids))
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM semantic_memory WHERE mem_id IN ({placeholders})",
                mem_ids
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_profiles_by_speaker(self, speaker_id: str = None, top_k: int = 100) -> list[dict]:
        """按 speaker name 查该人的 profile 画像节点（所有字段，SQL 精确查表）。

        profile 设计：每个人一个节点（多条字段记忆，speaker_id=name）。
        检索时按 name 精确查表，返回该人的全部画像字段，不做语义召回。

        speaker_id 为空时返回【全部 speaker 的画像】（作为全局背景，让答题模型
        知道每个人是什么人/喜欢什么），按 speaker + time 排序。
        """
        with self._conn() as c:
            if not speaker_id:
                rows = c.execute(
                    "SELECT * FROM semantic_memory "
                    "WHERE memory_type='profile' AND is_superseded=0 "
                    "ORDER BY speaker_id ASC, timestamp ASC"
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM semantic_memory "
                    "WHERE speaker_id=? AND memory_type='profile' AND is_superseded=0 "
                    "ORDER BY timestamp ASC",
                    (speaker_id,)
                ).fetchall()
        results = [self._row_to_dict(r) for r in rows if self._row_to_dict(r)]
        for d in results:
            d["score"] = 1.0  # 精确匹配，给满分
        return results[:top_k]

    def search_by_image_id(self, image_id: str, top_k: int = 10) -> list[dict]:
        """按 image_id（如 'D2:IMG_001'）精确查含该图的整条 turn 记忆。

        image_captions 是 JSON TEXT，key 为 image_id。用 LIKE 匹配，返回整条
        semantic memory（user+assistant 合并的整 turn）。用于"问某张图是什么"
        的 VR 题——语义检索召不到时，按 image_id 直接查表兜底。
        """
        if not image_id:
            return []
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM semantic_memory "
                "WHERE is_superseded=0 AND image_captions LIKE ? "
                "ORDER BY timestamp ASC",
                (f"%{image_id}%",)
            ).fetchall()
        results = [self._row_to_dict(r) for r in rows if self._row_to_dict(r)]
        for d in results:
            d["score"] = 1.0  # 精确匹配，给满分
        return results[:top_k]

    def _build_filter(self, speaker_id=None, addressee=None,
                      session_id=None, time_range=None,
                      include_history=False) -> tuple[str, list]:
        """构建 SQL 过滤条件。"""
        parts = []
        args = []
        if not include_history:
            parts.append("is_superseded=0")
        if speaker_id:
            parts.append("speaker_id=?")
            args.append(speaker_id)
        if addressee:
            parts.append("addressee=?")
            args.append(addressee)
        if session_id:
            parts.append("session_id=?")
            args.append(session_id)
        if time_range:
            parts.append("timestamp>=? AND timestamp<=?")
            args.extend(time_range)
        where = " AND ".join(parts) if parts else "1=1"
        return where, args

    def search_by_emb(self, index, ids_list: list, query_emb: np.ndarray,
                      top_k: int = 5, filter_kwargs: dict = None) -> list[dict]:
        """向量检索 + 元数据过滤。filter_kwargs 见 _build_filter。"""
        if len(ids_list) == 0:
            return []
        vec = np.array(query_emb, dtype=np.float32).reshape(1, -1)
        scores, indices = index.search(vec, min(top_k * 3, len(ids_list)))
        # 取 mem_id + 分数
        hit_ids = []
        score_map = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(ids_list):  # 越界保护
                continue
            mid = ids_list[idx]
            hit_ids.append(mid)
            score_map[mid] = float(score)
        # 过滤
        where, args = self._build_filter(**(filter_kwargs or {}))
        if not hit_ids:
            return []
        placeholders = ",".join("?" * len(hit_ids))
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM semantic_memory WHERE mem_id IN ({placeholders}) AND {where}",
                hit_ids + args
            ).fetchall()
        results = []
        for r in rows:
            d = self._row_to_dict(r)
            d["score"] = score_map.get(d["mem_id"], 0.0)
            results.append(d)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_refer_ids(self, mem_id: str) -> list[str]:
        """取一条记忆的 refer_ids（→ Raw turn_id）。"""
        mem = self.get_memory(mem_id)
        if not mem:
            return []
        return mem.get("refer_ids", [])

    def _row_to_dict(self, row) -> dict | None:
        if row is None:
            return None
        d = dict(row)
        d["modalities"] = json.loads(d.get("modalities") or "[]")
        d["entities"] = json.loads(d.get("entities") or "[]")
        d["image_captions"] = json.loads(d.get("image_captions") or "{}")
        d["image_paths"] = json.loads(d.get("image_paths") or "[]")
        d["refer_ids"] = json.loads(d.get("refer_ids") or "[]")
        return d
