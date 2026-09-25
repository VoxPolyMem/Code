"""
Speaker Registry V2 — SQLite（speaker 档案）+ 声纹向量 (ECAPA + EMA)

扩展自 pertype_v2 的 registry.py，增加：
  - voiceprint: ECAPA 192维 embedding, EMA 维护
  - identify_by_voiceprint(): 声纹识别 (两段式阈值)
  - ema_update(): EMA 声纹进化 (带门限保护)
  - apply_corrections(): 误判纠正 (回溯重评估 + 多句一致性分裂)

向后兼容：原有 register/identify/resolve_speaker_id/get/all_speakers 不变。
"""
import json
import sqlite3
import logging
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import config

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    faiss = None


@dataclass
class MatchResult:
    """声纹匹配结果。"""
    spk_id: str
    sim: float
    confidence: str  # high / low
    is_new: bool


class SpeakerRegistry:
    """
    说话人注册表：管理 speaker 档案 + 声纹向量 (EMA 进化)。

    声纹识别: identify_by_voiceprint(emb) → MatchResult
    LLM name: identify(spk_id, name, role) → 填 name (原有逻辑)
    """

    def __init__(self, db_path: str | Path = None):
        self.db_path = str(db_path or config.REGISTRY_DB_PATH)
        self._init_db()

        # 声纹参数
        self.recognition_threshold = getattr(config, "VOICEPRINT_RECOGNITION_THRESHOLD", 0.50)
        self.ema_gate = getattr(config, "VOICEPRINT_EMA_GATE", 0.40)
        self.ema_alpha = getattr(config, "VOICEPRINT_EMA_ALPHA", 0.3)

        # 内存中的声纹向量 {spk_id: np.ndarray}
        self._voiceprints: dict[str, np.ndarray] = {}
        self._voiceprint_counts: dict[str, int] = defaultdict(int)

        # 匹配历史 (用于纠正)
        self._history: list[dict] = []
        self._pending: dict[str, list] = defaultdict(list)

        self._load_voiceprints()

    # ── SQLite ──────────────────────────────────────

    def _conn(self):
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS speakers (
                    speaker_id        TEXT PRIMARY KEY,
                    name              TEXT,
                    role              TEXT,
                    identified        INTEGER DEFAULT 0,
                    voiceprint_count  INTEGER DEFAULT 0,
                    voiceprint_status TEXT DEFAULT 'confirmed',
                    first_seen        INTEGER,
                    last_seen         INTEGER,
                    sessions          TEXT DEFAULT '[]',
                    dataset           TEXT
                )
            """)
            # 声纹向量存为 BLOB (numpy 序列化)
            c.execute("""
                CREATE TABLE IF NOT EXISTS voiceprints (
                    speaker_id  TEXT PRIMARY KEY,
                    embedding   BLOB,
                    FOREIGN KEY (speaker_id) REFERENCES speakers(speaker_id)
                )
            """)

    def _load_voiceprints(self):
        """从 SQLite 加载声纹向量到内存。"""
        with self._conn() as c:
            rows = c.execute("SELECT speaker_id, embedding FROM voiceprints").fetchall()
            for row in rows:
                if row["embedding"]:
                    emb = np.frombuffer(row["embedding"], dtype=np.float32)
                    self._voiceprints[row["speaker_id"]] = emb
                    self._voiceprint_counts[row["speaker_id"]] = 0

    def _save_voiceprint(self, spk_id: str, emb: np.ndarray):
        """保存声纹向量到 SQLite。"""
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO voiceprints (speaker_id, embedding) VALUES (?, ?)",
                (spk_id, emb.astype(np.float32).tobytes())
            )

    # ── 原有方法 (向后兼容) ──────────────────────────

    def register(self, speaker_id: str, name: str = None, role: str = None,
                 identified: bool = False, first_seen: int = None,
                 sessions: list = None, dataset: str = None,
                 voiceprint: np.ndarray = None) -> str:
        """注册/更新 speaker。"""
        with self._conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO speakers
                (speaker_id, name, role, identified, voiceprint_count,
                 voiceprint_status, first_seen, last_seen, sessions, dataset)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (speaker_id, name, role, int(identified), 0, 'confirmed',
                  first_seen, int(time.time()),
                  json.dumps(sessions or []), dataset))
        if voiceprint is not None:
            self._voiceprints[speaker_id] = voiceprint
            self._save_voiceprint(speaker_id, voiceprint)
        return speaker_id

    def identify(self, speaker_id: str, name: str, role: str = None):
        """LLM 推断出身份后，标记 identified + 填 name。"""
        with self._conn() as c:
            c.execute(
                "UPDATE speakers SET name=?, role=?, identified=1 WHERE speaker_id=?",
                (name, role, speaker_id)
            )
        logger.info(f"Speaker identified: {speaker_id} → name={name}, role={role}")

    def get(self, speaker_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM speakers WHERE speaker_id=?", (speaker_id,)).fetchone()
            if row:
                d = dict(row)
                d["sessions"] = json.loads(d.get("sessions") or "[]")
                return d
            return None

    def resolve_speaker_id(self, name_or_id: str) -> str | None:
        with self._conn() as c:
            row = c.execute("SELECT speaker_id FROM speakers WHERE speaker_id=?",
                            (name_or_id,)).fetchone()
            if row:
                return row["speaker_id"]
            row = c.execute("SELECT speaker_id FROM speakers WHERE name=?",
                            (name_or_id,)).fetchone()
            if row:
                return row["speaker_id"]
            return None

    def all_speakers(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM speakers").fetchall()
            return [dict(r) for r in rows]

    # ── 声纹识别 (新增) ──────────────────────────────

    def identify_by_voiceprint(self, emb: np.ndarray,
                                audio_path: str = "") -> MatchResult:
        """
        用声纹识别说话人。

        两段式阈值:
          sim >= recognition_threshold → 归入 top1 SPK, EMA 更新
          sim < recognition_threshold → 注册新 SPK
        """
        vec = emb / (np.linalg.norm(emb) + 1e-8)

        if not self._voiceprints:
            # 第一个说话人
            spk_id = self._next_spk_id()
            self.register(spk_id, voiceprint=vec)
            result = MatchResult(spk_id=spk_id, sim=1.0, confidence="high", is_new=True)
            self._history.append({"spk_id": spk_id, "vec": vec, "audio_path": audio_path})
            return result

        # 和所有已知 SPK 算 cosine similarity
        sims = [(sid, float(np.dot(vec, vp))) for sid, vp in self._voiceprints.items()]
        sims.sort(key=lambda x: -x[1])
        best_spk, best_sim = sims[0]

        if best_sim >= self.recognition_threshold:
            # 归入已有 SPK
            self._ema_update(best_spk, vec)
            result = MatchResult(spk_id=best_spk, sim=best_sim, confidence="high", is_new=False)
        else:
            # 注册新 SPK
            spk_id = self._next_spk_id()
            self.register(spk_id, voiceprint=vec)
            result = MatchResult(spk_id=spk_id, sim=best_sim, confidence="low", is_new=True)

        self._history.append({"spk_id": result.spk_id, "vec": vec, "audio_path": audio_path})
        return result

    def _next_spk_id(self) -> str:
        """生成下一个 SPK_ID。"""
        with self._conn() as c:
            row = c.execute("SELECT COUNT(*) as n FROM speakers").fetchone()
            return f"SPK_{row['n'] + 1:03d}"

    def _ema_update(self, spk_id: str, new_vec: np.ndarray) -> bool:
        """EMA 声纹进化 (带门限保护)。"""
        old_vec = self._voiceprints.get(spk_id)
        if old_vec is None:
            return False

        sim = float(np.dot(old_vec, new_vec))
        if sim < self.ema_gate:
            logger.debug(f"EMA 拒绝更新 {spk_id}: sim={sim:.3f} < gate={self.ema_gate}")
            return False

        updated = (1 - self.ema_alpha) * old_vec + self.ema_alpha * new_vec
        updated = updated / (np.linalg.norm(updated) + 1e-8)
        self._voiceprints[spk_id] = updated
        self._voiceprint_counts[spk_id] += 1
        self._save_voiceprint(spk_id, updated)

        with self._conn() as c:
            c.execute(
                "UPDATE speakers SET voiceprint_count=voiceprint_count+1, last_seen=? WHERE speaker_id=?",
                (int(time.time()), spk_id)
            )
        return True

    def get_name(self, spk_id: str) -> str | None:
        """获取 SPK 的 name。"""
        info = self.get(spk_id)
        return info.get("name") if info else None

    # ── 误判纠正 (新增) ──────────────────────────────

    def apply_corrections(self) -> list[dict]:
        """
        回溯纠正误判。

        策略 1: 回溯重评估 — 用进化后的 profile 重新评估历史匹配
        策略 2: 多句一致性分裂 — pending 列表内部不一致时分裂新 SPK
        """
        corrected = []
        if len(self._history) < 3:
            return corrected

        # 策略 1: 回溯重评估 (目前两段式没有 pending, 暂跳过)
        # 如果未来加回三段式, 这里可以重新评估

        return corrected
