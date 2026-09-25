"""
Raw Store — SQLite，存原始 turn（永不删，只追加）
turn_id 格式：raw:{conv}:{session}:{seq}
"""
import json
import sqlite3
from pathlib import Path

import config


class RawStore:
    def __init__(self, db_path: str | Path = None):
        self.db_path = str(db_path or config.RAW_DB_PATH)
        self._init_db()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS raw_turns (
                    turn_id         TEXT PRIMARY KEY,
                    session_id      TEXT NOT NULL,
                    seq_in_session  INTEGER NOT NULL,
                    speaker_id      TEXT,
                    speaker         TEXT,
                    text            TEXT,
                    audio_path      TEXT,
                    image_paths     TEXT,
                    timestamp       INTEGER,
                    dataset         TEXT,
                    conv_id         TEXT,
                    referrers       TEXT DEFAULT '[]'
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_raw_session ON raw_turns(session_id)")

    def add_turn(self, turn_id: str, session_id: str, seq: int,
                 text: str = None, audio_path: str = None,
                 image_paths: list = None, timestamp: int = None,
                 dataset: str = None, conv_id: str = None,
                 speaker_id: str = None, speaker: str = None) -> str:
        """添加原始 turn（只追加）。返回 turn_id。"""
        with self._conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO raw_turns
                (turn_id, session_id, seq_in_session, speaker_id, speaker,
                 text, audio_path, image_paths, timestamp, dataset, conv_id, referrers)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '[]')
            """, (turn_id, session_id, seq, speaker_id, speaker,
                  text, audio_path,
                  json.dumps(image_paths or []), timestamp, dataset, conv_id))
        return turn_id

    def get_turn(self, turn_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM raw_turns WHERE turn_id=?", (turn_id,)).fetchone()
            return self._row_to_dict(row)

    def get_turns_by_session(self, session_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM raw_turns WHERE session_id=? ORDER BY seq_in_session",
                (session_id,)
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_turns_between(self, session_id: str, start_seq: int, end_seq: int) -> list[dict]:
        """取 session 内 seq 在 [start_seq, end_seq] 的 turn，按 seq 升序。

        供邻居扩窗用：检索命中某 turn 后，把 ±window 相邻 turn 拉出来打包成对话片段。
        """
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM raw_turns WHERE session_id=? "
                "AND seq_in_session BETWEEN ? AND ? ORDER BY seq_in_session",
                (session_id, start_seq, end_seq)
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def fetch_by_turn_ids(self, turn_ids: list[str]) -> list[dict]:
        """按 turn_id 列表批量取（fetch_raw 用）。"""
        if not turn_ids:
            return []
        placeholders = ",".join("?" * len(turn_ids))
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM raw_turns WHERE turn_id IN ({placeholders})",
                turn_ids
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def update_speaker(self, turn_id: str, speaker_id: str, speaker: str):
        """声纹/LLM 识别后回填 speaker。"""
        with self._conn() as c:
            c.execute(
                "UPDATE raw_turns SET speaker_id=?, speaker=? WHERE turn_id=?",
                (speaker_id, speaker, turn_id)
            )

    def add_referrer(self, turn_id: str, sem_mem_id: str):
        """添加反向索引（哪些 Semantic 引用本 turn）。"""
        turn = self.get_turn(turn_id)
        if not turn:
            return
        refs = turn.get("referrers", [])
        if not isinstance(refs, list):
            refs = json.loads(refs) if refs else []
        if sem_mem_id not in refs:
            refs.append(sem_mem_id)
            with self._conn() as c:
                c.execute(
                    "UPDATE raw_turns SET referrers=? WHERE turn_id=?",
                    (json.dumps(refs), turn_id)
                )

    def _row_to_dict(self, row) -> dict | None:
        if row is None:
            return None
        d = dict(row)
        d["image_paths"] = json.loads(d.get("image_paths") or "[]")
        d["referrers"] = json.loads(d.get("referrers") or "[]")
        return d
