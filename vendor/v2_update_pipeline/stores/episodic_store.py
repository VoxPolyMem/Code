"""
Episodic Store — SQLite 图（nodes + edges）

节点（无 mem_id，纯锚点）：Entity / Speaker
边（有 mem_id epi:<id>，是 episodic 记忆单元）：
  Relation / Temporal / Conflict
边只挂 refer_ids（→Semantic mem_id），不存内容。
"""
import json
import sqlite3
from pathlib import Path

import config


class EpisodicStore:
    def __init__(self, db_path: str | Path = None):
        self.db_path = str(db_path or config.EPISODIC_DB_PATH)
        self._init_db()

    def _conn(self):
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self):
        with self._conn() as c:
            # 节点（无 mem_id，node_id 是图内部标识）
            c.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id     TEXT PRIMARY KEY,
                    node_type   TEXT,
                    name        TEXT,
                    entity_type TEXT,
                    mem_ids     TEXT DEFAULT '[]',
                    first_seen  INTEGER,
                    last_seen   INTEGER
                )
            """)
            # 边（有 mem_id epi:<id>，refer_ids→Semantic）
            c.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    edge_id         TEXT PRIMARY KEY,
                    mem_id          TEXT,
                    edge_type       TEXT,
                    from_node       TEXT,
                    to_node         TEXT,
                    relation        TEXT,
                    speaker_id      TEXT,
                    speaker_a       TEXT,
                    speaker_b       TEXT,
                    timestamp       INTEGER,
                    timestamp_from  INTEGER,
                    timestamp_to    INTEGER,
                    refer_ids       TEXT DEFAULT '[]',
                    weight          REAL DEFAULT 1.0,
                    is_active       INTEGER DEFAULT 1
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_edge_type ON edges(edge_type)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_edge_active ON edges(is_active)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_node_name ON nodes(name)")

    # ── 节点 ────────────────────────────────────────────────

    def upsert_node(self, node_id: str, node_type: str, name: str,
                    entity_type: str = None, mem_id: str = None,
                    timestamp: int = None):
        """新建或更新节点（追加 mem_id）。"""
        with self._conn() as c:
            row = c.execute("SELECT * FROM nodes WHERE node_id=?", (node_id,)).fetchone()
            if row:
                mem_ids = json.loads(row["mem_ids"] or "[]")
                if mem_id and mem_id not in mem_ids:
                    mem_ids.append(mem_id)
                c.execute(
                    "UPDATE nodes SET mem_ids=?, last_seen=? WHERE node_id=?",
                    (json.dumps(mem_ids), timestamp, node_id)
                )
            else:
                c.execute("""
                    INSERT INTO nodes (node_id, node_type, name, entity_type, mem_ids, first_seen, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (node_id, node_type, name, entity_type,
                      json.dumps([mem_id] if mem_id else []), timestamp, timestamp))

    def find_node_by_name(self, name: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM nodes WHERE name=?", (name,)).fetchone()
            return dict(row) if row else None

    # ── 边 ──────────────────────────────────────────────────

    def add_edge(self, edge: dict) -> str:
        """添加边。edge 含 mem_id(edge epi:<id>), edge_type, refer_ids 等。"""
        edge_id = edge["mem_id"]  # 边用 mem_id 作 edge_id
        with self._conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO edges
                (edge_id, mem_id, edge_type, from_node, to_node, relation,
                 speaker_id, speaker_a, speaker_b, timestamp,
                 timestamp_from, timestamp_to, refer_ids, weight, is_active)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                edge_id, edge["mem_id"], edge["edge_type"],
                edge.get("from_node"), edge.get("to_node"), edge.get("relation"),
                edge.get("speaker_id"), edge.get("speaker_a"), edge.get("speaker_b"),
                edge.get("timestamp"),
                edge.get("timestamp_from"), edge.get("timestamp_to"),
                json.dumps(edge.get("refer_ids", [])),
                edge.get("weight", 1.0), edge.get("is_active", 1)
            ))
        return edge_id

    def deactivate_edge(self, edge_id: str):
        """标记边失效（update 时旧边）。"""
        with self._conn() as c:
            c.execute("UPDATE edges SET is_active=0 WHERE edge_id=?", (edge_id,))

    def get_conflict_edges(self, entity_name: str) -> list[dict]:
        """查某实体的 Conflict 边（两人矛盾）。"""
        with self._conn() as c:
            # 找该实体的节点，再找关联的 conflict 边
            node = c.execute("SELECT * FROM nodes WHERE name=?", (entity_name,)).fetchone()
            if not node:
                return []
            rows = c.execute(
                "SELECT * FROM edges WHERE edge_type='conflict' AND is_active=1 "
                "AND (from_node=? OR to_node=?)",
                (node["node_id"], node["node_id"])
            ).fetchall()
            return [self._edge_row_to_dict(r) for r in rows]

    def get_temporal_edges(self, entity_name: str) -> list[dict]:
        """查某实体的 Temporal 边（同人改口）。"""
        with self._conn() as c:
            node = c.execute("SELECT * FROM nodes WHERE name=?", (entity_name,)).fetchone()
            if not node:
                return []
            rows = c.execute(
                "SELECT * FROM edges WHERE edge_type='temporal' "
                "AND (from_node=? OR to_node=?) ORDER BY timestamp",
                (node["node_id"], node["node_id"])
            ).fetchall()
            return [self._edge_row_to_dict(r) for r in rows]

    def get_relation_edges(self, entity_name: str, relation: str = None) -> list[dict]:
        """查某实体的 active Relation 边。"""
        with self._conn() as c:
            node = c.execute("SELECT * FROM nodes WHERE name=?", (entity_name,)).fetchone()
            if not node:
                return []
            if relation:
                rows = c.execute(
                    "SELECT * FROM edges WHERE edge_type='relation' AND is_active=1 "
                    "AND relation=? AND (from_node=? OR to_node=?)",
                    (relation, node["node_id"], node["node_id"])
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM edges WHERE edge_type='relation' AND is_active=1 "
                    "AND (from_node=? OR to_node=?)",
                    (node["node_id"], node["node_id"])
                ).fetchall()
            return [self._edge_row_to_dict(r) for r in rows]

    def _edge_row_to_dict(self, row) -> dict:
        d = dict(row)
        d["refer_ids"] = json.loads(d.get("refer_ids") or "[]")
        return d
