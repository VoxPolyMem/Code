"""
BM25 Index — 稀有标签硬召回（精确匹配 + IDF 加权）

bge-m3 语义检索对全库只出现1次的稀有标签（如 'Focus Window', 'Trend Recovery',
'Alley Oop'）召回失败（分数被通用词稀释）。BM25 基于精确词匹配 + IDF 加权，
稀有词 IDF 高，能保证稀有标签命中。

索引基于 text + 所有 caption 全文（拼接），保证多图记忆的稀有标签在任意 caption 里都能命中。
"""
import math
import re
import json
import time
from collections import Counter, defaultdict


def tokenize(text: str) -> list:
    """英文分词：小写 + 去标点 + 按空格分。"""
    if not text:
        return []
    text = text.lower()
    return re.findall(r'[a-z0-9]+', text)


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []          # list of token lists
        self.mem_ids = []         # 对应 mem_id
        self.doc_lens = []
        self.avgdl = 0.0
        self.df = defaultdict(int)  # 文档频率
        self.idf = {}
        self.tf = []              # 每篇文档的 Counter
        self._dirty = False

    def add_doc(self, text: str, mem_id: str):
        tokens = tokenize(text)
        self.corpus.append(tokens)
        self.mem_ids.append(mem_id)
        self.doc_lens.append(len(tokens))
        counter = Counter(tokens)
        self.tf.append(counter)
        for word in counter:
            self.df[word] += 1
        self._dirty = True

    def _update_idf(self):
        N = len(self.corpus)
        self.avgdl = sum(self.doc_lens) / N if N else 0
        self.idf = {}
        for word, df in self.df.items():
            # BM25+ IDF（保证非负）
            self.idf[word] = math.log((N - df + 0.5) / (df + 0.5) + 1)
        self._dirty = False

    def search(self, query: str, top_k: int = 10) -> list:
        """返回 [(mem_id, score)]，按 score 降序。只返回 score>0 的。"""
        if self._dirty:
            self._update_idf()
        if not self.corpus:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = []
        avgdl = self.avgdl or 1
        for i, tf in enumerate(self.tf):
            score = 0.0
            dl = self.doc_lens[i]
            for q in query_tokens:
                f = tf.get(q, 0)
                if f == 0:
                    continue
                idf = self.idf.get(q, 0)
                score += idf * (f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avgdl)))
            if score > 0:
                scores.append((self.mem_ids[i], score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def build_from_store(self, sem_store) -> int:
        """从 SemanticStore 加载所有记忆的 text + caption，建索引。返回文档数。

        dolphinfs 上 SQLite 读偶发瞬时失败，重试5次，仍0行但DB有数据则报错退出（不静默）。
        """
        rows = None
        last_err = None
        for attempt in range(5):
            try:
                with sem_store._conn() as c:
                    rows = c.execute(
                        "SELECT mem_id, text, image_captions FROM semantic_memory WHERE is_superseded=0"
                    ).fetchall()
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"  [warn] BM25 read attempt {attempt+1} failed: {e}", flush=True)
                time.sleep(0.5 * (attempt + 1))
        if last_err is not None or rows is None:
            raise RuntimeError(
                f"[FATAL] BM25 build 读取5次全失败 (dolphinfs?): {last_err}. "
                f"不静默产出垃圾, 整个程序退出.")
        # sanity: rows 为空但 semantic_memory 实际有行 → 读失败，报错退出
        if not rows:
            with sem_store._conn() as c:
                n_rows = c.execute(
                    "SELECT COUNT(*) FROM semantic_memory WHERE is_superseded=0"
                ).fetchone()[0]
            if n_rows > 0:
                raise RuntimeError(
                    f"[FATAL] BM25 build 异常: semantic_memory 有 {n_rows} 行但读到0行 "
                    f"(dolphinfs 读失败). 不静默产出垃圾, 整个程序退出.")
        for row in rows:
            mem_id = row["mem_id"]
            text = row["text"] or ""
            caps_raw = row["image_captions"] or "{}"
            doc = text
            try:
                caps_dict = json.loads(caps_raw) if isinstance(caps_raw, str) else caps_raw
                if isinstance(caps_dict, dict):
                    for cap_text in caps_dict.values():
                        if cap_text:
                            doc += " " + cap_text
            except Exception:
                pass
            self.add_doc(doc, mem_id)
        self._update_idf()
        return len(self.corpus)
