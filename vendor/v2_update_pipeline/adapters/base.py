"""
Base Adapter — 数据集适配器基类

统一把各数据集的对话转成 Ingestor 能吃的 turn 列表。
子类实现 _load_case / _parse_turns / _build_history。
"""
from abc import ABC, abstractmethod

from build.ingest import Ingestor


class BaseAdapter(ABC):
    def __init__(self, dataset: str, skip_speaker_inference: bool = False):
        self.dataset = dataset
        self.skip_speaker_inference = skip_speaker_inference
        self.ingestor = Ingestor()

    @abstractmethod
    def load_case(self, case_path: str) -> dict:
        """加载一个 case，返回标准化结构。"""
        pass

    @abstractmethod
    def iter_turns(self, case: dict):
        """生成 (turn, history) 序列。turn 含 speaker_id/text/session_id/seq/timestamp 等。"""
        pass

    def ingest_case(self, case_path: str, max_turns: int = None,
                    fast_mode: bool = False) -> dict:
        """加载 + 写入整个 case 到 memory。返回统计。max_turns 限制（测试用）。"""
        case = self.load_case(case_path)
        conv_id = case.get("conv_id", case_path)

        n_turns = 0
        n_mem = 0
        n_failed = 0
        for turn, history in self.iter_turns(case):
            if max_turns and n_turns >= max_turns:
                break
            try:
                mem_id = self.ingestor.ingest_turn(
                    turn, history=history,
                    dataset=self.dataset, conv_id=conv_id,
                    skip_speaker_inference=self.skip_speaker_inference,
                    fast_mode=fast_mode
                )
            except Exception as e:
                # 单条 turn 失败不终止整个 ingest（避免丢后续记忆）
                n_failed += 1
                print(f"  [warn] turn {turn.get('session_id')}:{turn.get('seq')} "
                      f"ingest failed: {e}. Skipping (n_failed={n_failed}).", flush=True)
                mem_id = None
            n_turns += 1
            if n_turns % 20 == 0:
                print(f"  ...{n_turns} turns ingested (mem={n_mem}, failed={n_failed})", flush=True)
            if mem_id:
                n_mem += 1

        return {
            "conv_id": conv_id,
            "dataset": self.dataset,
            "n_turns": n_turns,
            "n_memories": n_mem,
        }
