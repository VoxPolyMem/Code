"""
Retrieval Logger — 检索每 round 详细信息 log 到文件

目录结构: logs/<config_tag>/<run_timestamp>/<case_name>/<qa_id>.log
- 每个QA（题目）单独一个log文件，便于定位bad case
- case_name 作为子目录，里面是各QA的log
"""
import json
from pathlib import Path
from datetime import datetime


class RetrievalLogger:
    def __init__(self, log_dir: str | Path, case_name: str):
        # log_dir/<case_name>/ 下每个QA一个log文件
        # 如 logs/rvllm_agpt_morig_img/hybrid_0803_1449/AI_Robotics_.../Q1.log
        self.log_dir = Path(log_dir) / case_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.case_name = case_name
        self._current_qa = None
        self._lines = []

    def start_qa(self, qa_id: str, question: str, gt: str, point: str):
        """开始一个 QA 的日志。"""
        self._current_qa = qa_id
        self._lines = [
            f"{'='*60}",
            f"QA: {qa_id} | point={point}",
            f"Question: {question}",
            f"Ground Truth: {gt}",
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"{'='*60}",
        ]

    def log_round(self, round_idx: int, route: dict, results: list,
                  new_added: int, sufficiency: dict, rewritten: str = None,
                  label: str = None):
        """记录一个 round 的详情。全量打印检索结果（不限5条，不截断）。

        label: 自定义块标题。eval 脚本记录「最终给答题模型的记忆」时传
        label="Final memory for answer"，避免和 orchestrator 内部 Round 0 混淆
        （eval 的 log_round(0,...) 记的是去重+截取后的最终 context，不是
        orchestrator Round 0 的原始检索结果）。默认用 "Round {round_idx}"。
        """
        self._lines.append(f"\n{'─'*50}")
        self._lines.append(f"[{label}]" if label else f"[Round {round_idx}]")
        if "routes" in route:
            self._lines.append(f"  Routes:")
            for r in route.get("routes", []):
                self._lines.append(f"    {r.get('route','?')}: query=\"{r.get('query','?')}\"")
            self._lines.append(f"  Sub-questions: {route.get('sub_questions', [])}")
            self._lines.append(f"  Confidence: {route.get('confidence', 0)}")
        else:
            self._lines.append(f"  Route: type={route.get('query_type')} "
                              f"tool={route.get('tool_choice')} "
                              f"speaker={route.get('speaker_hint')} "
                              f"time={route.get('time_range')}")
            self._lines.append(f"  Sub-questions: {route.get('sub_questions', [])}")
        self._lines.append(f"  Results: {len(results)} retrieved, +{new_added} new")
        for r in results:
            date = r.get("date", "?")
            session = r.get("session_id", "?")
            score = r.get("score", 0)
            text = r.get("text", "")
            self._lines.append(
                f"    [{r.get('mem_id')}] score={score:.4f} "
                f"date={date} session={session} "
                f"text={text}"
            )
            caps = r.get("caption")
            if caps and isinstance(caps, dict):
                for img_id, cap_text in caps.items():
                    self._lines.append(f"      Caption {img_id}: {cap_text}")
            img_paths = r.get("image_paths", [])
            if img_paths:
                self._lines.append(f"      image_paths: {img_paths}")
            raw_imgs = r.get("raw_images", [])
            if raw_imgs:
                self._lines.append(f"      raw_images: {len(raw_imgs)} images")
        self._lines.append(f"  Sufficiency: {json.dumps(sufficiency, ensure_ascii=False)}")
        if rewritten:
            self._lines.append(f"  Rewritten query: {rewritten}")

    def log_answer(self, prediction: str, judge: dict):
        """记录最终答案 + 评分。"""
        self._lines.append(f"\n{'─'*50}")
        self._lines.append(f"Prediction: {prediction}")
        self._lines.append(f"Judge: score={judge.get('score')} reason={judge.get('reasoning','')}")
        self._lines.append(f"{'='*60}\n")

    def flush(self):
        """每个QA单独写一个log文件: <case_dir>/<qa_id>.log"""
        if self._current_qa and self._lines:
            log_path = self.log_dir / f"{self._current_qa}.log"
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(self._lines))
        self._lines = []
