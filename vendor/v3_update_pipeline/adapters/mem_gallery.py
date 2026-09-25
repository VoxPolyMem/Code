"""
Mem-Gallery Adapter — Mem-Gallery 数据集适配

读 dialog JSON（user/assistant 二元 turn），转成统一 turn 列表。
单用户：speaker_id="user"/"assistant"，speaker=character_profile.name。
图片 caption 优先用 gpt-4o dense caption（HippoMM 生成），否则用数据集自带。
"""
import os
import json
from datetime import datetime, timezone
from pathlib import Path

from adapters.base import BaseAdapter

# dense caption 缓存（HippoMM gen_dense_captions.py 生成）
DENSE_CAPTION_CACHE = Path(os.environ.get(
    "MEMGALLERY_DENSE_CAPTIONS", "data/dense_captions_memgallery.json"
))
# Mem-Gallery 数据根（用于 resolve 图片相对路径）
MEM_GALLERY_ROOT = Path(os.environ.get("MEMGALLERY_ROOT", "data/Mem-Gallery"))

# caption 来源控制（默认 0=dataset caption，对齐 LTMemory run_bench.py:871）
# 实测: dense caption 和 dataset caption 语义完全不同，导致 embedding 偏移。
# LTMemory 用 dataset caption → 0.772; v2 用 dense caption → 0.605。改回对齐。
USE_DENSE_CAPTION = os.environ.get('USE_DENSE_CAPTION', '0') == '1'


class MemGalleryAdapter(BaseAdapter):
    def __init__(self):
        super().__init__(dataset="mem_gallery", skip_speaker_inference=True)
        self._dense_captions = self._load_dense_captions() if USE_DENSE_CAPTION else {}

    def _load_dense_captions(self) -> dict:
        """加载 gpt-4o dense caption 缓存（若有）。"""
        if DENSE_CAPTION_CACHE.exists():
            data = json.load(open(DENSE_CAPTION_CACHE))
            print(f"  [MemGallery] loaded dense captions: {len(data)} images")
            return data
        print(f"  [MemGallery] no dense caption cache, using dataset captions")
        return {}

    def _get_caption(self, img_rel_path: str, dataset_cap: str = "") -> str:
        """caption 来源: 默认 dataset caption（对齐 LTMemory），USE_DENSE_CAPTION=1 时用 dense。"""
        if USE_DENSE_CAPTION and self._dense_captions:
            key = f"memgallery:{img_rel_path}"
            if key in self._dense_captions:
                return self._dense_captions[key]
            if img_rel_path in self._dense_captions:
                return self._dense_captions[img_rel_path]
        return dataset_cap

    def _resolve_image_path(self, img_rel: str, case_path_stem: str) -> str:
        """把相对路径 resolve 成绝对路径。"""
        # img_rel 格式如 "../image/TopicName/D1_IMG_001.jpg"
        # 或 "image/TopicName/D1_IMG_001.jpg"
        p = Path(img_rel)
        # 尝试相对 Mem-Gallery root
        full = MEM_GALLERY_ROOT / "data" / img_rel.replace("../", "")
        if full.exists():
            return str(full)
        # 尝试相对 data/dialog
        full2 = MEM_GALLERY_ROOT / "data" / "dialog" / img_rel
        if full2.exists():
            return str(full2)
        return str(full)  # 返回最可能的，即使不存在

    def load_case(self, case_path: str) -> dict:
        data = json.load(open(case_path, encoding="utf-8"))
        char_name = data.get("character_profile", {}).get("name", "user")
        return {
            "conv_id": Path(case_path).stem,
            "character_name": char_name,
            "sessions": data.get("multi_session_dialogues", []),
            "case_path": case_path,
        }

    def iter_turns(self, case: dict):
        HISTORY_N = 4
        history = []
        char_name = case["character_name"]
        for session in case["sessions"]:
            sid = session.get("session_id", "S1")
            date_str = session.get("date")
            timestamp = int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) if date_str else None
            for seq, dlg in enumerate(session.get("dialogues", [])):
                # image
                image_captions = {}
                image_paths = []
                caps = dlg.get("image_caption", [])
                imgs = dlg.get("input_image", [])
                if isinstance(caps, str):
                    caps = [caps]
                if isinstance(imgs, str):
                    imgs = [imgs]
                for i, img in enumerate(imgs):
                    ds_cap = caps[i] if i < len(caps) else ""
                    # caption 来源: 默认 dataset caption（对齐 LTMemory），USE_DENSE_CAPTION=1 时用 dense
                    cap = self._get_caption(img, ds_cap)
                    # 对齐 LTMemory process_conversation: 用数据集自带 image_id（格式 "D1:IMG_001"）
                    # 不用 Path(img).stem（"D1_IMG_001" 下划线格式，VS 题 judge 按冒号格式评分会判错）
                    img_ids_data = dlg.get("image_id", [])
                    if isinstance(img_ids_data, str):
                        img_ids_data = [img_ids_data]
                    img_id = img_ids_data[i] if i < len(img_ids_data) else (Path(img).stem if img else f"img_{i}")
                    image_captions[img_id] = cap
                    image_paths.append(self._resolve_image_path(img, case["conv_id"]))

                # user turn + assistant turn
                user_text = dlg.get("user", "")
                asst_text = dlg.get("assistant", "")

                # 合并 user+assistant 成一个 turn（对齐 LTMemory process_conversation）
                # LTMemory: speaker_a = f"user ({char_name})"，text = f"{speaker_a}: {user_text}\nassistant: {asst_text}"
                speaker_a = f"user ({char_name})"
                combined_text_parts = []
                if user_text:
                    combined_text_parts.append(f"{speaker_a}: {user_text}")
                if asst_text:
                    combined_text_parts.append(f"assistant: {asst_text}")
                combined_text = "\n".join(combined_text_parts) if combined_text_parts else ""
                if not combined_text:
                    continue
                merged_turn = {
                    "speaker_id": "user",
                    "speaker": char_name,
                    "addressee": "assistant",
                    "text": combined_text,
                    "session_id": sid,
                    "seq": seq,
                    "timestamp": timestamp,
                    "date": date_str,          # V2: 日期字符串（对齐 LTMemory timestamp=session_date）
                    "image_captions": image_captions,
                    "image_paths": image_paths,
                }
                yield merged_turn, list(history[-HISTORY_N:])
                history.append({"speaker_id": char_name, "text": user_text})
                history.append({"speaker_id": "assistant", "text": asst_text})
