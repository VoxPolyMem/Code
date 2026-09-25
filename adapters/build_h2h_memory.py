"""Freeze one shared raw/fact/collection memory representation for H2HMem.

Uses the already generated v3 grounded fact cache; no LLM calls and no QA labels.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
FACT_ROOT = Path(os.environ.get(
    "H2H_SOURCE_ROOT", PACKAGE_ROOT / "data/h2h_source"
))
DATA_ROOT = Path(os.environ.get(
    "H2HMEM_ROOT", PACKAGE_ROOT / "data/H2HMEM"
))
CAPTION_PATH = Path(os.environ.get(
    "H2HMEM_CAPTIONS", PACKAGE_ROOT / "data/dense_captions_h2hmem.json"
))
DEFAULT_OUT = PACKAGE_ROOT / "artifacts" / "memory" / "h2h"


def build(dialogue: str, split: str, out_dir: Path):
    source = FACT_ROOT / f"{split}_{dialogue}.json"
    doc = json.loads(source.read_text(encoding="utf-8"))
    captions = json.loads(CAPTION_PATH.read_text(encoding="utf-8"))
    raw = []
    raw_ids = set()
    for turn in doc["turns"]:
        sid, seq = str(turn["turn_id"]).split("#", 1)
        ref = f"raw:{dialogue}:{sid}:{seq}"
        image_captions = {}
        image_paths = []
        for item in turn.get("images", []):
            image_id = str(item.get("image_id") or "")
            if not image_id:
                continue
            image_sid, filename = image_id.split(":", 1)
            key = f"h2hmem:{split}/{dialogue}/scenes/{image_sid}/image/{filename}"
            caption = str(captions.get(key) or item.get("caption") or "")
            image_captions[image_id] = caption
            image_paths.append(str(
                DATA_ROOT / split / dialogue / "scenes" / image_sid / "image" / filename
            ))
        text = str(turn.get("text") or "")
        retrieval_text = text
        if image_captions:
            retrieval_text += "\n" + "\n".join(
                f"image_id={key}; caption={value}"
                for key, value in image_captions.items()
            )
        raw.append({
            "node_id": ref,
            "layer": "raw",
            "text": text,
            "retrieval_text": retrieval_text,
            "speaker": turn.get("speaker") or "unknown",
            "session_id": sid,
            "ordinal": int(seq),
            "date": turn.get("date") or "",
            "image_captions": image_captions,
            "image_paths": image_paths,
            "refer_ids": [ref],
        })
        raw_ids.add(ref)

    facts = []
    missing = set()
    for item in doc["facts"]:
        refs = []
        for source_ref in item.get("turn_refs", []):
            sid, seq = str(source_ref).split("#", 1)
            ref = f"raw:{dialogue}:{sid}:{seq}"
            refs.append(ref)
            if ref not in raw_ids:
                missing.add(ref)
        text = str(item.get("text") or "").strip()
        if text and refs:
            facts.append({
                "node_id": f"fact:{dialogue}:{item.get('fact_id')}",
                "layer": "fact",
                "text": text,
                "retrieval_text": text,
                "speaker": item.get("speaker") or "unknown",
                "addressee": item.get("addressee") or "group",
                "session_id": item.get("session") or "",
                "date": item.get("date") or "",
                "images": item.get("images") or [],
                "refer_ids": list(dict.fromkeys(refs)),
            })
    if missing:
        raise RuntimeError(f"facts reference missing raw nodes: {sorted(missing)[:10]}")

    collections = []
    sessions = list(dict.fromkeys(node["session_id"] for node in raw))
    for sid in sessions:
        members = [fact for fact in facts if fact["session_id"] == sid]
        refs = list(dict.fromkeys(
            ref for fact in members for ref in fact["refer_ids"]
        ))
        if members and refs:
            collections.append({
                "node_id": f"collection:{dialogue}:{sid}",
                "layer": "collection",
                "text": f"Grounded event collection for {sid}.",
                "retrieval_text": " ".join(fact["text"] for fact in members),
                "session_id": sid,
                "date": members[0].get("date", ""),
                "member_fact_ids": [fact["node_id"] for fact in members],
                "refer_ids": refs,
            })
    result = {
        "source_fact_cache": str(source),
        "split": split,
        "dialogue": dialogue,
        "raw": raw,
        "facts": facts,
        "collections": collections,
        "meta": {
            "turns": len(raw),
            "facts": len(facts),
            "collections": len(collections),
            "empty_refer_ids": sum(
                not row.get("refer_ids") for layer in (raw, facts, collections) for row in layer
            ),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split}_{dialogue}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(path), **result["meta"]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="multi-party")
    parser.add_argument("--dialogue", action="append", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    for dialogue in args.dialogue:
        build(dialogue, args.split, args.out)


if __name__ == "__main__":
    main()
