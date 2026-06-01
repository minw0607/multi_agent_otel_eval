"""
Mind2Web dataset loader.

Streams from osunlp/Multimodal-Mind2Web on HuggingFace, keeping only
lightweight text metadata (no HTML or screenshots).
Caches to a local JSONL file so subsequent runs skip the download.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset


def _validate_jsonl(path: Path, min_tasks: int = 50):
    if not path.exists():
        return False, 0, "File not found"
    if path.stat().st_size < 50_000:
        return False, 0, f"File too small ({path.stat().st_size} bytes)"
    count, required = 0, {"confirmed_task", "website"}
    try:
        with open(path) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not required.issubset(obj.keys()):
                    return False, count, f"Missing keys at line {i}"
                count += 1
        return (count >= min_tasks), count, "OK" if count >= min_tasks else f"Only {count} tasks"
    except Exception as e:
        return False, 0, str(e)


def _stream_mind2web(output_path: Path, target_tasks: int = 300,
                     max_stream: int = 5000) -> int:
    KEEP = ["annotation_id", "website", "domain", "subdomain",
            "confirmed_task", "action_reprs"]

    print(f"Connecting to osunlp/Multimodal-Mind2Web (streaming)…")
    ds = load_dataset("osunlp/Multimodal-Mind2Web", split="train", streaming=True)

    seen, count, processed = set(), 0, 0
    with open(output_path, "w") as f:
        for item in ds:
            processed += 1
            if processed % 500 == 0:
                print(f"  {processed} rows scanned, {count} tasks saved…")
            if count >= target_tasks or processed >= max_stream:
                break
            ann_id = item.get("annotation_id")
            if not ann_id or ann_id in seen or not item.get("confirmed_task"):
                continue
            seen.add(ann_id)
            row = {k: item.get(k) for k in KEEP if k in item}
            actions = row.get("action_reprs")
            if isinstance(actions, str):
                try:
                    actions = json.loads(actions)
                except Exception:
                    actions = [actions]
            row["action_reprs"] = actions if isinstance(actions, list) else []
            if not row["action_reprs"]:
                row["action_reprs"] = [f"[action] Navigate -> {row['confirmed_task'][:50]}"]
            f.write(json.dumps(row) + "\n")
            count += 1

    print(f"Saved {count} tasks from {processed} rows.")
    return count


def load_mind2web(data_dir: Path, target_tasks: int = 300) -> List[Dict[str, Any]]:
    """
    Load Mind2Web tasks from local cache or HuggingFace stream.

    Returns a list of dicts with keys:
      annotation_id, website, domain, subdomain, confirmed_task, action_reprs
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / "mind2web_train.jsonl"

    # Clean stale HF cache from a failed full download
    stale = Path.home() / ".cache" / "huggingface" / "datasets" / "osunlp___mind2web"
    if stale.exists():
        try:
            size_gb = sum(f.stat().st_size for f in stale.rglob("*") if f.is_file()) / 1e9
            if size_gb > 0.1:
                print(f"Cleaning stale HF cache ({size_gb:.1f} GB)…")
                shutil.rmtree(stale)
        except Exception:
            pass

    valid, n_tasks, msg = _validate_jsonl(cache)
    if valid:
        print(f"Using cached dataset: {n_tasks} tasks ({cache})")
    else:
        print(f"Cache status: {msg} — downloading…")
        _stream_mind2web(cache, target_tasks=target_tasks)
        valid, n_tasks, msg = _validate_jsonl(cache)
        if not valid:
            raise RuntimeError(f"Download failed validation: {msg}")

    tasks = []
    with open(cache) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks
