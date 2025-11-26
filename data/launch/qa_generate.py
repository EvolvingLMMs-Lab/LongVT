#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QA Generate Module

This module generates question-answer pairs from merged video captions using OpenAI-compatible APIs.
It processes video segments, merges captions, and generates fine-grained QA pairs with temporal grounding.

Usage:
    python launch/qa_generate.py --input-dir /path/to/captions --output-dir /path/to/output
"""

import os
import json
import time
import random
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from typing import Dict, List, Optional, Tuple


# ========================= Prompts =========================
SYSTEM_PROMPT = """
You are an expert dataset curator. Your task is to create ONE fine-grained question–answer (QA) pair for one complete video, using the merged caption that describes all segments.
-------------------------------------------------------------------------
The input contains a complete video description with multiple time segments merged together:
Video description: {merged_caption}
-------------------------------------------------------------------------
1 · Generate ONE Question
 • Write 1 English question that is answerable exclusively from this complete video description.
 • Focus on concrete details (objects, colors, actions, temporal sequences, locations, etc.).
 • Question ≤ 50 words.
 • Someone who has not "seen" this exact video description must be unable to answer.
 • Before writing, reason about the dense captions and synthesize across segments; prefer questions that require multi-step temporal/spatial reasoning rather than single-frame facts.
 • Avoid subjective or evaluative questions (e.g., "How do these actions contribute to the video's overall emotional tone?" or "How do these actions reflect their emotions?").
 • Ask exactly one question—no multi-part or double-barreled questions. Avoid "X and Y" constructions (e.g., "Around what focal object do onlookers congregate, and in what visible manner do they show their involvement while the players briskly move the miniature soccer figures?").
 • IMPORTANT: Do NOT include precise timestamps (like "at 2.5 seconds" or "[1.0s - 3.2s]") in the question.
2 · Provide Answer
 • Give a short, direct answer (≤ 25 words) to the question, derived strictly from the video description.
3 · No Answer Leakage
 • The question must not reveal or encode the answer (or a synonym/unique cue) within the question itself).
 • Avoid leading phrasing such as "Which color is the red car?", "Is the trophy gold?", or any wording that makes the answer guessable without evidence.
 • Do not include rare proper nouns, numbers, or attributes that uniquely determine the answer unless they are necessary context in the description.
4 · Time Range
 • Specify the time range [start_time, end_time] in seconds that is most relevant to answer this question.
5 · Output JSON only
Return exactly the structure below—nothing more, nothing less:
{
  "video_id": "video_name",
  "question": "…",
  "answer": "…",
  "start_time": X.X,
  "end_time": Y.Y
}

-------------------------------------------------------------------------
Here is the complete video description you should process: 

"""

# Rate limiting
BASE_SLEEP = float(os.getenv("QA_GEN_SLEEP", "0.8"))
JITTER = 0.4

# Pricing (can be overridden via environment variables)
PRICE_IN_PER_1K = float(os.getenv("OPENAI_PRICE_IN", "0.002"))
PRICE_OUT_PER_1K = float(os.getenv("OPENAI_PRICE_OUT", "0.008"))


# ========================= Logging =========================
class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that works with tqdm progress bars."""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            pass


def setup_logging():
    """Setup logging configuration."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    h = TqdmLoggingHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)


setup_logging()


# ========================= Utility Functions =========================
def strip_code_fences(s: str) -> str:
    """Remove markdown code fences from a string."""
    s = s.strip()
    if s.startswith("```"):
        if s.startswith("```json"):
            s = s[len("```json"):].strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    return s


def extract_usage(resp) -> Tuple[int, int, int]:
    """
    Safely extract token usage from API response.
    Returns: (prompt_tokens, completion_tokens, total_tokens)
    """
    try:
        u = getattr(resp, "usage", None)
        if u:
            pt = getattr(u, "prompt_tokens", None)
            ct = getattr(u, "completion_tokens", None)
            tt = getattr(u, "total_tokens", None)
            if pt is None or ct is None or tt is None:
                u = resp.usage
                pt = u.get("prompt_tokens", 0) if isinstance(u, dict) else (pt or 0)
                ct = u.get("completion_tokens", 0) if isinstance(u, dict) else (ct or 0)
                tt = u.get("total_tokens", 0) if isinstance(u, dict) else (tt or (pt + ct))
            return int(pt or 0), int(ct or 0), int(tt or 0)
    except Exception:
        pass

    try:
        if hasattr(resp, "model_dump_json"):
            d = json.loads(resp.model_dump_json())
            u = d.get("usage", {})
            return int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)), int(u.get("total_tokens", 0))
    except Exception:
        pass
    return 0, 0, 0


def infer_video_key_and_meta(item: Dict, fallback_key: str) -> Tuple[str, str, str]:
    """
    Infer unique video key and metadata from a single segment.
    Returns: (video_key(stem), dataset_name, filename_with_ext)
    """
    for k in ("video_path", "VIDEO_PATH", "video", "path"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            p = Path(v)
            return p.stem, p.parent.name, p.name

    for k in ("video_name", "video_id", "vid", "videoId"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            name = Path(v).name
            stem = Path(v).stem
            return stem, "unknown_ds", name

    return fallback_key, "unknown_ds", f"{fallback_key}.mp4"


# ========================= OpenAI API =========================
def init_openai_client() -> Optional[OpenAI]:
    """Initialize OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if not api_key:
        logging.error("OPENAI_API_KEY environment variable not found")
        return None
    
    try:
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        logging.info("OpenAI client initialized successfully")
        return client
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        return None


def call_llm(client: OpenAI, merged_caption: str, model: str = "gpt-4o") -> Tuple[Optional[Dict], int, int, float]:
    """
    Call LLM to generate QA pair.
    Returns: (qa_json or None, prompt_tokens, completion_tokens, cost_usd)
    """
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": merged_caption},
        ]

        t0 = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        dt = time.time() - t0

        content = (resp.choices[0].message.content or "").strip()
        if not content:
            logging.error("LLM returned empty content")
            return None, 0, 0, 0.0

        cleaned = strip_code_fences(content)
        qa_data = json.loads(cleaned)

        # Calculate tokens and cost
        p_tok, c_tok, _ = extract_usage(resp)
        cost = (p_tok / 1000.0) * PRICE_IN_PER_1K + (c_tok / 1000.0) * PRICE_OUT_PER_1K

        if "question" in qa_data and "answer" in qa_data:
            logging.info(f"[{model}] OK | {p_tok}+{c_tok} tok | ${cost:.4f} | {dt:.2f}s")
            return qa_data, p_tok, c_tok, cost
        else:
            logging.error(f"JSON missing required fields: {list(qa_data.keys())}")
            return None, p_tok, c_tok, cost

    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing failed: {e}")
        return None, 0, 0, 0.0
    except Exception as e:
        logging.error(f"LLM call failed: {e}")
        return None, 0, 0, 0.0


# ========================= Data Loading =========================
def load_videos_and_groups(shard_path: str, group_size: int = 30) -> List[Dict]:
    """
    Load a shard JSON (list of segments), group by video, sort by time, and split into groups.
    
    Returns a list where each element contains:
    {
      "video_key": "...",
      "dataset": "...",
      "filename": "...",
      "groups": [
         { "merged_caption": "...", "group_start_time": .., "group_end_time": .., ... },
         ...
      ]
    }
    """
    path = Path(shard_path)
    if not path.exists():
        logging.error(f"File does not exist: {shard_path}")
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logging.error(f"Failed to read {shard_path}: {e}")
        return []

    if not isinstance(data, list):
        logging.error("Data format error: expected list of segments")
        return []

    fallback_key = path.stem

    grouped: Dict[str, Dict] = {}
    bad = 0
    for item in data:
        if not isinstance(item, dict) or not all(k in item for k in ("start_time", "end_time", "caption")):
            bad += 1
            continue
        vkey, ds, fname = infer_video_key_and_meta(item, fallback_key)
        if vkey not in grouped:
            grouped[vkey] = {"video_key": vkey, "dataset": ds, "filename": fname, "segments": []}
        grouped[vkey]["segments"].append(item)
    
    if bad:
        logging.warning(f"Skipped {bad} segments missing required fields.")

    # Build groups for each video
    videos: List[Dict] = []
    for vkey, meta in grouped.items():
        segs = meta["segments"]
        segs.sort(key=lambda x: x["start_time"])
        groups = []
        for i in range(0, len(segs), group_size):
            part = segs[i:i + group_size]
            merged_caption = "\n\n".join(
                f"Video_event {i + j + 1} [{s['start_time']:.1f}s - {s['end_time']:.1f}s]: {s['caption']}"
                for j, s in enumerate(part)
            )
            groups.append({
                "group_id": len(groups) + 1,
                "start_segment": i + 1,
                "end_segment": i + len(part),
                "segments_count": len(part),
                "group_start_time": part[0]["start_time"],
                "group_end_time": part[-1]["end_time"],
                "merged_caption": merged_caption
            })
        videos.append({
            "video_key": vkey,
            "dataset": meta["dataset"],
            "filename": meta["filename"],
            "groups": groups
        })

    videos.sort(key=lambda x: x["video_key"])
    logging.info(f"Shard '{path.name}': {len(videos)} videos, {sum(len(v['groups']) for v in videos)} groups total")
    return videos


# ========================= Processing =========================
def process_one_shard(
    shard_path: str,
    shard_idx: int,
    output_dir: str,
    model: str = "gpt-4o",
    group_size: int = 15
) -> None:
    """Process a single shard and generate QA pairs."""
    client = init_openai_client()
    if not client:
        return

    out_dir_shard = os.path.join(output_dir, f"shard_{shard_idx}")
    Path(out_dir_shard).mkdir(parents=True, exist_ok=True)

    videos = load_videos_and_groups(shard_path, group_size=group_size)
    total_videos = len(videos)
    if total_videos == 0:
        logging.warning(f"{shard_path} has no valid videos")
        return

    pbar = tqdm(total=total_videos, desc=f"Shard {shard_idx} videos", position=0, leave=True, dynamic_ncols=True)

    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for vid_idx, v in enumerate(videos, 1):
        vkey = v["video_key"]
        dataset = v["dataset"]
        filename = v["filename"]

        out_path = os.path.join(out_dir_shard, f"{dataset}_{vkey}_shard_{shard_idx}.json")

        if os.path.exists(out_path):
            logging.info(f"[SKIP] Already exists: {out_path}")
            pbar.update(1)
            continue

        results_for_video: List[Dict] = []
        logging.info(f"Processing video {vid_idx}/{total_videos} | {dataset}/{filename} | Groups: {len(v['groups'])}")

        for g in v["groups"]:
            qa, p_tok, c_tok, cost = call_llm(client, g["merged_caption"], model=model)
            total_prompt_tokens += p_tok
            total_completion_tokens += c_tok
            total_cost += cost

            if qa:
                results_for_video.append({
                    "video_name": filename,
                    "group_id": g["group_id"],
                    "start_segment": g["start_segment"],
                    "end_segment": g["end_segment"],
                    "segments_count": g["segments_count"],
                    "group_start_time": g["group_start_time"],
                    "group_end_time": g["group_end_time"],
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                    "qa_start_time": qa.get("start_time", g["group_start_time"]),
                    "qa_end_time": qa.get("end_time", g["group_end_time"]),
                    "merged_caption": g["merged_caption"]
                })
            else:
                logging.warning(f"[FAIL] {dataset}/{filename} group {g['group_id']} generation failed")

            time.sleep(BASE_SLEEP + random.random() * JITTER)

        if results_for_video:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results_for_video, f, ensure_ascii=False, indent=2)
            logging.info(f"[WRITE] {out_path} | Cumulative: ${total_cost:.2f} "
                        f"({total_prompt_tokens}+{total_completion_tokens} tok)")
        else:
            logging.warning(f"[EMPTY] {dataset}/{filename} no results to save")

        pbar.update(1)

    pbar.close()
    logging.info(f"Shard {shard_idx} complete | Estimated cost: ${total_cost:.2f} | "
                f"tokens: {total_prompt_tokens}+{total_completion_tokens}")


def find_all_json_files(base_dir: str) -> List[str]:
    """Find all JSON files in a directory recursively."""
    files = []
    for root, _, fs in os.walk(base_dir):
        for f in fs:
            if f.endswith(".json"):
                files.append(os.path.join(root, f))
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser(description="QA Generation from Video Captions")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory containing caption JSON files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for generated QA pairs")
    parser.add_argument("--shard-idx", type=int, default=0, help="Current shard index (0-based)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--group-size", type=int, default=15, help="Number of segments to merge per group")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use for QA generation")

    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be > 0")
    if not (0 <= args.shard_idx < args.num_shards):
        raise ValueError("--shard-idx must be in [0, num-shards)")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    shard_files = find_all_json_files(args.input_dir)
    if not shard_files:
        logging.error(f"No JSON files found in {args.input_dir}")
        return

    shard_files = sorted(shard_files)
    my_files = [p for i, p in enumerate(shard_files) if i % args.num_shards == args.shard_idx]

    logging.info(f"Total shard files: {len(shard_files)} | Current shard {args.shard_idx}/{args.num_shards} "
                f"processing {len(my_files)} files")
    
    for idx, shard_path in enumerate(my_files, 1):
        logging.info(f"\n=== Processing shard file {idx}/{len(my_files)}: {Path(shard_path).name} ===")
        process_one_shard(
            shard_path,
            shard_idx=args.shard_idx,
            output_dir=args.output_dir,
            model=args.model,
            group_size=args.group_size
        )


if __name__ == "__main__":
    main()

