#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iMCoTT (Interleaved Multimodal Chain-of-Tool-Thought) Generate Module

This module generates multi-turn reasoning traces with tool calling for video QA.
It uses a coarse-to-fine search strategy with global skimming and fine-grained inspection.

Usage:
    python launch/imcott_generate.py --input-file /path/to/qa.json --output-dir /path/to/output
"""

import os
import re
import json
import time
import random
import math
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from string import Template
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import cv2
from tqdm import tqdm

# Suppress OpenCV verbose logs
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass


# ==================== Configuration ====================
# API Configuration (set via environment variables)
# GOOGLE_API_KEY or GEMINI_API_KEY for Gemini
# OPENAI_API_KEY and OPENAI_BASE_URL for OpenAI-compatible APIs

MODEL_NAME = os.getenv("IMCOTT_MODEL", "gemini-2.5-pro")

# Pricing (can be overridden via environment variables)
PRICE_IN_PER_1K = float(os.getenv("IMCOTT_PRICE_IN", "0.00125"))
PRICE_OUT_PER_1K = float(os.getenv("IMCOTT_PRICE_OUT", "0.005"))

# Retry configuration
GEN_RETRIES = 6
GEN_BASE_DELAY = 1.0

# Concurrency configuration
MAX_UPLOAD_WORKERS = 1
JSON_WORKERS = 1

# Rate limiting
GEMINI_RPM = 500
GEMINI_TPM = 500_000

# Image processing
MAX_AREA = 224 * 224


# ==================== Prompts ====================
SYSTEM_PROMPT = """
You are a helpful assistant for long-video understanding and reasoning.

You may call one or more functions to answer my query.

<tools>
{"type":"function","function":{"name":"crop_video","description":"Crop a video to a specified duration (return the exact start/end timestamps you selected; no images).","parameters":{"type":"object","properties":{"video_path":{"type":"string","description":"Path to the video file"},"start_time":{"type":"number","description":"Start time in seconds"},"end_time":{"type":"number","description":"End time in seconds, must be > start_time"}},"required":["video_path","start_time","end_time"]},"strict":false}}
</tools>

For every function call, wrap a JSON object with the function name and its arguments inside <tool_call></tool_call> tags, immediately followed by a <tool_response> tag whose text is exactly **"The tool executed successfully."**.

--------------------------------------------------------------------
**Input you receive**

VIDEO_PATH            : ${VIDEO_PATH}
QUESTION              : ${QUESTION}
GROUND_TRUTH_ANSWER   : ${GROUND_TRUTH_ANSWER}
GROUND_TRUTH_TIME     : ${GROUND_TRUTH_TIME}
PREFERRED_TOOL_CALLS  : ${PREFERRED_TOOL_CALLS}
Additionally, a series of image frames sampled from the video (at most 512 frames) are provided as input for your reference.

--------------------------------------------------------------------
**We will run in TWO phases. Adopt a *coarse-to-fine* search mindset.**
"""

GLOBAL_SKIM_PROMPT = """
This is **PHASE-1 (global skim & planning — first <think> block)**.

- Reconstruct the visual storyline of the **entire video** by interpreting the sequence of provided frames (silent video). Do **not** mention that you are looking at static images or frames; narrate it as a continuous video scene.
- In ≈ 4–6 flowing sentences, narrate what the camera shows across the whole video (settings, actors, transitions).

- **Timestamp during thinking:** As you narrate, sprinkle human-readable time anchors for key moments (not only the final windows). Allowed styles include: ≈297s, around 298–300s, from 4:56 to 5:15, 295–300s, or [296.34s – 320.76s].

- **Planning (STRICT): propose only the Round-1 window.**
  Even if more than one round will be used (**PREFERRED_TOOL_CALLS ≥ 2**), **do NOT mention any later/confirmatory window.**
  Pick one tight span (≈3–8s) that plausibly contains the onset, the peak, or the immediate aftermath of the key evidence.

- **HARD CONSTRAINTS (must follow exactly):**
  1) You must output **exactly one non-empty** `<think>...</think>` block in this phase.  
  2) The `<think>` text must contain normal prose (3–6 sentences). Do **not** leave it blank, do **not** output placeholders, and do **not** output gibberish or non-text symbols.  
  3) Do **not** output `<tool_call>` / `<tool_response>` in this phase.  
  4) Use plain ASCII punctuation; avoid mojibake/garbled characters.

- **Formatting rule (mandatory):** every time range you provide must be in the strict form [S.s – E.e] with trailing s.
- **Inline placement (mandatory):** any bracketed window must appear **within a sentence**; never put a [S.s – E.e] window on its own line or as a list item.

- **Output shape:** output only <question>...</question> immediately followed by a single <think>...</think> block — **no blank lines** between these tags. Do not output <tool_call>, <tool_response>, <answer>, or <time> in this phase.
- If uncertain, you must still write a generic but coherent narration; leaving <think> blank is forbidden.
- If any constraint would be violated, self-correct within the same output and keep only one valid <think>.
"""

FINE_INSPECTION_PROMPT = """
You are now in **PHASE-2 (fine-grained inspection)**, **round ${ROUND_IDX}/${TOTAL_ROUNDS}**.

**You will continue an existing response.**  
Below is the response **so far** (global skim + any finished rounds). **Do not rewrite or repeat anything** from it; your job is to **append exactly the next blocks**.

--- Transcript so far (read-only) ---
${TRANSCRIPT_SO_FAR}
--- End of transcript so far ---

**What you additionally receive for this round**
- The planned window for this round: **[${S},${E}]** seconds.  
  **You must use this exact window in the <tool_call> (start_time=${S}, end_time=${E}). Do not modify.**
- **Attached frames:** images from the video segment of this interval (low resolution, ~224px).  
- The original QUESTION (for reference): ${QUESTION}
- Remember: total planned tool calls = **${PREFERRED_TOOL_CALLS}** (do not mention this fact in your prose).

**Time mention style for this round:** When you refer to timings inside your <think>, include human-readable anchors like ≈610–615s, at ~612s, 10:10–10:15, or [608.00s – 621.00s].  
If the exact FPS/timecode is unknown (e.g., frames are evenly sampled), avoid non-zero decimals: round to integer seconds and render with two decimals as **.00**.

**In the <think> block you append this round, include three parts (as prose, not bullet labels):**
1) **Evidence**: what this window [${S},${E}] shows that helps answer the question.  
2) **Integration**: how this confirms or revises your earlier hypothesis (mark outdated bits as "revised: …").  
3) **Self-reflection**: whether this window was mis-localized; if so, how you would correct it; otherwise note that it suffices for its subgoal.

**HARD CONSTRAINTS for every round (must follow exactly):**
- You must output **exactly one** `<tool_call>` block **followed by** **exactly one** `<tool_response>` block, **then** **exactly one** `<think>` block — **in this order**.
- Never start a round with `<think>`. Never output two `<think>` blocks in a row.  
- The `<think>` for this round must be **non-empty prose (3–6 sentences)**; avoid blank/placeholder/gibberish content and non-ASCII symbols.
- Across the **entire** PHASE-2, the total number of `<tool_call>` blocks must equal **${PREFERRED_TOOL_CALLS}** — **one per round**.
- If you realize a previous sentence would violate a constraint, **fix it by continuing correctly in this output**; do not restate previous phases.

**Your output for this round — append exactly the following blocks, in this order (do not omit any block):**
<tool_call>
{ "name":"crop_video", "arguments":{ "video_path":"${VIDEO_PATH}", "start_time": ${S}, "end_time": ${E} } }
</tool_call>
<tool_response>
The tool executed successfully.
</tool_response>
<think>
</think>
# … (repeat until you reach exactly ${PREFERRED_TOOL_CALLS} cycles) …

If and only if this is the **last round (${ROUND_IDX} == ${TOTAL_ROUNDS})**, **after** the <think> block, append exactly one more block named <answer> that contains only your final short answer (no explanations). Use this exact tag format:

<answer>
YOUR_FINAL_ANSWER
</answer>
"""

EXTRACT_WINDOWS_PROMPT = """
You will be given a text that contains a question and a single <think> block.

Your task: **infer the EXACT K inspection windows the author intended** and output them **only** inside a `<time>...</time>` block, comma-separated.

Rules:
- Prefer spans explicitly proposed for inspection/cropping.
- The text may use varied time expressions (≈, ~, mm:ss); normalize as needed.
- **If only one Round-1 window is present (typical when PREFERRED_TOOL_CALLS ≥ 2), derive K windows as follows:**
  - Window #1 = the Round-1 window as written.
  - Window #2..K = plausible follow-ups that would confirm/adjust the hypothesis: either tighten around the nucleus (3–8s), or shift slightly earlier/later, or expand to satisfy "after A before B". Keep them in the same local neighborhood; do not invent distant times.
- If a single enclosing range [S–E] (not Round-1) is stated, derive K concise sub-windows (≈4–6s) capturing onset, peak action, aftermath — all strictly inside [S–E].
- Never invent seconds outside what the text implies.

**Format strictly**:  
<time>[S.s – E.e] with trailing **s**, comma-separated, exactly **K** items and nothing else inside `<time>`.

K = ${K}

--- TEXT START ---
${PHASE1_TEXT}
--- TEXT END ---
"""


# ==================== Utility Functions ====================
def cost_usd(prompt_tok: int, output_tok: int) -> float:
    """Calculate cost in USD."""
    return (prompt_tok / 1000.0) * PRICE_IN_PER_1K + (output_tok / 1000.0) * PRICE_OUT_PER_1K


def fmt_secs(sec: float) -> str:
    """Format seconds."""
    return f"{sec:.2f}s"


def fmt_dur(t0: float, t1: float) -> str:
    """Format duration."""
    return f"{(t1 - t0):.2f}s"


def extract_think_text(text: str) -> str:
    """Extract text from <think> block."""
    m = re.search(r"<think>(.*?)</think>", text, flags=re.S | re.I)
    return (m.group(1) or "").strip() if m else ""


def think_is_nonempty(s: str) -> bool:
    """Check if think text is non-empty."""
    return bool(re.search(r"[A-Za-z0-9]", s)) and len(re.sub(r"\s+", " ", s)) > 60


def keep_until_first_think_and_time(text: str) -> str:
    """Keep text until first </think>."""
    try:
        end = text.index("</think>") + len("</think>")
        out = text[:end]
    except Exception:
        out = text
    out = re.sub(r"</question>\s*\n\s*<think>", "</question>\n<think>", out, flags=re.I)
    return out


def ensure_q_think_blocks(text: str, question: str) -> str:
    """Ensure proper question and think blocks."""
    t = (text or "").strip()
    
    def normalize_newlines(s: str) -> str:
        s = re.sub(r"<question>\s*", "<question>\n", s, flags=re.I)
        s = re.sub(r"\s*</question>", "\n</question>", s, flags=re.I)
        s = re.sub(r"<think>\s*", "<think>\n", s, flags=re.I)
        s = re.sub(r"\s*</think>", "\n</think>", s, flags=re.I)
        return s
    
    has_q = re.search(r"<question>.*?</question>", t, flags=re.S | re.I) is not None
    has_think = re.search(r"<think>.*?</think>", t, flags=re.S | re.I) is not None
    
    if has_q and has_think:
        return normalize_newlines(t)
    
    m = re.search(r"<time>.*?</time>", t, flags=re.S | re.I)
    time_part = ""
    head = t
    if m:
        time_part = t[m.start():m.end()]
        head = t[:m.start()].strip()
    
    res = f"<question>{question}</question>\n<think>\n{head}\n</think>"
    if time_part:
        res += "\n" + time_part
    return normalize_newlines(res)


# ==================== Rate Limiter ====================
class RateLimiter:
    """Simple rate limiter for RPM/TPM."""
    
    def __init__(self, rpm: int, tpm: int):
        self.rps = max(1e-6, rpm / 60.0)
        self.tps = max(1e-3, tpm / 60.0)
        self._r_allow = 0.0
        self._t_allow = 0.0
        self._last = time.monotonic()
        self._lock = Lock()
    
    def acquire(self, tokens_in: int):
        """Acquire rate limit tokens."""
        tokens_in = max(1, int(tokens_in or 0))
        with self._lock:
            now = time.monotonic()
            dt = now - self._last
            self._last = now
            self._r_allow = min(self._r_allow + dt * self.rps, self.rps * 2)
            self._t_allow = min(self._t_allow + dt * self.tps, self.tps * 2)
            
            wait = 0.0
            if self._r_allow < 1.0:
                wait = max(wait, (1.0 - self._r_allow) / self.rps)
            if self._t_allow < tokens_in:
                wait = max(wait, (tokens_in - self._t_allow) / self.tps)
            
            if wait > 0:
                time.sleep(wait)
                dt2 = time.monotonic() - self._last
                self._last += dt2
                self._r_allow = min(self._r_allow + dt2 * self.rps, self.rps * 2)
                self._t_allow = min(self._t_allow + dt2 * self.tps, self.tps * 2)
            
            self._r_allow -= 1.0
            self._t_allow -= tokens_in


# ==================== Video Processing ====================
def resize_keep_area(frame):
    """Resize frame while keeping area constraint."""
    h, w = frame.shape[:2]
    if h * w <= MAX_AREA:
        return frame
    scale = math.sqrt(MAX_AREA / (h * w))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def get_video_meta(video_path: str) -> Tuple[float, int, float]:
    """Get video metadata: (fps, total_frames, duration)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    if total_frames > 0:
        duration = total_frames / max(fps, 1e-6)
    else:
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        duration = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    cap.release()
    
    if duration <= 0:
        raise RuntimeError("Could not determine video duration for frame sampling.")
    
    return float(fps), int(total_frames), float(duration)


def read_frame_by_time(cap, t: float, fps: float, total_frames: int, duration: float):
    """Read frame at specific time."""
    if fps > 0 and total_frames > 0:
        idx = max(0, min(int(round(t * fps)), total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            return frame
    
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, min(t, duration - 1e-3)) * 1000.0)
    ret, frame = cap.read()
    if ret and frame is not None:
        return frame
    
    for dt in (0.05, 0.1, -0.05, -0.1, 0.2, -0.2, 0.3, -0.3):
        tt = max(0.0, min(t + dt, duration - 1e-3))
        cap.set(cv2.CAP_PROP_POS_MSEC, tt * 1000.0)
        ret, frame = cap.read()
        if ret and frame is not None:
            return frame
    
    return None


def plan_global_times(duration: float) -> List[float]:
    """Plan global sampling times."""
    if duration >= 512:
        return [duration * i / 512.0 for i in range(512)]
    t, times = 0.0, []
    while t < duration - 1e-6:
        times.append(t)
        t += 1.0
    if not times:
        times = [0.0]
    return times


def ensure_global_frames(video_path: str, out_dir: Path) -> List[str]:
    """Ensure global frames are extracted."""
    fps, total_frames, duration = get_video_meta(video_path)
    times = plan_global_times(duration)
    expected = len(times)
    
    if out_dir.exists():
        pngs = sorted([str(p) for p in out_dir.glob("frame_*.png")])
        if len(pngs) >= expected:
            print(f"[PHASE-1] Global frames cache hit: found {len(pngs)} >= expected {expected}")
            return pngs
        shutil.rmtree(out_dir, ignore_errors=True)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    print(f"[PHASE-1] Start global sampling: {len(times)} frames")
    image_paths: List[str] = []
    
    for idx, t in enumerate(tqdm(times, desc=f"Global sampling ({len(times)} frames)"), start=1):
        frame = read_frame_by_time(cap, t, fps, total_frames, duration)
        if frame is None:
            continue
        frame = resize_keep_area(frame)
        img_path = out_dir / f"frame_{idx:04d}.png"
        cv2.imwrite(str(img_path), frame)
        image_paths.append(str(img_path))
    
    cap.release()
    return image_paths


def plan_segment_times(s: float, e: float) -> List[float]:
    """Plan segment sampling times."""
    seg = max(e - s, 0.0)
    if seg <= 0:
        return [s]
    if seg > 512:
        return [s + seg * i / 512.0 for i in range(512)]
    t, times = s, []
    while t < e - 1e-6:
        times.append(t)
        t += 1.0
    if not times:
        times = [s]
    return times


def ensure_segment_frames(video_path: str, seg_dir: Path, s: float, e: float) -> List[str]:
    """Ensure segment frames are extracted."""
    fps, total_frames, duration = get_video_meta(video_path)
    times = plan_segment_times(s, e)
    expected = len(times)
    
    if seg_dir.exists():
        pngs = sorted([str(p) for p in seg_dir.glob("frame_*.png")])
        if len(pngs) >= expected:
            print(f"[PHASE-2] Segment frames cache hit: found {len(pngs)} >= expected {expected}")
            return pngs
        shutil.rmtree(seg_dir, ignore_errors=True)
    
    seg_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    print(f"[PHASE-2] Start segment sampling: {len(times)} frames for [{s:.3f}, {e:.3f}]")
    image_paths: List[str] = []
    
    for idx, t in enumerate(tqdm(times, desc=f"Segment {s:.3f}-{e:.3f}"), start=1):
        frame = read_frame_by_time(cap, t, fps, total_frames, duration)
        if frame is None:
            continue
        frame = resize_keep_area(frame)
        img_path = seg_dir / f"frame_{idx:04d}.png"
        cv2.imwrite(str(img_path), frame)
        image_paths.append(str(img_path))
    
    cap.release()
    return image_paths


# ==================== Main Processing ====================
def decide_rounds_for_video(video_name: str, duration_sec: float) -> int:
    """Decide number of tool call rounds based on video duration."""
    L_MIN_FOR_TWO = 15 * 60  # 15 minutes
    L_MAX_FOR_TWO = 20 * 60  # 20 minutes
    
    if not math.isfinite(duration_sec) or duration_sec < L_MIN_FOR_TWO:
        return 1
    
    x = max(L_MIN_FOR_TWO, min(duration_sec, L_MAX_FOR_TWO))
    p2 = (x - L_MIN_FOR_TWO) / max(1e-9, (L_MAX_FOR_TWO - L_MIN_FOR_TWO))
    rng = random.Random((hash(video_name) & 0xFFFFFFFF))
    return 2 if rng.random() < p2 else 1


def run_imcott_generation(
    vars_dict: Dict[str, Any],
    output_dir: Path,
    global_frames_dir: Path,
    segment_frames_dir: Path,
    api_client: Any,
    rate_limiter: RateLimiter,
) -> Dict[str, Any]:
    """
    Run iMCoTT generation for a single QA item.
    
    This is a simplified version that demonstrates the pipeline structure.
    For full implementation, integrate with your preferred LLM API (Gemini, OpenAI, etc.)
    """
    total_t0 = time.time()
    video_path = vars_dict["VIDEO_PATH"]
    print(f"[START] Pipeline for {video_path}")
    
    # Extract global frames
    video_stem = Path(video_path).stem
    global_dir = global_frames_dir / video_stem
    
    sample_t0 = time.time()
    global_image_paths = ensure_global_frames(video_path, global_dir)
    sample_t1 = time.time()
    print(f"[PHASE-1] Global sampling -> {len(global_image_paths)} frames in {fmt_dur(sample_t0, sample_t1)}")
    
    # Prepare PHASE-1 prompt
    global_prompt_text = Template(SYSTEM_PROMPT + "\n" + GLOBAL_SKIM_PROMPT).substitute(vars_dict)
    
    # Ground truth time hints
    gt = vars_dict.get("GROUND_TRUTH_TIME") or []
    s_int = e_int = None
    if isinstance(gt, (list, tuple)) and len(gt) == 2 and gt[1] > gt[0]:
        s_int = float(int(round(gt[0])))
        e_int = float(int(round(gt[1])))
    
    # Here you would call your LLM API for PHASE-1
    # This is a placeholder that shows the expected structure
    phase1_output = f"""<question>{vars_dict['QUESTION']}</question>
<think>
The video shows a sequence of events. Based on the visual content, 
I observe the main action occurring around the middle portion of the video.
The key evidence appears to be in the segment [{s_int:.2f}s – {e_int:.2f}s] 
where the relevant visual information is visible.
</think>"""
    
    # Extract windows from PHASE-1
    k = int(vars_dict.get("PREFERRED_TOOL_CALLS", 1))
    windows = [(s_int, e_int)] if s_int is not None else [(0.0, 10.0)]
    
    transcript = phase1_output
    
    # PHASE-2: Fine-grained inspection
    for idx, (s, e) in enumerate(windows, 1):
        print(f"\n--- PHASE-2 Round {idx}/{k} | window = [{s:.3f}, {e:.3f}] ---")
        
        seg_dir_name = f"{video_stem}_{s:.3f}_{e:.3f}"
        segment_dir = segment_frames_dir / seg_dir_name
        
        sample_t0 = time.time()
        segment_image_paths = ensure_segment_frames(video_path, segment_dir, s, e)
        sample_t1 = time.time()
        print(f"[PHASE-2] Segment sampling -> {len(segment_image_paths)} frames in {fmt_dur(sample_t0, sample_t1)}")
        
        # Prepare PHASE-2 prompt
        s_fmt = f"{s:.2f}" if k == 1 else f"{s:.3f}"
        e_fmt = f"{e:.2f}" if k == 1 else f"{e:.3f}"
        
        round_prompt = Template(FINE_INSPECTION_PROMPT).substitute({
            "ROUND_IDX": idx,
            "TOTAL_ROUNDS": k,
            "QUESTION": vars_dict["QUESTION"],
            "VIDEO_PATH": vars_dict["VIDEO_PATH"],
            "S": s_fmt,
            "E": e_fmt,
            "TRANSCRIPT_SO_FAR": transcript,
            "GROUND_TRUTH_ANSWER": vars_dict["GROUND_TRUTH_ANSWER"],
            "PREFERRED_TOOL_CALLS": vars_dict["PREFERRED_TOOL_CALLS"],
        })
        
        # Here you would call your LLM API for PHASE-2
        # This is a placeholder
        round_output = f"""<tool_call>
{{ "name":"crop_video", "arguments":{{ "video_path":"{vars_dict['VIDEO_PATH']}", "start_time": {s_fmt}, "end_time": {e_fmt} }} }}
</tool_call>
<tool_response>
The tool executed successfully.
</tool_response>
<think>
Examining the segment [{s_fmt}s – {e_fmt}s], I can see the visual evidence 
that helps answer the question. The content confirms my earlier hypothesis.
</think>"""
        
        if idx == k:
            round_output += f"""
<answer>
{vars_dict['GROUND_TRUTH_ANSWER']}
</answer>"""
        
        transcript += "\n" + round_output
    
    total_t1 = time.time()
    print(f"[END] Pipeline finishes in {fmt_dur(total_t0, total_t1)}")
    
    return {
        "windows": windows,
        "transcript": transcript,
        "total_elapsed_sec": total_t1 - total_t0,
    }


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="iMCoTT Generation for Video QA")
    parser.add_argument("--input-file", type=str, required=True, help="Input JSON file with QA data")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for generated traces")
    parser.add_argument("--video-root", type=str, required=True, help="Root directory for video files")
    parser.add_argument("--global-frames-dir", type=str, default="./global_sampling", help="Directory for global frames cache")
    parser.add_argument("--segment-frames-dir", type=str, default="./segment_sampling", help="Directory for segment frames cache")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    video_root = Path(args.video_root)
    global_frames_dir = Path(args.global_frames_dir)
    segment_frames_dir = Path(args.segment_frames_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load input data
    if not input_path.exists():
        print(f"[EXIT] Input file not found: {input_path}")
        return
    
    try:
        data = json.loads(input_path.read_text("utf-8"))
        if not isinstance(data, list):
            data = [data]
    except Exception as e:
        print(f"[EXIT] Failed to load input file: {e}")
        return
    
    # Build video name to path mapping
    print("[INFO] Scanning video root for video files...")
    name2path = {}
    for p in video_root.rglob("*.mp4"):
        name2path[p.name] = str(p.resolve())
    print(f"[INFO] Indexed {len(name2path)} mp4 files")
    
    # Initialize rate limiter
    rate_limiter = RateLimiter(GEMINI_RPM, GEMINI_TPM)
    
    # Process each QA item
    for idx, item in enumerate(tqdm(data, desc="Processing QA items")):
        try:
            question_text = str(item.get("question", "")).strip()
            answer_text = str(item.get("answer", "")).strip()
            s = float(item.get("qa_start_time", 0))
            e = float(item.get("qa_end_time", 0))
            
            if not question_text or not answer_text or not (e > s):
                continue
            
            vname = str(item.get("video_name", "unknown.mp4"))
            if not vname.endswith(".mp4"):
                vname += ".mp4"
            
            video_path = name2path.get(Path(vname).name)
            if not video_path:
                print(f"[SKIP] Video not found: {vname}")
                continue
            
            # Get video duration
            try:
                _, _, dur = get_video_meta(video_path)
            except Exception:
                dur = float("nan")
            
            calls = decide_rounds_for_video(vname, dur)
            
            # Check if already processed
            video_stem = Path(video_path).stem
            out_txt = output_dir / f"{video_stem}_qa{idx:06d}_{calls}_round_output.txt"
            out_sum = output_dir / f"{video_stem}_qa{idx:06d}_{calls}_summary.json"
            
            if out_txt.exists() and out_sum.exists():
                continue
            
            # Prepare variables
            local_vars = {
                "VIDEO_PATH": video_path,
                "QUESTION": question_text,
                "GROUND_TRUTH_ANSWER": answer_text,
                "GROUND_TRUTH_TIME": [s, e],
                "PREFERRED_TOOL_CALLS": calls,
            }
            
            # Run generation
            result = run_imcott_generation(
                local_vars,
                output_dir,
                global_frames_dir,
                segment_frames_dir,
                None,  # API client placeholder
                rate_limiter,
            )
            
            # Save outputs
            out_txt.write_text(result["transcript"], encoding="utf-8")
            summary = {
                "input_file": str(input_path),
                "video": video_path,
                "calls": calls,
                "windows": result.get("windows"),
                "total_elapsed_sec": result.get("total_elapsed_sec"),
                "qa_index": idx,
                "qa_start_time": s,
                "qa_end_time": e,
            }
            out_sum.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            
        except Exception as e:
            print(f"[ERROR] idx={idx}: {e}")
            continue
    
    print("[DONE] iMCoTT generation complete!")


if __name__ == "__main__":
    main()

