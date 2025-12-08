#!/usr/bin/env python3
"""
LongVT Single Sample Inference

Demonstrates how to perform single-sample inference with LongVT model.
Requires a running vLLM server with tool calling enabled.

Requirements:
    1. Start vLLM server with tool calling support (see run_single_inference.sh)
    2. Install dependencies: pip install openai qwen-vl-utils torch torchvision opencv-python

Usage:
    python single_inference.py \
        --video_path /path/to/video.mp4 \
        --question "What is happening in the video?" \
        --api_base http://localhost:8000/v1

Parameters align with eval benchmark settings:
    - fps: 1 (sample 1 frame per second)
    - max_frames: 512 (maximum frames to extract)
    - max_pixels: 50176 (224*224, frame resolution)
"""

import argparse
import base64
import json
import os
import sys
from io import BytesIO
from typing import Optional

from openai import OpenAI


# ============== Tool Definition ==============
def crop_video_local(video_path: str, start_time: float, end_time: float) -> list:
    """
    Crop video and return base64 encoded frames.
    Mirrors the behavior of MCP server's crop_video tool.
    """
    import cv2
    import torch
    from PIL import Image
    from qwen_vl_utils import fetch_video
    from torchvision.transforms.functional import to_pil_image

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Get video duration
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    if start_time >= duration:
        raise ValueError(f"start_time ({start_time}s) exceeds video duration ({duration:.2f}s)")
    if end_time > duration:
        end_time = duration

    video_ele = {
        "type": "video",
        "video": f"file://{video_path}",
        "fps": 1,
        "min_frames": 1,
        "max_frames": 128,
        "min_pixels": 28 * 28,
        "max_pixels": 224 * 224,
        "video_start": start_time,
        "video_end": end_time,
    }
    video_frames = fetch_video(video_ele)
    video_frames = video_frames.to(torch.uint8)
    images = [to_pil_image(frame) for frame in video_frames]

    image_contents = []
    for img in images:
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        image_contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })

    return image_contents


# ============== Tool Schema ==============
CROP_VIDEO_TOOL = {
    "type": "function",
    "function": {
        "name": "crop_video",
        "description": "Crop a video to a specified duration. Use this tool to zoom in on specific time segments for detailed analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "video_path": {"type": "string", "description": "Path to the video file"},
                "start_time": {"type": "number", "description": "Start time in seconds"},
                "end_time": {"type": "number", "description": "End time in seconds"}
            },
            "required": ["video_path", "start_time", "end_time"]
        }
    }
}


# ============== Prompts ==============
# For tool calling mode (default)
TOOL_PROMPT = (
    "Think first, call **crop_video** if needed, then answer. "
    "Format strictly as: <think>...</think> <tool_call>...</tool_call> (if needed) <answer>...</answer>."
)

# For reasoning mode without tool calling (--no_tool)
SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags. "
    "Please provide a clear, concise response within <answer></answer> tags that directly addresses the question."
)


def encode_video_frames(video_path: str, fps: int = 1, max_frames: int = 512, max_pixels: int = 50176) -> list:
    """
    Encode video frames to base64 for model input.
    
    Args:
        video_path: Path to video file
        fps: Frames per second for sampling (default: 1)
        max_frames: Maximum number of frames to extract (default: 512)
        max_pixels: Maximum pixels per frame, e.g. 224*224=50176 (default: 50176)
    
    Note: These parameters should match eval benchmark settings:
        fps=1, max_frames=512, max_pixels=50176 (224*224)
    """
    import torch
    from qwen_vl_utils import fetch_video
    from torchvision.transforms.functional import to_pil_image

    video_ele = {
        "type": "video",
        "video": f"file://{video_path}",
        "fps": fps,
        "min_frames": 1,
        "max_frames": max_frames,
        "min_pixels": 28 * 28,  # 784
        "max_pixels": max_pixels,  # 224*224 = 50176
    }
    video_frames = fetch_video(video_ele)
    video_frames = video_frames.to(torch.uint8)
    images = [to_pil_image(frame) for frame in video_frames]

    image_contents = []
    for img in images:
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        image_contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })

    return image_contents


def run_inference(
    video_path: str,
    question: str,
    api_base: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    model_name: Optional[str] = None,
    max_tokens: int = 4096,
    fps: int = 1,
    max_frames: int = 512,
    max_pixels: int = 50176,
    max_tool_rounds: int = 5,
    enable_tool: bool = True,
    verbose: bool = True,
):
    """
    Run single sample inference with LongVT model.
    
    Args:
        video_path: Path to the video file
        question: Question about the video
        api_base: vLLM server API base URL (default: http://localhost:8000/v1)
        api_key: API key (default: EMPTY for local vLLM)
        model_name: Model name (auto-detected from server if None)
        max_tokens: Maximum tokens for generation (default: 4096)
        fps: Frames per second for video encoding (default: 1, matches eval benchmark)
        max_frames: Maximum number of frames to encode (default: 512, matches eval benchmark)
        max_pixels: Maximum pixels per frame (default: 50176=224*224, matches eval benchmark)
        max_tool_rounds: Maximum number of tool calling iterations (default: 5)
        enable_tool: Enable crop_video tool for global-to-local reasoning (default: True)
        verbose: Print detailed progress logs (default: True)
    
    Returns:
        dict: Contains 'response' (final text), 'tool_calls' (history), 'num_rounds'
    """
    client = OpenAI(api_key=api_key, base_url=api_base)
    
    if model_name is None:
        models = client.models.list()
        model_name = models.data[0].id
        if verbose:
            print(f"[INFO] Using model: {model_name}")

    if verbose:
        print(f"[INFO] Encoding video: {video_path}")
    video_contents = encode_video_frames(video_path, fps=fps, max_frames=max_frames, max_pixels=max_pixels)
    if verbose:
        print(f"[INFO] Encoded {len(video_contents)} frames")

    user_content = video_contents + [
        {"type": "text", "text": f"{question} {TOOL_PROMPT} The Video path for this video is: {video_path}" if enable_tool else question}
    ]
    
    # Build messages: add system prompt for non-tool mode
    if enable_tool:
        messages = [{"role": "user", "content": user_content}]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
    
    all_responses = []
    tool_call_history = []
    
    for round_idx in range(max_tool_rounds + 1):
        if verbose:
            print(f"\n[ROUND {round_idx}] Calling model...")
        
        kwargs = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        
        if enable_tool:
            kwargs["tools"] = [CROP_VIDEO_TOOL]
            kwargs["tool_choice"] = "auto"
        
        response = client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        
        if message.content:
            all_responses.append(message.content)
            if verbose:
                content = message.content[:500] + "..." if len(message.content) > 500 else message.content
                print(f"[RESPONSE] {content}")
        
        if finish_reason == "tool_calls" and message.tool_calls:
            if verbose:
                print(f"[TOOL CALLS] Processing {len(message.tool_calls)} call(s)")
            
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    } for tc in message.tool_calls
                ]
            })
            
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                if verbose:
                    print(f"[TOOL] {func_name}: {func_args}")
                
                tool_call_history.append({"name": func_name, "arguments": func_args})
                
                if func_name == "crop_video":
                    try:
                        result_images = crop_video_local(
                            video_path=func_args["video_path"],
                            start_time=func_args["start_time"],
                            end_time=func_args["end_time"]
                        )
                        if verbose:
                            print(f"[TOOL RESULT] Cropped {len(result_images)} frames")
                        
                        tool_content = result_images + [
                            {"type": "text", "text": f"Cropped {func_args['start_time']}s-{func_args['end_time']}s, got {len(result_images)} frames."}
                        ]
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_content
                        })
                    except Exception as e:
                        if verbose:
                            print(f"[TOOL ERROR] {e}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": [{"type": "text", "text": f"Error: {e}"}]
                        })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": [{"type": "text", "text": f"Unknown tool: {func_name}"}]
                    })
        else:
            break
    
    return {
        "response": "\n".join(all_responses),
        "tool_calls": tool_call_history,
        "num_rounds": round_idx + 1,
    }


def main():
    parser = argparse.ArgumentParser(description="LongVT Single Sample Inference")
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file")
    parser.add_argument("--question", type=str, required=True, help="Question about the video")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1", help="vLLM server URL")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API key")
    parser.add_argument("--model_name", type=str, default=None, help="Model name (auto-detected if not set)")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max generation tokens")
    parser.add_argument("--fps", type=int, default=1, help="Video sampling FPS")
    parser.add_argument("--max_frames", type=int, default=512, help="Max frames to encode")
    parser.add_argument("--max_pixels", type=int, default=50176, help="Max pixels per frame (224*224=50176)")
    parser.add_argument("--no_tool", action="store_true", help="Disable tool calling")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video not found: {args.video_path}")
        sys.exit(1)
    
    result = run_inference(
        video_path=args.video_path,
        question=args.question,
        api_base=args.api_base,
        api_key=args.api_key,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        fps=args.fps,
        max_frames=args.max_frames,
        max_pixels=args.max_pixels,
        enable_tool=not args.no_tool,
        verbose=not args.quiet,
    )
    
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Tool calls: {len(result['tool_calls'])}")
    for i, tc in enumerate(result['tool_calls']):
        print(f"  [{i+1}] {tc['name']}: {tc['arguments']}")
    print(f"Rounds: {result['num_rounds']}")
    print("-" * 60)
    print(result['response'])


if __name__ == "__main__":
    main()

