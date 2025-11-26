#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clip Caption Module

This module generates captions for video clips using VLM services.

Usage:
    python launch/clip_caption.py --input_path detect_results.json --output_path captions.json
"""

import argparse
import asyncio
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import SERVER_MAPPING
from torchvision.transforms.functional import to_pil_image

from data.constant import VIDEO_CAPTION_PROMPT
from data.server.openai import ChatCompletionRequest
from data.utils import encode_image, process_video


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--server", type=str, required=False, default="openai")
    parser.add_argument("--fps", type=int, required=False, default=1)
    parser.add_argument("--limit", type=int, required=False, default=None)
    parser.add_argument("--shard-size", type=int, required=False, default=1)
    parser.add_argument("--shard-index", type=int, required=False, default=0)
    return parser.parse_args()


def create_messages(video_path, start_time, end_time, fps):
    frames = process_video(video_path, start_time=start_time, end_time=end_time, fps=fps)
    frames = frames.to(torch.uint8)
    frame_pil = [to_pil_image(frame) for frame in frames]
    frames = [encode_image(frame) for frame in frame_pil]
    messages = [{"role": "user", "content": [{"type": "text", "text": VIDEO_CAPTION_PROMPT}]}]
    for frame in frames:
        messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{frame}"}})
    return messages


async def generate_caption(task_list, server):
    sem = asyncio.Semaphore(cpu_count() // 2)
    messages, video_path, start_time, end_time = task_list

    async with sem:
        request = ChatCompletionRequest(
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            messages=messages,
            max_tokens=32768,
            temperature=0.7,
        )
        response = await server.chat_completion_async(request)
        choices = response.choices[0]
        caption_dict = {
            "start_time": start_time,
            "caption": choices["message"]["content"],
            "end_time": end_time,
            "video_path": video_path,
        }
        return caption_dict


def generate_caption_sync(task_list, server):
    messages, video_path, start_time, end_time = task_list
    request = ChatCompletionRequest(
        model="Qwen/Qwen2.5-VL-72B-Instruct",
        messages=messages,
        max_tokens=32768,
        temperature=0.7,
    )
    response = server.chat_completion(request)
    choices = response.choices[0]
    caption_dict = {
        "start_time": start_time,
        "caption": choices["message"]["content"],
        "end_time": end_time,
        "video_path": video_path,
    }
    return caption_dict


def run(video_path, start_time, end_time, fps, server_name):
    server = SERVER_MAPPING[server_name]()
    try:
        messages = create_messages(video_path, start_time, end_time, fps)
    except Exception as e:
        print(f"Error creating messages for {video_path} at {start_time} - {end_time}: {e}")
        return None

    request = ChatCompletionRequest(
        model="Qwen/Qwen2.5-VL-72B-Instruct",
        messages=messages,
        max_tokens=32768,
        temperature=0.7,
    )
    response = server.chat_completion(request)
    choices = response.choices[0]
    caption_dict = {
        "start_time": start_time,
        "caption": choices["message"]["content"],
        "end_time": end_time,
        "video_path": video_path,
    }
    return caption_dict


def main():
    args = parse_args()
    input_path = args.input_path

    with open(input_path) as f:
        input_data = json.load(f)

    if args.limit is not None:
        input_data = input_data[: args.limit]

    if args.shard_size > 1:
        input_data = input_data[args.shard_index :: args.shard_size]
        output_path = f"{args.output_path.replace('.json', f'_shard_{args.shard_index}.json')}"
    else:
        output_path = args.output_path

    with ProcessPoolExecutor(max_workers=cpu_count() - 8) as executor:
        futures = []
        results = []
        info_list = []
        for item in input_data:
            video_path = item["video_path"]
            start_time = item["start_time"]
            end_time = item["end_time"]
            if end_time - start_time > 10:
                # Split into 10s segments
                for i in range(0, int(end_time - start_time), 10):
                    futures.append(
                        executor.submit(
                            run,
                            video_path,
                            start_time=start_time + i,
                            end_time=start_time + i + 10,
                            fps=args.fps,
                            server_name=args.server,
                        )
                    )
                    end_time = start_time + i + 10
                    info_list.append((video_path, start_time + i, end_time))
            else:
                futures.append(
                    executor.submit(
                        run,
                        video_path,
                        start_time=start_time,
                        end_time=end_time,
                        fps=args.fps,
                        server_name=args.server,
                    )
                )
                info_list.append((video_path, start_time, end_time))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
            if future.result() is not None:
                results.append(future.result())

    # Sort results by video_path and start_time
    results.sort(key=lambda x: (x["video_path"], x["start_time"]))

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
