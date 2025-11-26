#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect Segment Module

This module detects scene boundaries in long videos and segments them into clips.

Usage:
    python launch/detect_segment.py --input_file /path/to/video_list.txt --output_path results.json
"""

import argparse
import collections
import glob
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
from utils import detect_scenes, get_video_length


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=False)
    parser.add_argument("--input_file", type=str, required=False)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_parts", type=int, required=False, default=10)
    return parser.parse_args()


def write_scenes(scenes, output_path):
    output_list = []
    for video_path, scene_list in scenes.items():
        for scene in scene_list:
            start_time, end_time = scene
            output_list.append(
                {
                    "video_path": video_path,
                    "start_time": start_time.get_seconds(),
                    "end_time": end_time.get_seconds(),
                }
            )

    # Sort by start time, given the video_path is the same, sort by start_time
    output_list.sort(key=lambda x: (x["video_path"], x["start_time"]))
    with open(output_path, "w") as f:
        json.dump(output_list, f, indent=4)


# Split video into equal parts
def split_video(video_length, num_parts):
    part_length = video_length / num_parts
    time_chunks = []
    for i in range(num_parts):
        start_time = i * part_length
        end_time = start_time + part_length
        time_chunks.append((start_time, end_time))
    return time_chunks


def main():
    args = parse_args()
    if args.input_dir:
        input_files = glob.glob(os.path.join(args.input_dir, "*.mp4"))
    elif args.input_file:
        # Input file is a txt file, each line is a video path
        with open(args.input_file) as f:
            input_files = f.readlines()
            input_files = [line.strip() for line in input_files]
    else:
        raise ValueError("Either --input_dir or --input_file must be provided")

    output_dict = {}

    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(get_video_length, input_file) for input_file in input_files]
        for idx, future in tqdm(enumerate(as_completed(futures)), total=len(futures), desc="Getting video lengths"):
            video_length = future.result()
            if video_length > 0:
                output_dict[input_files[idx]] = video_length

    task_list = []
    for input_file, video_length in output_dict.items():
        time_chunks = split_video(video_length, args.num_parts)
        time_chunks[-1] = (time_chunks[-1][0], time_chunks[-1][1] - 1)  # Remove the last 1s to avoid overflow
        for start_time, end_time in time_chunks:
            task_list.append((input_file, start_time, end_time))

    output_scenes = collections.defaultdict(list)
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(detect_scenes, input_file, start_time, end_time)
            for input_file, start_time, end_time in task_list
        ]
        for idx, future in tqdm(enumerate(as_completed(futures)), total=len(futures), desc="Detecting scenes"):
            scenes = future.result()
            video_name = task_list[idx][0]
            output_scenes[video_name].extend(scenes)

    write_scenes(output_scenes, args.output_path)


if __name__ == "__main__":
    main()
