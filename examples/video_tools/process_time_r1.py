# Copyright 2025 Individual Contributor: Kaichen Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os

from datasets import Dataset
from qwen_vl_utils import fetch_video
from torchvision.transforms.functional import to_pil_image

SYSTEM_PROMPT = (
    "You are an AI assistant designed to identify the most relevant temporal segment in a "
    "video that corresponds to a given natural language description."
    " Your task is to analyze the video and provide a precise time range that best matches the description."
    " Return your answer as a list of two floats: [start_time, end_time]"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Process video time data.")
    parser.add_argument("--video_data_folder", type=str, required=True, help="Path to the video data folder.")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second for video processing.")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Set this to limit the number of frames per video. Default is None, which means will use fps=1 by default",
    )
    parser.add_argument(
        "--max_pixels", type=int, default=151200, help="Maximum number of pixels for video frames. Default is 360 x 420"
    )
    parser.add_argument("--local_path", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output JSON file.",
        default="~/data/video_time_r1.parquet",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples to process. Default is None, which means all samples will be processed.",
    )
    return parser.parse_args()


def load_video(video_path, max_pixels, fps, nframes):
    """
    Load a video from the specified path and return its frames.
    """
    video_dict = {
        "video": video_path,
        "max_pixels": max_pixels,
        "fps": fps,
    }

    if nframes is not None:
        video_dict["nframes"] = nframes
        video_dict.pop("fps", None)  # Remove fps if nframes is specified
    video = fetch_video(video_dict)
    return [to_pil_image(frame) for frame in video]


def wrapper(args):
    video_path, max_pixels, fps, nframes = args
    return load_video(video_path, max_pixels, fps, nframes)


def main():
    args = parse_args()

    # Load the JSON data
    with open(args.local_path) as f:
        data = json.load(f)

    if args.limit is not None:
        data = data[: args.limit]

    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"The final answer MUST BE put in <answer> </answer> tags."
    )

    # with Pool(processes=16) as pool:
    # # Prepare video paths for processing
    # video_paths = [
    # (
    # os.path.join(args.video_data_folder, os.path.basename(da["video"])),
    # args.max_pixels,
    # args.fps,
    # args.max_frames,
    # )
    # for da in data
    # ]

    # # Load videos in parallel
    # video_frames = list(tqdm(pool.imap(wrapper, video_paths), total=len(video_paths), desc="Loading videos"))

    def gen():
        for idx, da in enumerate(data):
            video_path = da["video"]
            # Origin Json has appended video path, we need to extract the basename
            video_path = os.path.basename(video_path)
            true_video_path = os.path.join(args.video_data_folder, video_path)
            question = da["sentence"]
            answer = da["pred"]

            prompt = "<video>" + f"{question} {instruction_following}"
            prompt += "\n The Video path for this video is: " + true_video_path
            if not os.path.exists(true_video_path):
                print(f"Video file {true_video_path} does not exist.")
                continue

            video_dict = {
                "type": "video",
                "video": f"file://{true_video_path}",
                "fps": args.fps,
                "min_frames": 1,
                "max_frames": args.max_frames,
                "max_pixels": 360 * 420,
            }
            if args.max_frames is None:
                video_dict.pop("max_frames", None)

            data_dict = {
                "data_source": "TimeR1",
                "prompt": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "videos": [video_dict],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": "train",
                    "index": idx,
                    "answer": answer,
                    "question": question,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "crop_video": {
                            "create_kwargs": {"dummy": "dummy"},
                            # "execute_kwargs": {},
                            # "calc_reward_kwargs": {},
                            # "release_kwargs": {},
                        },
                    },
                },
            }
            yield data_dict

    dataset = Dataset.from_generator(
        gen,
    )
    # frames = process_video(dataset[0]["videos"][0])
    print(dataset[0])
    dataset.to_parquet(args.output_path)


if __name__ == "__main__":
    main()
