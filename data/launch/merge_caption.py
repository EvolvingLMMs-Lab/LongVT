import argparse
import asyncio
import json
import os
import sys

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constant import MERGE_CAPTION_PROMPT
from server import SERVER_MAPPING
from server.openai import ChatCompletionRequest

model = "judge"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--server", type=str, required=False, default="openai")
    return parser.parse_args()


def create_caption_messages(caption1, caption2):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": MERGE_CAPTION_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": f"Caption 1: {caption1}\nCaption 2: {caption2}"}]},
    ]
    request = ChatCompletionRequest(
        model=model,
        messages=messages,
        max_tokens=32768,
        temperature=0.7,
    )
    return request


async def merge_caption(event, server):
    if len(event["caption"]) == 1:
        event["caption"] = event["caption"][0]
        return event

    if len(event["caption"]) == 2:
        request = create_caption_messages(event["caption"][0], event["caption"][1])
        response = await server.chat_completion_async(request)
        event["caption"] = response.choices[0]["message"]["content"].strip()
        return event

    if len(event["caption"]) > 2:
        mid = len(event["caption"]) // 2
        event1 = event["caption"][:mid]
        event2 = event["caption"][mid:]
        event1 = await merge_caption(event1, server)
        event2 = await merge_caption(event2, server)
        event["caption"] = event1["caption"] + event2["caption"]
        return event

    return event


def merge_caption_sync(event, server):
    if len(event["caption"]) == 1:
        event["caption"] = event["caption"][0]
        return event

    if len(event["caption"]) == 2:
        request = create_caption_messages(event["caption"][0], event["caption"][1])
        response = server.chat_completion(request)
        event["caption"] = response.choices[0]["message"]["content"].strip()
        return event

    if len(event["caption"]) > 2:
        mid = len(event["caption"]) // 2
        event1 = event["caption"][:mid]
        event2 = event["caption"][mid:]
        event1 = merge_caption_sync(event1, server)
        event2 = merge_caption_sync(event2, server)
        event["caption"] = event1["caption"] + event2["caption"]
        return event

    return event


async def run_caption(events, server):
    sem = asyncio.Semaphore(32)

    async def process(event):
        async with sem:
            return await merge_caption(event, server)

    tasks = [asyncio.create_task(process(event)) for event in events]
    pbar = tqdm(total=len(tasks), desc="Merging captions")
    results = []
    for completed_task in asyncio.as_completed(tasks):
        result = await completed_task
        results.append(result)
        pbar.update(1)
    pbar.close()
    return results


def main():
    args = parse_args()
    input_path = args.input_path
    output_path = args.output_path

    with open(input_path) as f:
        input_data = json.load(f)

    server = SERVER_MAPPING[args.server]()

    results = asyncio.run(run_caption(input_data, server))
    # Sort result by video path and start time
    results.sort(key=lambda x: (x["video_path"], x["start_time"]))
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
