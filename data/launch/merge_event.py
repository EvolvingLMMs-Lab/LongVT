import argparse
import collections
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constant import MERGE_EVENT_PROMPT
from server import SERVER_MAPPING
from server.openai import ChatCompletionRequest

model = "judge"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--server", type=str, required=False, default="openai")
    return parser.parse_args()


def create_event_messages(caption1, caption2):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": MERGE_EVENT_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": f"Caption 1: {caption1}\nCaption 2: {caption2}"}]},
    ]
    request = ChatCompletionRequest(
        model=model,
        messages=messages,
        max_tokens=16,
        temperature=0.7,
    )
    return request


def merge_two_events(event1, event2, server):
    caption1 = event1["caption"][0]
    caption2 = event2["caption"][0]
    request = create_event_messages(caption1, caption2)
    response = server.chat_completion(request)
    response = response.choices[0]["message"]["content"].strip()
    if "yes" in response.lower():
        new_event = {
            "start_time": event1["start_time"],
            "end_time": event2["end_time"],
            "caption": [caption1, caption2],
            "video_path": event1["video_path"],
        }
        return [new_event]
    else:
        return [event1, event2]


def merge_events(events, server_name):
    server = SERVER_MAPPING[server_name]()
    # Edge case: only one event, return
    if len(events) == 1:
        return events

    # Edge case: only two events, merge them
    if len(events) == 2:
        return merge_two_events(events[0], events[1], server)

    # If more than two events, split into two parts and merge them recursively
    if len(events) > 2:
        # Split into two parts and merge them recursively
        mid = len(events) // 2
        event1 = events[:mid]
        event2 = events[mid:]
        event1 = merge_events(event1, server_name)
        event2 = merge_events(event2, server_name)

        # Compare the boundary of the two events
        event1_end_seg = event1[-1]
        event2_start_seg = event2[0]
        event1 = event1[:-1]
        event2 = event2[1:]
        merged_events = merge_two_events(event1_end_seg, event2_start_seg, server)
        return event1 + merged_events + event2

    return events


def run(video2caption, server_name):
    results = []
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(merge_events, events, server_name) for events in video2caption.values()]
        pbar = tqdm(total=len(futures), desc="Merging events")
        for future in futures:
            results.extend(future.result())
            pbar.update(1)
        pbar.close()
    return results


def main():
    args = parse_args()
    input_path = args.input_path
    output_path = args.output_path

    with open(input_path) as f:
        input_data = json.load(f)

    video2caption = collections.defaultdict(list)
    for item in input_data:
        item["caption"] = [item["caption"]]
        video2caption[item["video_path"]].append(item)

    results = run(video2caption, args.server)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
