import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_path = args.output_path

    data_list = []

    for file in os.listdir(input_dir):
        if file.endswith(".json"):
            with open(os.path.join(input_dir, file)) as f:
                data = json.load(f)
                data_list.extend(data)

    # Sort data_list by video_path, start_time
    data_list.sort(key=lambda x: (x["video_path"], x["start_time"]))
    with open(output_path, "w") as f:
        json.dump(data_list, f, indent=4)


if __name__ == "__main__":
    main()
