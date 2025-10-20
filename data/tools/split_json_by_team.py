import argparse
import collections
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.constant import Team


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.input_path) as f:
        data = json.load(f)
    output_list = collections.defaultdict(list)
    for item in data:
        output_list[item["video_path"]].append(item)

    video_path_by_team = collections.defaultdict(list)
    video_path_list = list(output_list.keys())
    video_path_list.sort()
    team_name = [team.value for team in Team]
    for i, video_path in enumerate(video_path_list):
        video_path_by_team[team_name[i % len(Team)]].append(video_path)

    # Make dir for each team and save the path according to the team
    for team in Team:
        os.makedirs(os.path.join(args.output_dir, team.value), exist_ok=True)
        for video_path in video_path_by_team[team.value]:
            video_base_path = os.path.basename(video_path).split(".")[0]
            with open(os.path.join(args.output_dir, team.value, f"{video_base_path}.json"), "w") as f:
                json.dump(output_list[video_path], f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
