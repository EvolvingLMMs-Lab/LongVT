#!/usr/bin/env python3
"""
main.py: Data Pipeline 脚本。
包含 Data Filtering、Video Scene Detect & Segmentation、Caption Generation、Event Grouping、Timestamp Refinement、Output Finalization等 modules。
"""
import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from decord import VideoReader, cpu
from openai import OpenAI
from tqdm import tqdm

try:
    import cv2
except ImportError as e:
    raise ImportError("需要安装 OpenCV 库。请运行: pip install opencv-python") from e
try:
    import openai
except ImportError as e:
    # OpenAI API library is required for calling Qwen series model
    raise ImportError("需要安装 OpenAI API 库。请运行: pip install openai") from e
try:
    from scenedetect import ContentDetector, detect
except ImportError as e:
    # PySceneDetect is required for scene detection
    raise ImportError("需要安装 PySceneDetect 库。请运行: pip install scenedetect") from e

# 性能调研：OpenCV vs FFmpeg帧提取
# OpenCV 的 VideoCapture 使用 C++ 封装的 FFmpeg libavcodec 解码帧:contentReference[oaicite:0]{index=0}，
# 而 ffmpeg-python 等通过子进程管道读帧会引入额外的上下文切换和数据传输开销:contentReference[oaicite:1]{index=1}。
# 实测中 OpenCV 逐帧读取可达约692 FPS，而通过管道方式约373 FPS:contentReference[oaicite:2]{index=2}。
# 因此在密集视频处理任务中，直接使用 OpenCV 读取视频帧具有较高性能和便利性。

# 配置日志格式
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def filter_videos(input_dir, output_dir):
    """
    数据筛选模块：
    遍历输入目录中的视频文件，按要求将视频归类到不同目录：
      - 时长小于60秒的视频归入 short 目录
      - 分辨率（宽和高）都小于480的视频归入 low 目录
      - 同时满足上述两个条件的归入 short_low 目录
    将视频文件移动到对应文件夹，并统计每类数量，将统计结果写入 JSON 文件保存。此外，打印日志信息。
    返回值：返回筛选后需要进一步处理的“合格”视频列表（既不属于 short/low 的视频）。
    """
    short_videos = []
    low_res_videos = []
    short_low_videos = []
    qualified_videos = []
    # 创建输出分类目录
    short_dir = os.path.join(output_dir, "short")
    low_dir = os.path.join(output_dir, "low")
    short_low_dir = os.path.join(output_dir, "short_low")
    os.makedirs(short_dir, exist_ok=True)
    os.makedirs(low_dir, exist_ok=True)
    os.makedirs(short_low_dir, exist_ok=True)
    # 遍历输入目录的所有文件
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if not os.path.isfile(file_path):
            continue  # 跳过非文件（如文件夹）
        # 打开视频获取元信息
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            logging.warning(f"无法打开视频文件: {filename}")
            continue
        # 获取时长（秒）和分辨率（宽高）
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        # 有些视频元信息可能无法获取，用防御性编程处理
        if fps and frame_count:
            duration = frame_count / fps
        else:
            duration = 0.0
        is_short = duration and duration < 60
        is_lowres = width and height and width < 480 and height < 480
        # 判断分类
        if is_short and is_lowres:
            # 同时短且低分辨率
            short_low_videos.append(filename)
            dest = os.path.join(short_low_dir, filename)
            os.replace(file_path, dest)
            logging.info(f"{filename} -> short_low (时长{duration:.1f}s, 分辨率{int(width)}x{int(height)})")
        elif is_short:
            short_videos.append(filename)
            dest = os.path.join(short_dir, filename)
            os.replace(file_path, dest)
            logging.info(f"{filename} -> short (时长{duration:.1f}s)")
        elif is_lowres:
            low_res_videos.append(filename)
            dest = os.path.join(low_dir, filename)
            os.replace(file_path, dest)
            logging.info(f"{filename} -> low (分辨率{int(width)}x{int(height)})")
        else:
            # 合格视频
            qualified_videos.append(file_path)
    # 输出统计信息到 JSON 文件
    stats = {
        "short": {"count": len(short_videos), "videos": short_videos},
        "low": {"count": len(low_res_videos), "videos": low_res_videos},
        "short_low": {"count": len(short_low_videos), "videos": short_low_videos},
        "qualified": {"count": len(qualified_videos), "videos": [os.path.basename(v) for v in qualified_videos]},
    }
    stats_path = os.path.join(output_dir, "filter_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logging.info(
        f"筛选完成: 短视频{len(short_videos)}个, 低分辨率视频{len(low_res_videos)}个, 短且低视频{len(short_low_videos)}个, 合格视频{len(qualified_videos)}个."
    )
    logging.info(f"筛选统计已保存至 {stats_path}")
    return qualified_videos


def detect_worker(video_path):
    try:
        segments = detect_scenes_and_segments(video_path)
        for seg in segments:
            seg["abs_video_path"] = video_path
        return segments
    except Exception as e:
        return {"error": f"{video_path}: 场景检测失败, {e}"}


def detect_scenes_and_segments(video_path, window=5):
    """
    视频场景检测与分片模块：
    给定一个视频路径，使用 PySceneDetect 进行场景切分，然后按 {window} 秒滑动窗口将场景进一步切片。
    如果检测到多个场景 (>1)，则按场景切割视频并对每个场景片段再按 {window} 秒滑窗进行切片；
    如果只有单一场景，则对整段视频按 {window} 秒滑窗切分；
    场景内部最后如果余下短于 {window} 的视频片段则与之前片段合并。
    返回值：片段列表，其中每个片段包含开始时间start_time（秒）, 结束时间end_time（秒）, 原始视频文件名, 所属场景编号scene_idx。
    """
    video_name = os.path.basename(video_path)
    try:
        scene_list = detect(video_path, ContentDetector(), start_in_scene=True)
    except Exception as e:
        logging.error(f"场景检测失败: {video_name}, 错误信息: {e}")
        return []

    segments = []
    scene_idx = 0
    for scene_start, scene_end in scene_list:
        scene_idx += 1
        start_sec = scene_start.get_seconds()
        end_sec = scene_end.get_seconds()
        if end_sec <= start_sec:
            continue
        # 临时列表存当前场景的所有 segment
        tmp = []
        seg_start = start_sec
        while seg_start < end_sec:
            seg_end = min(seg_start + window, end_sec)
            tmp.append(
                {
                    "video": video_name,
                    "scene_idx": scene_idx,
                    "start_time": round(seg_start, 3),
                    "end_time": round(seg_end, 3),
                }
            )
            seg_start += window
        # 如果最后一个 segment 时长 < window 且至少有两个 segment，则合并
        if len(tmp) >= 2:
            last = tmp[-1]
            duration_last = last["end_time"] - last["start_time"]
            if duration_last < window:
                # 合并倒数第二个与最后一个
                tmp[-2]["end_time"] = last["end_time"]
                tmp.pop(-1)
        segments.extend(tmp)
    tqdm.write(f"{video_name}: 分场景后切为 {len(segments)} 段")
    return segments


def caption_worker(seg):
    video, start, end = seg["abs_video_path"], seg["start_time"], seg["end_time"]
    base = os.path.splitext(os.path.basename(video))[0]
    out_dir = os.path.join("/pfs/training-data/zuhaoyang/data/train/segment_frames", f"{base}_{int(start)}")
    frames = extract_frames_from_segment(video, start, end, out_dir)
    if not frames:
        return None

    caption = generate_caption_for_segment(frames)
    seg["frame_paths"], seg["caption"] = frames, caption
    return seg


def extract_frames_from_segment(video_path, start, end, out_dir, max_frames=5):
    """
    从 video_path 的 [start,end) 中抽取至多 max_frames 帧，均匀抽样保存到 out_dir。
    返回 frame_paths 列表。
    """
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        return [os.path.join(out_dir, fname) for fname in sorted(os.listdir(out_dir))]

    os.makedirs(out_dir, exist_ok=True)
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frames = int((end - start) * fps)
    if total_frames <= 0:
        return []
    sample_count = min(max_frames, total_frames)
    idxs = [int(start * fps) + int(i * total_frames / sample_count) for i in range(sample_count)]
    idxs = [min(i, len(vr) - 1) for i in idxs]
    frames = vr.get_batch(idxs).asnumpy()
    frame_paths = []
    for idx, frame in enumerate(frames):
        fname = os.path.join(out_dir, f"frame_{idx:02d}.jpg")
        cv2.imwrite(fname, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_paths.append(fname)
    return frame_paths


def generate_caption_for_segment(frame_paths):
    """
    使用 Qwen2.5‑VL-72B 的多图 image_url 模式调用 API。
    传入本地 file:// URL + 文本提示，获得一句话的 caption。
    """
    api_base = "https://sd1kauvth3egrfsbiup70.apigateway-cn-shanghai.volceapi.com/v1"
    client = OpenAI(api_key="EMPTY", base_url=api_base)
    urls = [{"type": "image_url", "image_url": {"url": f"file://{p}"}} for p in frame_paths]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {
            "role": "user",
            "content": urls
            + [
                {
                    "type": "text",
                    "text": "Please regard the image sequence as a video segment and describe this segment in one sentence.",
                }
            ],
        },
    ]  # 建议：1. 1 fs 太稀疏了，可能需要抽 ＞ 5帧，还是会损失很多信息；2. 一句话的信息还是太少了，可以考虑写一个negative prompt，重点输出object event相关
    try:
        chat = client.chat.completions.create(model="Qwen/Qwen2.5-VL-72B-Instruct", messages=messages)
        return chat.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"调用 Qwen2.5‑VL 接口失败: {e}")
        return ""


def flag_segments_pairwise(segments):
    """
    对同一个视频内的相邻 segment 进行两两比较，判断是否为同一个事件。
    segments 已按 abs_video_path, start_time 排序。
    返回新列表, 每个 segment 增加字段:
      - merged: bool（是否和下一个segment属于同一事件）
      - event_idx: int（同一个事件内统一编号，从0开始）
    """
    api_base = "https://sd1kauvth3egrfsbiup70.apigateway-cn-shanghai.volceapi.com/v1"
    client = OpenAI(api_key="EMPTY", base_url=api_base)
    result = []
    current_event = 1
    system_prompt = (
        "You are a helpful assistant for judging if two *consecutive* video segments describe the **same event**."
    )
    for i, seg in enumerate(tqdm(segments, total=len(segments))):
        merged_flag = False
        if i < len(segments) - 1 and seg["abs_video_path"] == segments[i + 1]["abs_video_path"]:
            user_query = f"The caption of the first segment: {seg['caption']}\n \
            The caption of the second segment: {segments[i+1]['caption']}\n \
            Please answer 'yes' or 'no'."
            messages = [
                {"role": "system", "content": [{"type": "text", "text": f"{system_prompt}"}]},
                {"role": "user", "content": [{"type": "text", "text": f"{user_query}"}]},
            ]
            try:
                chat = client.chat.completions.create(model="Qwen/Qwen2.5-VL-72B-Instruct", messages=messages)
                ans = chat.choices[0].message.content.strip().lower()
                merged_flag = ans.startswith("yes")
            except Exception as e:
                logging.error(f"pairwise API error at segments {i} & {i+1}: {e}")
        seg_event_idx = current_event
        result.append({**seg, "merged": merged_flag, "event_idx": seg_event_idx})
        if not merged_flag:
            current_event += 1
    return result


def merge_captions_pairwise(flagged_segments):
    """
    对已打标的 segments（有 merged + event_idx）按照 event_idx 分组，
    为需要合并的事件调用 Qwen 生成 summary caption 且合并时间戳。
    """
    api_base = "https://sd1kauvth3egrfsbiup70.apigateway-cn-shanghai.volceapi.com/v1"
    client = OpenAI(api_key="EMPTY", base_url=api_base)
    events = {}
    for seg in flagged_segments:
        key = (seg["abs_video_path"], seg["event_idx"])
        events.setdefault(key, []).append(seg)
    merged_events = []
    system_prompt = "You are a helpful assistant for summarizing short video events.\n \
                Given a list of segment captions and their timestamps,\n \
                you are expected to produce one concise sentence that accurately describes the entire event.\n \
                Please focus on combining key actions, participants, and context.\n \
                Please avoid repeating details and ensure the summary covers the full time span.\n \
                Example Input:\n \
                ['- [0s-2s] A person opens a door.', '- [2s-4s] A person steps inside the room.', ...]\n \
                Expected output: 'A person opens a door and then steps into the room.'"
    for (video_path, ev_idx), segs in tqdm(events.items()):
        segs = sorted(segs, key=lambda s: s["start_time"])
        start_time = segs[0]["start_time"]
        end_time = segs[-1]["end_time"]
        if len(segs) == 1:
            summary = segs[0]["caption"]
        else:
            caption_list = [f"- [{int(s['start_time'])}s–{int(s['end_time'])}s] {s['caption']}" for s in segs]
            user_query = (
                "Here are captions of the segments belonging to one event:\n"
                f"{caption_list}\n"
                "Please summarize this event in one concise sentence."
            )
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_query}]},
            ]
            try:
                chat = client.chat.completions.create(model="Qwen/Qwen2.5-VL-72B-Instruct", messages=messages)
                summary = chat.choices[0].message.content.strip()
            except Exception as e:
                logging.error(f"Global summary API error for event {video_path}, idx {ev_idx}: {e}")
                summary = "."
        merged_events.append(
            {
                "video": segs[0]["video"],
                "abs_video_path": video_path,
                "event_idx": ev_idx,
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
                "caption": summary,
            }
        )
    return merged_events


def merge_segments_global(segments):
    """
    事件聚合模块 - 方法二（全局聚类）：
    将所有片段的caption一次性提供给LLM，让其基于语义对片段进行全局聚类、分组和合并为事件。
    实际实现中，会调用大型模型对所有caption进行聚类和摘要，这里用简化逻辑模拟。
    返回值：全局聚类/合并后的事件列表，结构同上。
    """
    if not segments:
        return []
    events = []
    # 简化实现：不进行任何合并，直接将每个片段作为单独事件
    for seg in segments:
        caption = seg.get("caption", "")
        events.append(
            {
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "caption": caption,
                "merged": False,
                "sub_captions": [caption] if caption else [],
            }
        )
    # 实际情况下，这里会使用LLM将相关片段合并。
    return events


def choose_best_events(events1, events2):
    """
    使用LLM判断两个聚合结果哪个更合理：
    将方法一和方法二得到的事件列表提供给LLM，请其判断哪种划分更合理。
    这里简化地以事件数量为依据：事件数量较少且不为空则认为更合理（假设更少的事件表示更好的聚合）。
    返回值：挑选出的较优事件列表。
    """
    if not events1:
        return events2
    if not events2:
        return events1
    # 简单规则：默认选择事件数量较少的方案
    best = events1 if len(events1) <= len(events2) else events2
    # TODO: 实际实现中应调用LLM，根据语义连贯性和事件完整性来评估选择
    return best


def refine_event_timestamps(events, video_path, use_llm=False):
    """
    时间戳精细化模块：
    尝试将每段事件的起止时间各扩展1秒，提取新的clip，判断原caption是否仍成立，以此微调事件边界。
    方法：
      - 方法一：使用embedding计算扩展后clip与原clip内容的相似度，如果相似度高则认为caption仍适用。
      - 方法二：使用LLM判断扩展片段的内容是否仍符合原caption描述。
    参数 use_llm 控制使用LLM判断（True）还是embedding相似度（False）。
    返回值：调整时间戳后的事件列表。
    """
    refined_events = []
    # 预获取视频总时长
    total_duration = 0
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and frame_count:
            total_duration = frame_count / fps
        cap.release()
    for i, ev in enumerate(events):
        start = ev["start_time"]
        end = ev["end_time"]
        # 计算扩展后的时间范围（不超出视频范围，不跨场景）
        # 获取该事件所属场景范围，用场景检测结果或假定事件未跨场景
        scene_start = start
        scene_end = end
        # 扩展1秒
        ext_start = max(scene_start, start - 1.0)
        ext_end = min(scene_end if scene_end else total_duration, end + 1.0)
        # 防止与相邻事件重叠
        if i > 0:
            prev_end = events[i - 1]["end_time"]
            if ext_start < prev_end:
                ext_start = start  # 不能向前扩展
        if i < len(events) - 1:
            next_start = events[i + 1]["start_time"]
            if ext_end > next_start:
                ext_end = end  # 不能向后扩展
        # 如果扩展后与原边界没有变化，则跳过
        if ext_start == start and ext_end == end:
            refined_events.append(ev)
            continue
        # 判断caption是否仍成立
        caption = ev["caption"]
        if use_llm:
            # 方法二：通过LLM判断
            # 这里简化为重新生成扩展clip的caption并比较文本
            new_caption = generate_caption_for_segment(video_path, ext_start, ext_end)
            # 简单比较：如果新caption包含原caption的关键字，则认为仍成立
            valid = False
            if caption and new_caption:
                common = set(caption.split()) & set(new_caption.split())
                if common:
                    valid = True
        else:
            # 方法一：通过embedding相似度判断（这里不实际计算，只用占位逻辑）
            valid = True  # 假设内容变化不大，则caption仍适用
        if valid:
            # 调整事件时间戳为扩展后的范围
            ev["start_time"] = round(ext_start, 3)
            ev["end_time"] = round(ext_end, 3)
        refined_events.append(ev)
    return refined_events


def save_events_to_json(video_path, events, output_dir):
    """
    输出模块：
    将给定视频的事件列表保存为JSON文件。JSON结构包括视频ID和事件数组，每个事件包含start_time, end_time, caption, merged标记, sub_captions列表。
    """
    video_name = os.path.basename(video_path)
    video_id = os.path.splitext(video_name)[0]
    output_data = {"video_id": video_id, "events": []}
    for ev in events:
        output_data["events"].append(
            {
                "start_time": round(ev["start_time"], 3),
                "end_time": round(ev["end_time"], 3),
                "caption": ev.get("caption", ""),
                "merged": ev.get("merged", False),
                "sub_captions": ev.get("sub_captions", []),
            }
        )
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{video_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logging.info(f"视频 {video_name} 的事件结果已保存: {json_path}")


def main():
    """
    主函数：解析命令行参数，执行整个处理流程。
    """
    parser = argparse.ArgumentParser(description="data pipeline for dense video captioning")
    parser.add_argument("--input_dir", type=str, required=True, help="输入视频目录路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出结果目录路径")
    parser.add_argument(
        "--use_llm_refine", action="store_true", help="是否使用LLM进行时间戳微调（默认使用embedding方法）"
    )
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    use_llm = args.use_llm_refine
    # 筛选视频并分类
    qualified_videos = filter_videos(input_dir, output_dir)
    # 逐个处理合格视频
    for video_path in qualified_videos:
        video_name = os.path.basename(video_path)
        logging.info(f"开始处理视频: {video_name}")
        # 检测场景并切片
        segments = detect_scenes_and_segments(video_path)
        if not segments:
            logging.info(f"视频 {video_name} 无法切分片段，跳过后续处理")
            continue
        # 为每个片段生成描述caption
        for seg in segments:
            cap_text = generate_caption_for_segment(video_path, seg["start_time"], seg["end_time"])
            seg["caption"] = cap_text
        # 方法一：相邻合并 (Iterative)
        events1 = merge_segments_pairwise(segments)
        # 方法二：全局聚合（LLM）
        events2 = merge_segments_global(segments)
        # 选择更优的事件划分
        final_events = choose_best_events(events1, events2)
        # 时间戳精细化
        refined_events = refine_event_timestamps(final_events, video_path, use_llm=use_llm)
        # 保存结果
        save_events_to_json(video_path, refined_events, output_dir)
        logging.info(f"完成视频 {video_name} 的处理\n")


if __name__ == "__main__":
    ## Video Directories
    all_data_dirs = [
        "/pfs/training-data/zuhaoyang/data/train/activitynet",
        "/pfs/training-data/zuhaoyang/data/train/didemo",
        "/pfs/training-data/zuhaoyang/data/train/ego4d_naq",
        "/pfs/training-data/zuhaoyang/data/train/hacs",
        "/pfs/training-data/zuhaoyang/data/train/queryd",
        "/pfs/training-data/zuhaoyang/data/train/tacos",
        "/pfs/training-data/zuhaoyang/data/train/TimeR1-Dataset/TimeRFT_data/timerft_data",
        "/pfs/training-data/zuhaoyang/data/eval/charades",
    ]
    video_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")

    ## Generate Caption for Segment
    with open("./segments.json") as f:
        segments = json.load(f)
    enriched = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as exe:
        futures = {exe.submit(caption_worker, seg): seg for seg in segments}
        for f in tqdm(as_completed(futures), total=len(futures), desc="CaptionGen"):
            res = f.result()
            if res:
                enriched.append(res)
    with open("./segments_with_captions.json", "w") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
