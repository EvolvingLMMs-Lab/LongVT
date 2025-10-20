import base64
from io import BytesIO

from decord import VideoReader
from PIL import Image
from qwen_vl_utils import fetch_video
from scenedetect import ContentDetector, detect

MAX_PIXELS = 360 * 420


def encode_image(
    image: Image,
):
    output_buffer = BytesIO()
    image.save(output_buffer, format="PNG")
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return base64_str


def get_video_length(video_path: str) -> float:
    try:
        vr = VideoReader(video_path)
        return len(vr) / vr.get_avg_fps()
    except Exception as e:
        print(f"Error getting video length for {video_path}: {e}")
        return -1


def process_video(video_path: str, fps: int, start_time: float = 0, end_time: float = None):
    video_dict = {
        "type": "video",
        "video": f"file://{video_path}",
        "fps": fps,
        "max_pixels": MAX_PIXELS,
        "video_start": start_time,
        "video_end": end_time,
    }
    return fetch_video(video_dict)


def detect_scenes(video_path: str, start_time: float = 0, end_time: float = None):
    scenes = detect(
        video_path,
        ContentDetector(),
        start_in_scene=True,
        show_progress=False,
        start_time=start_time,
        end_time=end_time,
    )
    return scenes
