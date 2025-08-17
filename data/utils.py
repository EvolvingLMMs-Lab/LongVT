import base64

from decord import VideoReader
from PIL import Image
from qwen_vl_utils import fetch_video
from scenedetect import ContentDetector, detect

MAX_PIXELS = 360 * 420


# Encode a PIL Image to base64
def encode_image(image: Image.Image) -> str:
    return base64.b64encode(image.tobytes()).decode("utf-8")


def get_video_length(video_path: str) -> float:
    try:
        vr = VideoReader(video_path)
        return len(vr) / vr.get_avg_fps()
    except Exception as e:
        print(f"Error getting video length for {video_path}: {e}")
        return -1


def process_video(video_path: str, fps: int):
    video_dict = {
        "type": "video",
        "video": f"file://{video_path}",
        "fps": fps,
        "max_pixels": MAX_PIXELS,
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
