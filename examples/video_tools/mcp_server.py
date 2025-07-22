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

import base64
from io import BytesIO

import numpy as np
from decord import VideoReader, cpu
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent
from PIL import Image

app = FastMCP("Video Tools MCP Server", "0.1.0")


@app.tool(name="crop_video", description="Crop a video to a specified duration.")
def crop_video(video_path: str, start_time: float, end_time: float) -> list[ImageContent]:
    """
    Crop a video to a specified duration.

    Args:
        video_path (str): Path to the video file.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.

    Returns:
        str: Path to the cropped video file.
    """
    # Placeholder for actual video cropping logic
    nframes = 16
    vr = VideoReader(video_path, ctx=cpu(0))
    start_frame = int(start_time * vr.get_avg_fps())
    end_frame = int(end_time * vr.get_avg_fps())
    uniform_sampled_frames = np.linspace(start_frame, end_frame - 1, nframes, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # Turn np into base64 encoded
    images = [Image.fromarray(frame) for frame in spare_frames]
    # Encode images to base64
    image_contents = []
    for img in images:
        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        image_contents.append(ImageContent(type="image", data=base64_str, mimeType="image/png"))

    return image_contents


if __name__ == "__main__":
    app.run()
