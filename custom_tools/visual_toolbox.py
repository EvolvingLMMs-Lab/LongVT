# Copyright 2023-2025 SGLang Team
# Copyright Amazon.com, Inc. or its affiliates.
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import logging
import os
from math import ceil, floor
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class VisualToolBox(BaseTool):
    """A demo tool for calculating the reward of geo3k.
    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            type: "function"
            function:
                name: "image_zoom_in_tool"
                description: "Zoom in on a specific region of an image by cropping
                it based on a bounding box (bbox) and an optional object label."
                parameters:
                type: object
                properties:
                    bbox_2d:
                    type: array
                    items:
                        type: number
                    minItems: 4
                    maxItems: 4
                    description: "The bounding box of the region to zoom in, as [x1, y1, x2, y2],
                    where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."
                    label:
                    type: string
                    description: "The name or label of the object in the specified bounding box (optional)."
                required:
                    - bbox_2d
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "images": kwargs["images"]["image"],
            "tool_call_count": 0,
        }
        return instance_id, None

    def validate_bbox(self, left, top, right, bottom):
        try:
            assert left < right and bottom > top, f"invalid shape for {left=}, {top=}, {right=}, {bottom=}"
            height = bottom - top
            width = right - left
            assert max(height, width) / min(height, width) <= 100, (
                f"aspect ratio error: {left=}, {top=}, {right=}, {bottom=}"
            )
            assert min(height, width) > 30, f"{height=}, {width=} is too small"
            return True
        except Exception as err:
            print(f" [ERROR vl_agent #2] {err=}")
            return False

    def maybe_resize_bbox(self, left, top, right, bottom):
        left = max(0, left)
        top = max(0, top)
        right = min(self.width, right)
        bottom = min(self.height, bottom)
        if not self.validate_bbox(left, top, right, bottom):
            return None

        height = bottom - top
        width = right - left
        if height < 28 or width < 28:
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            ratio = 28 / min(height, width)
            new_half_height = ceil(height * ratio * 0.5)
            new_half_width = ceil(width * ratio * 0.5)
            new_left = floor(center_x - new_half_width)
            new_right = ceil(center_x + new_half_width)
            new_top = floor(center_y - new_half_height)
            new_bottom = ceil(center_y + new_half_height)
            if not self.validate_bbox(new_left, new_top, new_right, new_bottom):
                return None
            return [new_left, new_top, new_right, new_bottom]
        return [left, top, right, bottom]

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        self._instance_dict[instance_id]["tool_call_count"] += 1
        from verl.utils.dataset.vision_utils import process_image

        # img1 = kwargs['images']
        img1 = self._instance_dict[instance_id]["images"]
        img = process_image(img1[0]) if isinstance(img1, list) else process_image(img1)

        self.width, self.height = img.size
        # args = parameters.get("arguments", "")

        try:
            # Zoom in by cropping the image
            bbox = parameters["bbox_2d"]
            bbox = self.maybe_resize_bbox(*bbox)
            if not bbox:
                raise ValueError("ZOOM IN ARGUMENTS ARE INVALID")
            # img = Image.open(image_path)

            cropped_img = img.crop(bbox)
            # cropped_img = cropped_img.resize((28, 28), Image.BICUBIC)
            current_image = cropped_img

            return {"image": [current_image]}, 0, {}

        except Exception as e:
            return {"text": e}, 0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return self._instance_dict[instance_id]["tool_call_count"]

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
