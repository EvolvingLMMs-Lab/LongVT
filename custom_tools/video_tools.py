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
import json
import logging
import os
from io import BytesIO

from PIL import Image

from verl.tools.mcp_base_tool import MCPBaseTool

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class VideoTools(MCPBaseTool):
    async def execute(self, instance_id, parameters, **kwargs):
        try:
            result_text, metadata = await self._call_tool(instance_id, parameters)
            image_list = metadata["images"]
            from verl.utils.dataset.vision_utils import process_image

            images = [process_image(image) for image in image_list]
            text = "The tool executed successfully. Here are the processed result"

            return {"image": images, "text": text}, 0.0, {}

        except Exception as e:
            error_result = json.dumps({"result": f"Tool execution failed: {e}"})
            logger.error(f"[MCPBaseTool] Execution failed: {e}")
            return error_result, 0.0, {"error": str(e)}

    def _parse_tool_result(self, content):
        image_contents = [part.data for part in filter(lambda x: x.type == "image", content)]
        # Convert base64 string to PIL image
        image_lists = []
        for image_content in image_contents:
            im = Image.open(BytesIO(base64.b64decode(image_content)))
            image_lists.append(im)

        return "", {"images": image_lists, "api_request_error": ""}
