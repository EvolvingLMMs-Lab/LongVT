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
import logging
import os
from io import BytesIO
from typing import Optional, Tuple
from uuid import uuid4

from PIL import Image

from verl.tools.mcp_base_tool import MCPBaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

from fastmcp import Client

from verl.tools.utils.mcp_clients.utils import mcp2openai


# I don't know why, but official mcp client is not working and kept hanging.
# Create a client to connect to the mcp server instead directly
class MCPClientDemo:
    def __init__(self, server_path: str):
        """
        Initialize MCP client
        :param server_path: Path to MCP server script
        """
        self.server_path = server_path
        self.client = None

    async def initialize(self):
        """
        Initialize client connection
        """
        if self.client is None:
            # Configure local stdio server
            config = {
                "mcpServers": {
                    "local_server": {
                        "command": "python",
                        "args": [self.server_path]
                    }
                }
            }
            self.client = Client(config)

    async def run(self):
        """
        Get all tool information from the server and convert to OpenAI format
        :return: Tool list in OpenAI format
        """
        await self.initialize()
        
        tool_schemas = []
        async with self.client:
            # Get all tool information registered by the server
            tools_response = await self.client.list_tools_mcp()
            
            # Convert MCP tool format to OpenAI function call format
            for tool in tools_response.tools:
                openai_tool = mcp2openai(tool)
                tool_schemas.append(openai_tool)
                print(openai_tool)
        
        return tool_schemas

    async def run_tool(self, tool_name: str, tool_args: dict):
        """
        Run a specific tool with the given arguments.
        :param tool_name: Name of the tool to run.
        :param tool_args: Arguments for the tool.
        :return: Result of the tool execution.
        """
        await self.initialize()
        
        async with self.client:
            result = await self.client.call_tool_mcp(tool_name, tool_args)
            return result


class VideoTools(MCPBaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        server_path = config.get("server_path")
        self.client = MCPClientDemo(server_path)
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> Tuple[str, ToolResponse]:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"tool_call_count": 0}
        return instance_id, ToolResponse()

    async def _call_tool(self, instance_id, parameters):
        err_msg = None
        try:
            result = await self.client.run_tool(self.name, parameters)
        except Exception as e:
            err_msg = f"Tool execution failed: {e}"
            logger.error(f"[VideoTools] Execution failed: {e}")
        result, metadata = self._parse_tool_result(result.content)
        metadata["api_request_error"] = "" if not err_msg else err_msg
        return result, metadata

    async def execute(self, instance_id, parameters, **kwargs):
        try:
            self._instance_dict[instance_id]["tool_call_count"] += 1  # tool call count
            result_text, metadata = await self._call_tool(instance_id, parameters)

            # Check for API request errors from MCP call
            api_error = metadata.get("api_request_error", "").strip()
            if api_error:
                error_msg = f"Tool execution failed: {api_error}"
                logger.error(f"[VideoTools] MCP call failed: {api_error}")
                return ToolResponse(text=error_msg), 0.0, {"error": api_error}

            image_list = metadata["images"]
            from verl.utils.dataset.vision_utils import process_image

            images = [process_image(image) for image in image_list]

            # Generate dynamic success message with frame count
            success_msg = "The tool executed successfully. Here are the processed result"
            return ToolResponse(image=images, text=success_msg), 0.0, {"success": True}

        except Exception as e:
            error_msg = f"Tool execution failed: {e}"
            logger.error(f"[VideoTools] Execution failed: {e}")
            return ToolResponse(text=error_msg), 0.0, {"error": str(e)}

    # tool call count
    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["tool_call_count"]

    def _parse_tool_result(self, content):
        # Check for text content that might contain error messages
        text_parts = [part.text for part in filter(lambda x: x.type == "text", content)]

        # Look for error indicators in text content
        api_error = ""
        for text in text_parts:
            error_keywords = ["error", "failed", "exception", "validation error"]
            if any(error_keyword in text.lower() for error_keyword in error_keywords):
                api_error = text
                logger.error(f"[VideoTools] MCP response contains error: {api_error}")
                return "", {"images": [], "api_request_error": api_error}

        # Parse image content
        image_contents = [part.data for part in filter(lambda x: x.type == "image", content)]

        # Convert base64 string to PIL image
        image_lists = []
        for image_content in image_contents:
            im = Image.open(BytesIO(base64.b64decode(image_content)))
            image_lists.append(im)

        return "", {"images": image_lists, "api_request_error": ""}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
