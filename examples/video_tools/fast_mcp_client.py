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

import asyncio

from fastmcp import Client

from verl.tools.utils.mcp_clients.utils import mcp2openai


class MCPClientDemo:
    def __init__(self, server_path: str):
        """
        初始化MCP客户端
        :param server_path: MCP服务端脚本路径
        """
        self.server_path = server_path
        self.client = None

    async def initialize(self):
        """
        初始化客户端连接
        """
        if self.client is None:
            # 配置本地stdio服务端
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
        获取服务端的所有工具信息并转换为OpenAI格式
        :return: OpenAI格式的工具列表
        """
        await self.initialize()
        
        tool_schemas = []
        async with self.client:
            # 获取服务端注册的所有工具信息
            tools_response = await self.client.list_tools_mcp()
            
            # 将MCP工具格式转换为OpenAI函数调用格式
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


async def main():
    """主函数，演示工具使用与不使用的对比"""
    # 创建MCP客户端，连接到指定服务端
    client = MCPClientDemo(server_path="examples/video_tools/mcp_server.py")
    # 执行天气查询示例
    result = await client.run()
    result = await client.run_tool("crop_video", {"video_path": "../test.mp4", "start_time": 10.0, "end_time": 20.0})
    return result


if __name__ == "__main__":
    # 运行异步主函数
    result = asyncio.run(main())
