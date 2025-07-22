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

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClientDemo:
    def __init__(self, server_path: str):
        """
        初始化MCP客户端
        :param server_path: MCP服务端脚本路径
        """
        self.server_path = server_path
        # 创建OpenAI客户端，连接到兼容API的阿里云DashScope服务
        self.llm = None

    async def run(self):
        """
        执行用户查询，对比使用工具和不使用工具的结果
        :param user_query: 用户问题
        :return: 对比结果字典
        """
        # 配置标准IO通信的服务端参数
        server_params = StdioServerParameters(command="python", args=[self.server_path])
        # 建立与MCP服务端的连接
        async with stdio_client(server=server_params) as (read_stream, write_stream):
            # 创建客户端会话
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # 获取服务端注册的所有工具信息
                tools = (await session.list_tools()).tools

                # 将MCP工具格式转换为OpenAI函数调用格式
                functions = []
                for tool in tools:
                    functions.append(
                        {
                            "name": tool.name,
                            "description": tool.description or "",
                            # 使用工具的输入模式或默认模式
                            "parameters": tool.inputSchema
                            or {
                                "type": "object",
                                "properties": {"city_name": {"type": "string", "description": "城市名称"}},
                                "required": ["city_name"],
                            },
                        }
                    )
                    print(functions[-1])

                return functions

    async def run_tool(self, tool_name: str, tool_args: dict):
        """
        Run a specific tool with the given arguments.
        :param tool_name: Name of the tool to run.
        :param tool_args: Arguments for the tool.
        :return: Result of the tool execution.
        """
        server_params = StdioServerParameters(command="python", args=[self.server_path])
        async with stdio_client(server=server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                result = await session.call_tool(tool_name, tool_args)
                return result


async def main():
    """主函数，演示工具使用与不使用的对比"""
    # 创建MCP客户端，连接到指定服务端
    client = MCPClientDemo(server_path="examples/video_tools/mcp_server.py")
    # 执行天气查询示例
    result = await client.run()
    result = await client.run_tool(
        "crop_video", {"video_path": "/path/to/video.mp4", "start_time": 10.0, "end_time": 20.0}
    )
    return result

    # 格式化输出对比结果


if __name__ == "__main__":
    # 运行异步主函数
    result = asyncio.run(main())
