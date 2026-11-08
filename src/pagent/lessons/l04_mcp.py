"""第 4 课：MCP 客户端 —— 连接 MCP Server，发现并调用它的 Tools / Resources / Prompts。

这里直接使用 FastMCP 的 Client（LangChain 的 MCPAdapter 底层就是它），
看清楚协议层面发生了什么：
  1. 启动 / 连接 Server（stdio 子进程 或 HTTP）
  2. tools/list      -> 拿到工具名 + 描述 + JSON Schema
  3. tools/call      -> 调用工具，拿到 content（给模型看的文本）+ structuredContent（结构化数据）
  4. resources/read  -> 读取只读资源
  5. prompts/get     -> 获取提示词模板
最后演示 MCPAdapter 如何把它们变成 LangChain Tool，交给 Agent 使用。

运行：uv run python -m pagent.lessons.l04_mcp
"""

from __future__ import annotations

import asyncio

from fastmcp import Client

from pagent.agent import default_mcp_config


async def main() -> None:
    config = default_mcp_config()
    print("MCP 配置：", config, "\n")

    async with Client(config) as client:
        tools = await client.list_tools()
        print("== tools/list ==")
        for t in tools:
            print(f"- {t.name}: {t.description}\n    schema={t.input_schema}")

        add = next(t.name for t in tools if t.name.endswith("add_todo"))
        lst = next(t.name for t in tools if t.name.endswith("list_todos"))
        print("\n== tools/call ==")
        r = await client.call_tool(add, {"title": "学习 MCP 协议", "due": "2026-12-01"})
        print("content:", r.content[0].text if r.content else None)
        print("structured:", r.structured_content)
        r = await client.call_tool(lst, {})
        print("待办列表:", r.structured_content)

        print("\n== resources/list & read ==")
        for res in await client.list_resources():
            data = await client.read_resource(res.uri)
            print(f"- {res.uri}: {data[0].text}")

        print("\n== prompts/list & get ==")
        for p in await client.list_prompts():
            got = await client.get_prompt(p.name)
            print(f"- {p.name}: {got.messages[0].content.text}")

    # ---- 交给 LangChain：一行把 MCP 工具变成 Agent 能用的 Tool ----
    from langchain.mcp import MCPAdapter

    async with MCPAdapter(config) as adapter:
        lc_tools = await adapter.list_tools()
        print("\n== 转换成 LangChain Tool ==")
        for t in lc_tools:
            print(f"- {t.name} ({type(t).__name__})")


if __name__ == "__main__":
    asyncio.run(main())
