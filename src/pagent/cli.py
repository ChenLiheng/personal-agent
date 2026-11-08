"""交互式命令行：流式输出 + 人工审批 + 多会话。

命令：
  /new          开一个新会话（新 thread_id，短期记忆清空，长期记忆还在）
  /thread <id>  切换到某个历史会话（从 SQLite checkpoint 恢复）
  /user <id>    切换用户（长期记忆按用户隔离）
  /memories     查看当前用户的长期记忆
  /reindex      重建知识库向量索引（改了 data/knowledge 之后执行）
  /exit         退出
"""

from __future__ import annotations

import asyncio
import json
import uuid

from langchain_core.messages import AIMessageChunk
from langgraph.types import Command
from rich.console import Console
from rich.prompt import Prompt

from pagent.memory import UserContext, memory_namespace

console = Console()


async def stream_turn(agent, inp, config, ctx) -> None:
    """跑一轮，边生成边打印 token。stream_mode=messages 拿 token，updates 拿每步结果。"""
    printed = False
    async for chunk in agent.astream(
        inp, config, context=ctx, stream_mode=["messages", "updates"], version="v2"
    ):
        if chunk["type"] != "messages" or chunk["ns"]:  # ns 非空 = 来自子图，这里不展示
            continue
        token, meta = chunk["data"]
        if isinstance(token, AIMessageChunk) and meta.get("langgraph_node") == "model" and token.text:
            if not printed:
                console.print("[bold cyan]PAgent:[/bold cyan] ", end="")
                printed = True
            console.print(token.text, end="", markup=False, highlight=False)
    if printed:
        console.print()


def ask_decisions(interrupt_value: dict) -> list[dict]:
    """把 HITL 中断里的每个待审批动作展示给用户，收集 approve / edit / reject。"""
    decisions = []
    for action in interrupt_value["action_requests"]:
        console.rule("[yellow]需要你审批的工具调用[/yellow]")
        console.print(f"工具: [bold]{action['name']}[/bold]")
        console.print_json(json.dumps(action["args"], ensure_ascii=False))
        choice = Prompt.ask("批准(a) / 修改参数(e) / 拒绝(r)", choices=["a", "e", "r"], default="a")
        if choice == "a":
            decisions.append({"type": "approve"})
        elif choice == "e":
            raw = Prompt.ask("输入新的参数 JSON")
            decisions.append(
                {"type": "edit", "edited_action": {"name": action["name"], "args": json.loads(raw)}}
            )
        else:
            reason = Prompt.ask("拒绝理由（会告诉模型）", default="用户拒绝了这次操作")
            decisions.append({"type": "reject", "message": reason})
    return decisions


async def chat() -> None:
    from pagent.agent import open_agent
    from pagent.rag import build_index

    console.print("[bold]PAgent[/bold] 启动中（加载向量模型、连接 MCP 服务器）…")
    async with open_agent() as agent:
        ctx = UserContext(user_id="default")
        thread_id = uuid.uuid4().hex[:8]
        console.print(f"就绪。会话 [green]{thread_id}[/green]，用户 [green]{ctx.user_id}[/green]。输入 /exit 退出。\n")

        while True:
            try:
                text = Prompt.ask("[bold green]你[/bold green]").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text == "/exit":
                break
            if text == "/new":
                thread_id = uuid.uuid4().hex[:8]
                console.print(f"新会话 {thread_id}")
                continue
            if text.startswith("/thread "):
                thread_id = text.split(maxsplit=1)[1]
                console.print(f"切换到会话 {thread_id}")
                continue
            if text.startswith("/user "):
                ctx = UserContext(user_id=text.split(maxsplit=1)[1])
                console.print(f"切换到用户 {ctx.user_id}")
                continue
            if text == "/memories":
                items = await agent.store.asearch(memory_namespace(ctx.user_id), limit=50)
                for it in items:
                    console.print(f"- ({it.key}) {it.value['text']}")
                if not items:
                    console.print("(暂无)")
                continue
            if text == "/reindex":
                console.print(f"已重建索引：{build_index()} 个文档块")
                continue

            config = {"configurable": {"thread_id": thread_id}}
            inp = {"messages": [{"role": "user", "content": text}]}
            try:
                await stream_turn(agent, inp, config, ctx)
                # 处理 Human-in-the-loop：图停在中断处，等待我们 resume
                while (state := await agent.aget_state(config)).interrupts:
                    decisions = ask_decisions(state.interrupts[0].value)
                    await stream_turn(agent, Command(resume={"decisions": decisions}), config, ctx)
            except Exception as e:
                console.print(f"[red]出错了: {type(e).__name__}: {e}[/red]")


def main() -> None:
    asyncio.run(chat())


if __name__ == "__main__":
    main()
