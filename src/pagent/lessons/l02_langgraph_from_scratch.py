"""第 2 课：用 LangGraph 手搭一个 Agent 图（理解 create_agent 内部在做什么）。

LangGraph 的四个核心概念：
  - State   ：在节点之间流动的数据（这里是 messages 列表）。reducer 决定如何合并更新，
              `add_messages` 表示“追加”而不是“覆盖”。
  - Node    ：一个函数，输入 State，返回对 State 的部分更新。
  - Edge    ：固定边（A 之后一定到 B）或条件边（根据 State 决定下一步去哪）。
  - Checkpointer：每一步之后保存 State 快照 -> 多轮对话记忆、暂停/恢复、时间旅行。

本课的图：

    START -> agent --(有 tool_calls?)--是--> approval --> tools --> agent ...
                    \\--否--> END

  approval 节点里用 `interrupt()` 实现 Human-in-the-loop：
  图执行到这里会暂停并保存状态，等人类用 Command(resume=...) 回复后从断点继续。

运行：uv run python -m pagent.lessons.l02_langgraph_from_scratch
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from pagent.tools import calculator, get_current_time, write_file

TOOLS = [calculator, get_current_time, write_file]
SENSITIVE = {"write_file"}  # 需要人工审批的工具


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_graph(model, checkpointer=None):
    model_with_tools = model.bind_tools(TOOLS)  # 把工具的 JSON Schema 交给模型

    def agent_node(state: State) -> dict:
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def route_after_agent(state: State) -> Literal["approval", "__end__"]:
        last = state["messages"][-1]
        return "approval" if isinstance(last, AIMessage) and last.tool_calls else END

    def approval_node(state: State) -> Command[Literal["tools", "agent"]]:
        last: AIMessage = state["messages"][-1]
        sensitive = [c for c in last.tool_calls if c["name"] in SENSITIVE]
        if not sensitive:
            return Command(goto="tools")
        # 暂停！interrupt 的参数会被返回给调用方，调用方 resume 的值成为这里的返回值
        answer = interrupt({"question": "是否允许执行以下工具调用？", "calls": sensitive})
        if answer == "yes":
            return Command(goto="tools")
        # 拒绝：为每个 tool_call 伪造一个“被拒绝”的工具结果，再回到 agent
        rejected = [
            ToolMessage(content="用户拒绝执行该操作", tool_call_id=c["id"]) for c in last.tool_calls
        ]
        return Command(goto="agent", update={"messages": rejected})

    g = StateGraph(State)
    g.add_node("agent", agent_node)
    g.add_node("approval", approval_node)
    g.add_node("tools", ToolNode(TOOLS))  # 预置节点：执行 tool_calls，返回 ToolMessage
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent)
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer or InMemorySaver())


if __name__ == "__main__":
    from pagent.config import get_chat_model

    graph = build_graph(get_chat_model())
    print("图结构（Mermaid，可粘贴到 https://mermaid.live 查看）：")
    print(graph.get_graph().draw_mermaid())

    config = {"configurable": {"thread_id": "lesson-2"}}

    def run(inp):
        for update in graph.stream(inp, config, stream_mode="updates"):
            for node, data in update.items():
                if node == "__interrupt__":
                    print(f"⏸  中断: {data[0].value}")
                    continue
                for m in data.get("messages", []):
                    print(f"[{node}] {type(m).__name__}: {m.text or m.tool_calls}")

    run({"messages": [{"role": "user", "content": "算一下 (17*23+5)/3，然后把结果写进 result.txt"}]})
    if graph.get_state(config).interrupts:
        ans = input("批准写文件吗？(yes/no) ").strip()
        run(Command(resume=ans))

    # 同一个 thread_id，模型记得上文 —— 这就是 checkpointer 提供的短期记忆
    run({"messages": [{"role": "user", "content": "刚才算出来的结果是多少？"}]})
    print(f"\n本会话共 {len(graph.get_state(config).values['messages'])} 条消息被保存在 checkpoint 中")
