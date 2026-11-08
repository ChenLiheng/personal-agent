"""第 6 课：多 Agent —— Router（路由）模式。

常见多 Agent 架构：
  - Agents-as-tools / Supervisor：主 Agent 把子 Agent 当工具调用（PAgent 的 ask_researcher）
  - Router：先用一个轻量的分类步骤决定交给哪个专家，专家各自有独立的提示词和工具   <- 本课
  - Handoff / Swarm：Agent 之间互相“转交”控制权（客服 -> 退款专员）
  - Plan-and-Execute：先出计划，再逐步执行，每步可以是不同的 Agent
经验法则：**能用单 Agent + 好工具解决的，不要上多 Agent。** 多 Agent 的价值在于
隔离上下文（每个子 Agent 的上下文更干净）和分工（不同的提示词/工具/模型）。

本课的图：
    START -> router --(math)--> math_expert --> END
                   \\-(knowledge)--> kb_expert --> END
                   \\-(chat)--> chit_chat --> END

运行：uv run python -m pagent.lessons.l06_multi_agent
"""

from __future__ import annotations

from typing import Literal

from langchain.agents import create_agent
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel

from pagent.config import get_chat_model
from pagent.rag import make_search_tool
from pagent.tools import calculator


class Route(BaseModel):
    """把用户的问题分给最合适的专家。"""

    expert: Literal["math", "knowledge", "chat"]
    reason: str


def build_router_graph(model, kb_tool):
    router_llm = model.with_structured_output(Route)
    math_expert = create_agent(model, [calculator], system_prompt="你是数学专家，所有计算都必须用 calculator。")
    kb_expert = create_agent(model, [kb_tool], system_prompt="你是 Agent 技术专家，先检索知识库再回答，注明来源。")

    def router(state: MessagesState) -> Command[Literal["math_expert", "kb_expert", "chit_chat"]]:
        route = router_llm.invoke(
            [{"role": "system", "content": "判断问题类型：math=需要计算；knowledge=AI/Agent 技术问题；chat=闲聊"}]
            + state["messages"]
        )
        print(f"🧭 路由到 {route.expert}：{route.reason}")
        target = {"math": "math_expert", "knowledge": "kb_expert", "chat": "chit_chat"}[route.expert]
        return Command(goto=target)  # Command 同时表达“更新状态”和“跳转到哪个节点”

    def chit_chat(state: MessagesState) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("router", router)
    g.add_node("math_expert", math_expert)  # 编译好的 Agent 本身就是一个图，可以直接当节点（子图）
    g.add_node("kb_expert", kb_expert)
    g.add_node("chit_chat", chit_chat)
    g.add_edge(START, "router")
    for n in ("math_expert", "kb_expert", "chit_chat"):
        g.add_edge(n, END)
    return g.compile()


if __name__ == "__main__":
    graph = build_router_graph(get_chat_model(), make_search_tool())
    for q in ["1234 * 5678 - 999 等于多少？", "MCP 和 Function Calling 有什么区别？", "你好呀"]:
        print(f"\n用户：{q}")
        out = graph.invoke({"messages": [{"role": "user", "content": q}]})
        print("回答：", out["messages"][-1].text[:300])
