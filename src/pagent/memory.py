"""记忆系统：短期记忆 vs 长期记忆。

短期记忆（Short-term / Thread memory）
-------------------------------------
= 当前这一次对话的消息历史。LangGraph 用 **Checkpointer** 实现：每走一步就把
整个 State（含 messages）存一份快照，按 thread_id 区分会话。
好处：断点续跑、Human-in-the-loop 暂停/恢复、时间旅行调试都靠它。
历史太长时用 SummarizationMiddleware 压缩（见 agent.py）。

长期记忆（Long-term / Cross-thread memory）
-----------------------------------------
= 跨会话也要记住的东西：用户偏好、事实、过往经验。LangGraph 用 **Store** 实现：
一个按 namespace 组织的 JSON 文档库，可以开启向量索引做语义搜索。
写入方式两种：
  1. 热路径（hot path）：给模型一个 `remember` 工具，让它自己决定记什么 —— 本项目采用
  2. 后台（background）：对话结束后再用另一个 LLM 抽取记忆，不拖慢响应
读取方式：每轮调用模型前，按用户问题语义检索相关记忆，注入 system prompt（见 middleware.py）。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from langchain.tools import ToolRuntime, tool


@dataclass
class UserContext:
    """运行时上下文（Runtime Context）：每次调用时传入、不会被模型修改的数据，
    例如当前用户 id、权限、数据库连接。工具和中间件通过 runtime.context 读取。"""

    user_id: str = "default"


def memory_namespace(user_id: str) -> tuple[str, ...]:
    return ("users", user_id, "memories")


def store_index_config(embeddings, dims: int) -> dict:
    # fields 指定对 value 里的哪个字段做向量化
    return {"dims": dims, "embed": embeddings, "fields": ["text"]}


@tool
async def remember(fact: str, runtime: ToolRuntime[UserContext]) -> str:
    """把一条关于用户的长期信息保存下来（偏好、身份、长期目标、重要约定等），
    以便未来的任何对话都能用到。只保存稳定、有复用价值的事实，用一句完整的陈述句。"""
    ns = memory_namespace(runtime.context.user_id)
    # 先查是否已有几乎相同的记忆，避免重复堆积
    existing = await runtime.store.asearch(ns, query=fact, limit=1)
    if existing and existing[0].score is not None and existing[0].score > 0.95:
        await runtime.store.aput(ns, existing[0].key, {"text": fact})
        return f"已更新记忆: {fact}"
    await runtime.store.aput(ns, uuid.uuid4().hex[:8], {"text": fact})
    return f"已记住: {fact}"


@tool
async def recall(query: str, runtime: ToolRuntime[UserContext]) -> str:
    """按语义检索关于用户的长期记忆。"""
    items = await runtime.store.asearch(memory_namespace(runtime.context.user_id), query=query, limit=5)
    if not items:
        return "没有相关记忆。"
    return "\n".join(f"- ({it.key}) {it.value['text']}" for it in items)


@tool
async def forget(memory_id: str, runtime: ToolRuntime[UserContext]) -> str:
    """删除一条长期记忆，memory_id 来自 recall 的结果括号里的编号。"""
    await runtime.store.adelete(memory_namespace(runtime.context.user_id), memory_id)
    return f"已删除记忆 {memory_id}"


MEMORY_TOOLS = [remember, recall, forget]
