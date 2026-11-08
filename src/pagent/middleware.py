"""自定义 Middleware（中间件）。

LangChain 1.x 的 `create_agent` 内部就是一个 LangGraph 循环：
    before_agent -> [before_model -> model -> after_model -> tools]* -> after_agent
Middleware 允许你在这些节点上“挂钩子”，而不必自己重写整张图：
  - before_model / after_model: 改消息、做裁剪、做护栏检查
  - wrap_model_call:  包住一次模型调用（换模型、改 system prompt、重试、缓存）
  - wrap_tool_call:   包住一次工具调用（日志、计时、权限、重试、审批）
  - dynamic_prompt:   每次调用模型前动态生成 system prompt
框架自带的（Summarization、HumanInTheLoop、ToolRetry、ModelCallLimit、PII…）见 agent.py。
"""

from __future__ import annotations

import time

from langchain.agents.middleware import ModelRequest, dynamic_prompt, wrap_tool_call
from langchain_core.messages import HumanMessage
from rich.console import Console

from pagent.memory import memory_namespace

console = Console(stderr=True)

BASE_PROMPT = """你是 PAgent，一个用来学习 Agent 技术的个人助手。

工作原则：
- 需要精确计算时用 calculator；需要当前时间用 get_current_time。
- 关于 AI Agent 技术的问题，先用 search_knowledge_base 检索知识库，再结合检索结果回答，并注明来源文件。
- 复杂的调研类问题可以委派给 researcher 子 Agent。
- 用户透露了稳定的个人偏好或事实时，用 remember 记下来。
- 待办事项通过 todo 相关的 MCP 工具管理。
- 回答使用中文，简洁、有条理。"""


@dynamic_prompt
async def memory_aware_prompt(request: ModelRequest) -> str:
    """每次调用模型前：用最近一条用户消息去检索长期记忆，拼进 system prompt。"""
    store = request.runtime.store
    ctx = request.runtime.context
    if store is None or ctx is None:
        return BASE_PROMPT
    last_user = next((m for m in reversed(request.messages) if isinstance(m, HumanMessage)), None)
    if last_user is None:
        return BASE_PROMPT
    items = await store.asearch(memory_namespace(ctx.user_id), query=str(last_user.content), limit=5)
    if not items:
        return BASE_PROMPT
    memories = "\n".join(f"- {it.value['text']}" for it in items)
    return f"{BASE_PROMPT}\n\n<关于用户的长期记忆>\n{memories}\n</关于用户的长期记忆>"


@wrap_tool_call
async def log_tool_calls(request, handler):
    """给每次工具调用打日志 + 计时，相当于一个最简单的“可观测性”实现。"""
    call = request.tool_call
    console.print(f"[dim]🔧 调用工具 {call['name']}({call['args']})[/dim]")
    t0 = time.perf_counter()
    result = await handler(request)
    content = str(getattr(result, "content", result))
    console.print(f"[dim]   ↳ {time.perf_counter() - t0:.2f}s, 返回 {content[:120]!r}[/dim]")
    return result
