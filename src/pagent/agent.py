"""把所有零件组装成一个完整的 Agent。

┌──────────────────────────── PAgent (create_agent / LangGraph) ───────────────────────────┐
│ Middleware: 动态提示词(注入长期记忆) · 摘要压缩 · 人工审批 · 工具重试 · 调用上限 · PII · 日志 │
│                                                                                           │
│   ┌────────┐  tool_calls  ┌──────────────────────────────────────────────────────────┐   │
│   │  LLM   │ ───────────▶ │ Tools                                                    │   │
│   │        │ ◀─────────── │  本地 @tool: calculator / time / fetch_url / write_file   │   │
│   └────────┘ ToolMessage  │  RAG:  search_knowledge_base ──▶ Chroma 向量库           │   │
│                           │  记忆: remember / recall / forget ──▶ Store(SQLite+向量)  │   │
│                           │  MCP:  todo server（子进程 stdio）                        │   │
│                           │  子 Agent: ask_researcher（多 Agent 协作）                │   │
│                           └──────────────────────────────────────────────────────────┘   │
│ Checkpointer(SQLite): 短期记忆 / 断点续跑 / HITL 暂停恢复                                 │
└───────────────────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolRetryMiddleware,
)
from langchain.tools import tool
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore

from pagent.config import PROJECT_ROOT, RUNTIME_DIR, get_chat_model, get_embeddings
from pagent.memory import MEMORY_TOOLS, UserContext, store_index_config
from pagent.middleware import log_tool_calls, memory_aware_prompt
from pagent.rag import get_vectorstore, make_search_tool
from pagent.tools import LOCAL_TOOLS, fetch_url

RESEARCHER_PROMPT = """你是一个调研员子 Agent。针对给你的调研任务：
1. 用 search_knowledge_base 多角度检索（至少换 2 组关键词）；必要时用 fetch_url 读取网页。
2. 输出一份结构化的小结：要点列表 + 每条的来源。不要寒暄。"""


def default_mcp_config() -> dict[str, Any]:
    """MCP 服务器配置（和 Claude Desktop / Cursor 的 mcpServers 格式一致）。

    默认只有本项目自带的 todo 服务；项目根目录放一个 mcp_servers.json
    就可以接入更多现成的 MCP 服务（文件系统、GitHub、浏览器、数据库……）。
    """
    servers: dict[str, Any] = {
        "todo": {"command": sys.executable, "args": ["-m", "pagent.mcp_server"]},
    }
    extra = PROJECT_ROOT / "mcp_servers.json"
    if extra.exists():
        servers.update(json.loads(extra.read_text(encoding="utf-8")).get("mcpServers", {}))
    return {"mcpServers": servers}


def build_researcher(model: BaseChatModel, kb_tool):
    """多 Agent 的最简单形态：“把子 Agent 包装成工具”（Supervisor / Agents-as-tools 模式）。
    主 Agent 负责理解意图和汇总，子 Agent 带着自己独立的上下文专心干一件事，
    只把结论返回，不会把检索过程的大量中间内容塞满主 Agent 的上下文。"""
    researcher = create_agent(
        model, tools=[kb_tool, fetch_url], system_prompt=RESEARCHER_PROMPT, name="researcher"
    )

    @tool
    async def ask_researcher(task: str) -> str:
        """把一个需要多次检索、综合多处资料的调研任务交给调研员子 Agent，返回它的调研小结。
        task 要写清楚：调研什么、关注哪些方面。"""
        result = await researcher.ainvoke({"messages": [{"role": "user", "content": task}]})
        return result["messages"][-1].text

    return ask_researcher


@asynccontextmanager
async def open_agent(
    model: BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
    *,
    db_path: str | None = None,
    mcp_target: Any = None,
    vectorstore=None,
):
    """异步上下文管理器：打开数据库 / MCP 连接，产出编译好的 Agent，退出时统一清理。"""
    from langchain.mcp import MCPAdapter

    model = model or get_chat_model()
    embeddings = embeddings or get_embeddings()
    dims = len(embeddings.embed_query("维度探测"))
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    db = db_path or str(RUNTIME_DIR / "agent.sqlite")

    async with AsyncExitStack() as stack:
        # 短期记忆：Checkpointer
        checkpointer = await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(db))
        # 长期记忆：带向量索引的 Store
        store = await stack.enter_async_context(
            AsyncSqliteStore.from_conn_string(db, index=store_index_config(embeddings, dims))
        )
        await store.setup()
        # MCP：连接服务器，拿到工具列表（已经转换成 LangChain Tool）
        adapter = await stack.enter_async_context(MCPAdapter(mcp_target or default_mcp_config()))
        mcp_tools = await adapter.list_tools()

        kb_tool = make_search_tool(vectorstore or get_vectorstore(embeddings))
        tools = [*LOCAL_TOOLS, kb_tool, *MEMORY_TOOLS, *mcp_tools, build_researcher(model, kb_tool)]

        agent = create_agent(
            model,
            tools=tools,
            middleware=[
                # 1) 每次调用模型前，检索长期记忆并生成 system prompt
                memory_aware_prompt,
                # 2) 护栏：用户输入里出现信用卡号就打码，不发给模型
                PIIMiddleware("credit_card", strategy="mask"),
                # 3) 上下文管理：历史超过约 8000 token 时，把旧消息摘要成一条，保留最近 12 条
                SummarizationMiddleware(model, trigger=("tokens", 8000), keep=("messages", 12)),
                # 4) Human-in-the-loop：有副作用的工具先暂停，等人批准/修改/拒绝
                HumanInTheLoopMiddleware(
                    interrupt_on={"write_file": {"allowed_decisions": ["approve", "edit", "reject"]}}
                ),
                # 5) 容错：网络类工具失败自动指数退避重试
                ToolRetryMiddleware(max_retries=2, tools=["fetch_url"], initial_delay=0.5),
                # 6) 成本控制：单次运行最多调用模型 15 次，防止死循环
                ModelCallLimitMiddleware(run_limit=15, exit_behavior="end"),
                # 7) 可观测性：打印每次工具调用
                log_tool_calls,
            ],
            checkpointer=checkpointer,
            store=store,
            context_schema=UserContext,
            name="pagent",
        )
        yield agent
