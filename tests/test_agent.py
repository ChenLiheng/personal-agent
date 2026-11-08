"""端到端测试：用剧本模型驱动完整的 PAgent，验证工具、MCP、长期记忆、HITL、短期记忆都串起来了。"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from pagent.agent import open_agent
from pagent.config import HashingEmbeddings
from pagent.lessons.l02_langgraph_from_scratch import build_graph
from pagent.memory import UserContext, memory_namespace
from pagent.rag import get_vectorstore


def call(name, args, id):
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


@pytest.fixture
def agent_factory(isolated):
    from pagent.mcp_server import mcp

    def make(model):
        emb = HashingEmbeddings()
        return open_agent(
            model,
            emb,
            db_path=str(isolated / "agent.sqlite"),
            mcp_target=mcp,  # 进程内 MCP Server
            vectorstore=get_vectorstore(emb, persist_dir=isolated / "chroma"),
        )

    return make


@pytest.mark.asyncio
async def test_full_agent_flow(agent_factory, scripted_model, isolated):
    model = scripted_model(
        # 第 1 轮：一次并行调用 3 个工具（本地工具 + 长期记忆 + MCP 工具）
        AIMessage(
            content="",
            tool_calls=[
                call("calculator", {"expression": "6*7"}, "c1"),
                call("remember", {"fact": "用户最喜欢的语言是 Python"}, "c2"),
                call("add_todo", {"title": "学习 LangGraph"}, "c3"),
            ],
        ),
        AIMessage(content="算好了，是 42，已记住你的偏好并添加了待办。"),
        # 第 2 轮：写文件 -> 被 HITL 拦截
        AIMessage(content="", tool_calls=[call("write_file", {"filename": "a.md", "content": "hi"}, "c4")]),
        AIMessage(content="文件已写好。"),
        # 第 3 轮：检索知识库
        AIMessage(content="", tool_calls=[call("search_knowledge_base", {"query": "MCP 传输方式"}, "c5")]),
        AIMessage(content="MCP 支持 stdio 和 Streamable HTTP。"),
    )
    config = {"configurable": {"thread_id": "t1"}}
    ctx = UserContext(user_id="alice")

    async with agent_factory(model) as agent:
        # ---- 第 1 轮 ----
        out = await agent.ainvoke({"messages": [{"role": "user", "content": "6*7=? 我喜欢 Python，加个待办"}]}, config, context=ctx)
        tool_results = {m.tool_call_id: m.content for m in out["messages"] if isinstance(m, ToolMessage)}
        assert tool_results["c1"] == "42"
        assert "已记住" in tool_results["c2"]
        assert "学习 LangGraph" in str(tool_results["c3"])
        memories = await agent.store.asearch(memory_namespace("alice"), query="编程语言")
        assert memories[0].value["text"] == "用户最喜欢的语言是 Python"

        # ---- 第 2 轮：Human-in-the-loop ----
        await agent.ainvoke({"messages": [{"role": "user", "content": "写个文件"}]}, config, context=ctx)
        state = await agent.aget_state(config)
        assert state.interrupts, "write_file 应该被拦截等待审批"
        assert state.interrupts[0].value["action_requests"][0]["name"] == "write_file"
        assert not (isolated / "workspace" / "a.md").exists()  # 批准前不会执行

        await agent.ainvoke(Command(resume={"decisions": [{"type": "approve"}]}), config, context=ctx)
        assert (isolated / "workspace" / "a.md").read_text() == "hi"

        # ---- 第 3 轮：RAG ----
        out = await agent.ainvoke({"messages": [{"role": "user", "content": "MCP 有哪些传输方式"}]}, config, context=ctx)
        rag_result = next(m for m in out["messages"] if isinstance(m, ToolMessage) and m.tool_call_id == "c5")
        assert "03-function-calling-mcp.md" in rag_result.content

        # 短期记忆：同一个 thread 的所有消息都在 checkpoint 里
        assert len(out["messages"]) >= 12


@pytest.mark.asyncio
async def test_long_term_memory_survives_restart(agent_factory, scripted_model):
    m1 = scripted_model(
        AIMessage(content="", tool_calls=[call("remember", {"fact": "用户住在杭州"}, "x")]),
        AIMessage(content="记住了"),
    )
    async with agent_factory(m1) as agent:
        await agent.ainvoke({"messages": [{"role": "user", "content": "我住杭州"}]},
                            {"configurable": {"thread_id": "a"}}, context=UserContext("bob"))
    # 重新打开（模拟重启进程），新会话也能取到长期记忆
    async with agent_factory(scripted_model(AIMessage(content="ok"))) as agent:
        items = await agent.store.asearch(memory_namespace("bob"), query="住在哪里")
        assert items[0].value["text"] == "用户住在杭州"


def test_lesson2_graph_interrupt_and_reject(scripted_model, isolated):
    model = scripted_model(
        AIMessage(content="", tool_calls=[call("write_file", {"filename": "x.txt", "content": "1"}, "w1")]),
        AIMessage(content="好的，不写了。"),
    )
    graph = build_graph(model)
    config = {"configurable": {"thread_id": "l2"}}
    graph.invoke({"messages": [{"role": "user", "content": "写文件"}]}, config)
    assert graph.get_state(config).interrupts
    out = graph.invoke(Command(resume="no"), config)
    assert out["messages"][-1].text == "好的，不写了。"
    assert not (isolated / "workspace" / "x.txt").exists()
