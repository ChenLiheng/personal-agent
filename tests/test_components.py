"""单个组件的测试：工具、RAG、MCP Server。"""

import pytest
from fastmcp import Client

from pagent.config import HashingEmbeddings
from pagent.rag import get_vectorstore, load_documents, make_search_tool, split_markdown
from pagent.tools import calculator, get_current_time, write_file


def test_calculator():
    assert calculator.invoke({"expression": "(3 + 5) * 2 ** 3"}) == "64"
    assert "计算失败" in calculator.invoke({"expression": "__import__('os')"})


def test_tool_schema_generated_from_signature():
    schema = calculator.tool_call_schema.model_json_schema()
    assert schema["required"] == ["expression"]
    assert "数学表达式" in calculator.description


def test_time_tool():
    assert "未知时区" in get_current_time.invoke({"timezone": "Mars/Base"})


def test_write_file_is_sandboxed(isolated):
    assert "只能写入" in write_file.invoke({"filename": "../evil.txt", "content": "x"})
    write_file.invoke({"filename": "a/b.txt", "content": "hello"})
    assert (isolated / "workspace" / "a" / "b.txt").read_text() == "hello"


def test_split_markdown_keeps_titles_and_overlap():
    text = "# 标题A\n" + "字" * 1200 + "\n# 标题B\n短内容"
    docs = split_markdown(text, "x.md", chunk_size=500, overlap=100)
    assert [d.metadata["section"] for d in docs] == ["标题A"] * 3 + ["标题B"]
    assert all(d.page_content.startswith(d.metadata["section"]) for d in docs)


def test_rag_search_finds_relevant_chunk(tmp_path):
    assert len(load_documents()) > 10
    vs = get_vectorstore(HashingEmbeddings(), persist_dir=tmp_path / "chroma")
    search = make_search_tool(vs, k=2)  # 首次使用自动建索引
    out = search.invoke({"query": "什么时候用向量数据库"})
    assert "04-rag-vector-db.md" in out


@pytest.mark.asyncio
async def test_mcp_server_in_memory(isolated):
    from pagent.mcp_server import mcp

    async with Client(mcp) as client:  # 进程内直连，不起子进程
        names = {t.name for t in await client.list_tools()}
        assert {"add_todo", "list_todos", "complete_todo"} <= names
        r = await client.call_tool("add_todo", {"title": "写测试"})
        assert r.structured_content["id"] == 1
        await client.call_tool("complete_todo", {"todo_id": 1})
        summary = await client.read_resource("todo://summary")
        assert "已完成 1 条" in summary[0].text
        prompt = await client.get_prompt("weekly_review")
        assert "待办" in prompt.messages[0].content.text
