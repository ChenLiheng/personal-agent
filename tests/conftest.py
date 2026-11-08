"""测试不调用真实大模型，也不下载 embedding 模型：
- 用 ScriptedChatModel 按剧本依次返回预设的 AIMessage（含 tool_calls）
- 用 HashingEmbeddings 代替真实向量模型
这也是测试 Agent 的常用技巧：把“模型决策”固定下来，只测试编排与工具逻辑。
"""

import json
import os

from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

os.environ["EMBEDDING_BACKEND"] = "hash"  # 必须在 import pagent 之前设置

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel


class ScriptedChatModel(GenericFakeChatModel):
    """在 GenericFakeChatModel 基础上补两件事：
    1. bind_tools：直接忽略工具定义（剧本里已经写好要调用什么）
    2. _stream：原版流式模式下会丢掉只有 tool_calls 的消息，这里整条作为一个 chunk 输出
    """

    def bind_tools(self, tools, **kwargs):
        return self

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self._generate(messages, stop, run_manager, **kwargs).generations[0].message
        chunks = [
            {"name": c["name"], "args": json.dumps(c["args"], ensure_ascii=False), "id": c["id"], "index": i}
            for i, c in enumerate(msg.tool_calls)
        ]
        yield ChatGenerationChunk(
            message=AIMessageChunk(content=msg.content, tool_call_chunks=chunks, id=msg.id)
        )


@pytest.fixture
def scripted_model():
    def make(*messages):
        return ScriptedChatModel(messages=iter(messages))

    return make


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """把所有会写磁盘的地方重定向到临时目录。"""
    import pagent.mcp_server as mcp_server
    import pagent.tools as tools

    monkeypatch.setattr(tools, "WORKSPACE_DIR", tmp_path / "workspace")
    monkeypatch.setattr(mcp_server, "DB_FILE", tmp_path / "todos.json")
    return tmp_path
