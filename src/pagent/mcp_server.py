"""一个独立的 MCP Server：待办事项（Todo）服务。

MCP 是什么？
-----------
Model Context Protocol：一个开放协议，规定“AI 应用（Host/Client）”和
“能力提供方（Server）”之间怎么通信（JSON-RPC）。好比 AI 世界的 USB-C：
你写一次 MCP Server，Claude Desktop / Claude Code / Cursor / 你自己的 Agent 都能直接用。

MCP Server 能暴露三类东西：
  - Tools     模型可调用的函数（≈ function calling，最常用）
  - Resources 只读数据，用 URI 访问（文件、数据库记录），由应用决定何时读取
  - Prompts   预置的提示词模板，通常由用户主动选择（如斜杠命令）

传输方式：
  - stdio            Client 把 Server 作为子进程启动，用标准输入输出通信（本地工具首选）
  - Streamable HTTP  Server 作为 HTTP 服务独立部署，可远程、多客户端共享

和 Function Calling 的关系：Function Calling 是“模型 <-> 你的程序”之间的接口；
MCP 是“你的程序 <-> 外部工具服务”之间的接口。Agent 从 MCP Server 拿到工具列表后，
仍然是转换成 function calling 的格式交给模型。

运行方式：
  uv run python -m pagent.mcp_server              # stdio（被 Agent 自动拉起，不用手动跑）
  uv run python -m pagent.mcp_server --http 8765  # HTTP: http://127.0.0.1:8765/mcp
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from fastmcp import FastMCP

DB_FILE = Path(__file__).resolve().parents[2] / ".runtime" / "todos.json"

mcp = FastMCP("todo-server", instructions="管理用户的待办事项。")


def _load() -> list[dict]:
    if DB_FILE.exists():
        return json.loads(DB_FILE.read_text(encoding="utf-8"))
    return []


def _save(todos: list[dict]) -> None:
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    DB_FILE.write_text(json.dumps(todos, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------------- Tools ----------------------------- #
@mcp.tool
def add_todo(title: str, due: str | None = None) -> dict:
    """新增一条待办。due 是可选的截止日期，格式 YYYY-MM-DD。"""
    todos = _load()
    item = {
        "id": max((t["id"] for t in todos), default=0) + 1,
        "title": title,
        "due": due,
        "done": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    todos.append(item)
    _save(todos)
    return item  # 返回 dict 时，FastMCP 会同时生成 structuredContent（结构化输出）


@mcp.tool
def list_todos(include_done: bool = False) -> list[dict]:
    """列出待办事项，默认只列未完成的。"""
    return [t for t in _load() if include_done or not t["done"]]


@mcp.tool
def complete_todo(todo_id: int) -> str:
    """把指定 id 的待办标记为已完成。"""
    todos = _load()
    for t in todos:
        if t["id"] == todo_id:
            t["done"] = True
            _save(todos)
            return f"已完成: {t['title']}"
    return f"找不到 id={todo_id} 的待办"


# ----------------------------- Resources ----------------------------- #
@mcp.resource("todo://summary")
def todo_summary() -> str:
    """待办概览（只读资源）。"""
    todos = _load()
    done = sum(t["done"] for t in todos)
    return f"共 {len(todos)} 条待办，已完成 {done} 条，未完成 {len(todos) - done} 条。"


# ----------------------------- Prompts ----------------------------- #
@mcp.prompt
def weekly_review() -> str:
    """生成一个“每周回顾”的提示词模板。"""
    return "请列出我所有未完成的待办，按截止日期排序，并给出本周的优先级建议。"


if __name__ == "__main__":
    if "--http" in sys.argv:
        port = int(sys.argv[sys.argv.index("--http") + 1])
        mcp.run(transport="http", host="127.0.0.1", port=port)
    else:
        mcp.run(show_banner=False)  # 默认 stdio
