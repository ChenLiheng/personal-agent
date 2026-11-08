"""本地工具（Function Calling 的“函数”那一半）。

学习要点
--------
- 模型本身不会执行任何代码。所谓“调用工具”，是模型输出一段结构化 JSON：
  {"name": "calculator", "args": {"expression": "1+1"}}，由我们的程序真正执行，
  再把结果作为 ToolMessage 喂回模型。
- `@tool` 装饰器会读取：函数名 -> 工具名，docstring -> 工具描述，
  类型注解 -> JSON Schema 参数。**描述写得好不好，直接决定模型会不会用、用得对不对。**
- 工具返回值最好是简短、结构清晰的字符串；报错也要返回可读信息，让模型能自我纠正。
- 有副作用的工具（写文件、发邮件、下单）要配合 Human-in-the-loop 审批，见 agent.py。
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from langchain.tools import tool

from pagent.config import WORKSPACE_DIR

# --------------------------------------------------------------------------- #
# 1. 纯函数工具：安全计算器
# --------------------------------------------------------------------------- #
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("只支持 + - * / ** % 和括号")


@tool
def calculator(expression: str) -> str:
    """计算一个数学表达式并返回结果，例如 "(3 + 5) * 2 ** 3"。

    大模型不擅长精确算术，遇到任何需要精确计算的地方都应调用本工具。
    """
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:  # 把错误返回给模型，而不是抛异常让整个 Agent 崩掉
        return f"计算失败: {e}"


# --------------------------------------------------------------------------- #
# 2. 读取外部世界：时间
# --------------------------------------------------------------------------- #
@tool
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取指定 IANA 时区的当前日期和时间，例如 "Asia/Shanghai"、"America/New_York"。"""
    try:
        return datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M:%S %A (%Z)")
    except Exception as e:
        return f"未知时区 {timezone}: {e}"


# --------------------------------------------------------------------------- #
# 3. 网络工具：抓取网页
# --------------------------------------------------------------------------- #
@tool
def fetch_url(url: str, max_chars: int = 4000) -> str:
    """抓取一个 http(s) 网页并返回纯文本（截断到 max_chars 个字符）。
    适合读取用户给出的链接、文档页面。"""
    if not url.startswith(("http://", "https://")):
        return "只支持 http/https 链接"
    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True, headers={"User-Agent": "pagent/0.1"})
        resp.raise_for_status()
    except Exception as e:
        return f"请求失败: {e}"
    text = resp.text
    if "html" in resp.headers.get("content-type", ""):
        import re

        text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
    return text[:max_chars]


# --------------------------------------------------------------------------- #
# 4. 有副作用的工具：写文件（会被 Human-in-the-loop 拦截审批）
# --------------------------------------------------------------------------- #
def _safe_path(filename: str):
    path = (WORKSPACE_DIR / filename).resolve()
    if WORKSPACE_DIR.resolve() not in path.parents:
        raise ValueError("只能写入 workspace/ 目录")
    return path


@tool
def write_file(filename: str, content: str) -> str:
    """把内容写入 workspace/ 目录下的文件（会覆盖同名文件）。filename 是相对路径，如 "notes/today.md"。"""
    try:
        path = _safe_path(filename)
    except ValueError as e:
        return str(e)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"已写入 {path.relative_to(WORKSPACE_DIR.parent)}（{len(content)} 字符）"


@tool
def list_workspace_files() -> str:
    """列出 workspace/ 目录下已有的文件。"""
    if not WORKSPACE_DIR.exists():
        return "(空)"
    files = [str(p.relative_to(WORKSPACE_DIR)) for p in WORKSPACE_DIR.rglob("*") if p.is_file()]
    return "\n".join(sorted(files)) or "(空)"


LOCAL_TOOLS = [calculator, get_current_time, fetch_url, write_file, list_workspace_files]
