"""第 1 课：不用任何框架，手写一个 Agent 循环。

所有 Agent 框架的内核都是下面这个 while 循环（ReAct 模式：思考 -> 行动 -> 观察）：

    messages = [user 问题]
    while True:
        resp = LLM(messages, tools=工具的 JSON Schema 列表)
        if resp 里没有 tool_calls:  return resp.text      # 模型认为可以直接回答了
        for call in resp.tool_calls:
            result = 执行本地函数(call.name, call.args)       # 由“我们”执行，不是模型
            messages.append(工具结果, 并带上 call.id)          # id 用来对应是哪次调用
        # 回到循环顶部，让模型看着工具结果继续思考

两大主流协议格式不一样（但思想完全相同），本课两种都实现了：
  - OpenAI Chat Completions 格式：几乎所有国产模型 / Ollama / vLLM 都兼容
  - Anthropic Messages 格式：Claude 原生格式，DeepSeek 等也提供兼容端点

运行：uv run python -m pagent.lessons.l01_raw_function_calling
"""

from __future__ import annotations

import json
from datetime import datetime

from pagent.config import settings


# ---------------- 1. 定义工具：普通 Python 函数 + 手写 JSON Schema ----------------
def get_weather(city: str) -> str:
    fake = {"北京": "晴 12°C", "上海": "小雨 16°C", "杭州": "多云 15°C"}
    return fake.get(city, "未知城市")


def get_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


FUNCTIONS = {"get_weather": get_weather, "get_time": get_time}

# 工具说明书：名称、描述、参数的 JSON Schema。模型只看得到这些，看不到函数实现。
TOOL_SPECS = [
    {
        "name": "get_weather",
        "description": "查询某个城市当前的天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名，如 北京"}},
            "required": ["city"],
        },
    },
    {
        "name": "get_time",
        "description": "获取当前时间",
        "parameters": {"type": "object", "properties": {}},
    },
]

QUESTION = "现在几点了？顺便告诉我北京和上海的天气，哪个更适合出门？"


def run_tool(name: str, args: dict) -> str:
    print(f"  🔧 执行 {name}({args})")
    try:
        return str(FUNCTIONS[name](**args))
    except Exception as e:
        return f"工具出错: {e}"


# ---------------- 2a. OpenAI 格式的循环 ----------------
def run_openai_style(question: str, max_steps: int = 6) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    tools = [{"type": "function", "function": spec} for spec in TOOL_SPECS]
    messages: list[dict] = [{"role": "user", "content": question}]

    for step in range(max_steps):
        resp = client.chat.completions.create(model=settings.llm_model, messages=messages, tools=tools)
        msg = resp.choices[0].message
        print(f"[第 {step + 1} 步] finish_reason={resp.choices[0].finish_reason}")
        if not msg.tool_calls:
            return msg.content or ""
        # 必须把带 tool_calls 的 assistant 消息原样放回历史
        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:  # 模型可能一次返回多个调用（并行工具调用）
            result = run_tool(call.function.name, json.loads(call.function.arguments or "{}"))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return "达到最大步数，停止。"


# ---------------- 2b. Anthropic 格式的循环 ----------------
def run_anthropic_style(question: str, max_steps: int = 6) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    # 区别 1：字段名是 input_schema 而不是 parameters
    tools = [{"name": s["name"], "description": s["description"], "input_schema": s["parameters"]} for s in TOOL_SPECS]
    messages: list[dict] = [{"role": "user", "content": question}]

    for step in range(max_steps):
        resp = client.messages.create(model=settings.llm_model, max_tokens=1024, messages=messages, tools=tools)
        print(f"[第 {step + 1} 步] stop_reason={resp.stop_reason}")
        # 区别 2：内容是 block 列表，tool_use 是其中一种 block
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            return "".join(b.text for b in resp.content if b.type == "text")
        messages.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})
        # 区别 3：工具结果放在 role=user 的消息里，类型是 tool_result
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": b.id, "content": run_tool(b.name, b.input)}
                    for b in tool_uses
                ],
            }
        )
    return "达到最大步数，停止。"


if __name__ == "__main__":
    print(f"provider={settings.llm_provider} model={settings.llm_model}\n问题：{QUESTION}\n")
    runner = run_anthropic_style if settings.llm_provider == "anthropic" else run_openai_style
    print("\n最终回答：\n" + runner(QUESTION))
