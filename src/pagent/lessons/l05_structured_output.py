"""第 5 课：结构化输出 —— 让模型返回“程序能直接用”的数据，而不是一段话。

两种常见实现：
  1. Tool 策略：把 Pydantic 模型伪装成一个“工具”，强制模型调用它，参数就是结构化结果。
     几乎所有支持 function calling 的模型都能用。
  2. Provider 原生策略：厂商原生的 JSON Schema 约束解码（OpenAI / Claude 等都已支持），更可靠。
`create_agent(response_format=...)` 会自动选择合适的策略；Agent 跑完工具循环后，
结果放在 result["structured_response"] 里。

运行：uv run python -m pagent.lessons.l05_structured_output
"""

from __future__ import annotations

from typing import Literal

from langchain.agents import create_agent
from pydantic import BaseModel, Field

from pagent.config import get_chat_model
from pagent.tools import calculator


class Task(BaseModel):
    title: str = Field(description="任务标题，动词开头")
    priority: Literal["高", "中", "低"]
    estimated_hours: float = Field(description="预估工时（小时）")


class StudyPlan(BaseModel):
    goal: str
    tasks: list[Task]
    total_hours: float = Field(description="所有任务工时之和，必须用 calculator 精确计算")


if __name__ == "__main__":
    model = get_chat_model()

    # 方式一：不需要工具时，直接 with_structured_output
    extractor = model.with_structured_output(Task)
    print(extractor.invoke("下周三前要把 MCP server 写完，挺急的，估计 6 个小时"))

    # 方式二：Agent 先用工具，最后按 schema 输出
    agent = create_agent(model, tools=[calculator], response_format=StudyPlan)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "帮我制定一个两周学会 LangGraph 的计划，4~6 个任务"}]}
    )
    plan: StudyPlan = result["structured_response"]
    print(plan.model_dump_json(indent=2))
