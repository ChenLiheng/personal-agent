# Agent 是什么

Agent = LLM + 工具 + 循环。LLM 负责推理和决策，工具负责与外部世界交互，循环让模型能够“观察工具结果后继续思考”，直到任务完成。Anthropic 将其定义为“LLM 在循环中自主使用工具来完成任务”的系统；与之相对的是 Workflow（工作流），它的步骤由代码预先编排，LLM 只在固定节点上被调用。

# Workflow 与 Agent 的取舍

能用确定性的 Workflow 解决的问题，就不要使用自主 Agent。常见的 Workflow 模式包括：Prompt Chaining（串联）、Routing（路由）、Parallelization（并行）、Orchestrator-Workers（编排者与工作者）、Evaluator-Optimizer（评估与优化循环）。只有当步骤无法预知、需要模型根据中间结果动态决定下一步时，才需要 Agent。Agent 更灵活，但成本更高、延迟更大、行为更难预测。

# ReAct 循环

ReAct（Reason + Act）是最经典的 Agent 模式：思考 -> 调用工具 -> 观察结果 -> 再思考。现代模型原生支持 tool calling，ReAct 不再需要用提示词解析 "Action:" 文本，而是直接由模型输出结构化的 tool_calls。循环在模型不再请求工具时结束，此外还需要设置最大步数，防止死循环。

# Agent 的核心组件清单

1. 模型（Model）：支持 tool calling 的 LLM，可以按任务难度选择大模型或小模型。
2. 指令（System Prompt）：角色、目标、工具使用原则、输出格式。
3. 工具（Tools）：本地函数、API、MCP 服务器、子 Agent。
4. 记忆（Memory）：短期记忆（当前对话历史）与长期记忆（跨会话的用户事实、偏好）。
5. 知识（Knowledge/RAG）：通过检索获取私有文档。
6. 编排（Orchestration）：循环、分支、并行、多 Agent。
7. 护栏（Guardrails）：输入输出检查、人工审批、权限、调用上限。
8. 可观测与评估（Observability & Eval）：追踪每一步，并用数据集持续评估。

# 上下文工程

上下文工程（Context Engineering）是 2025 年以后 Agent 开发中最重要的概念：模型每一步能看到什么，决定了它做得好不好。具体手段包括精简工具描述、按需检索（而不是一次性塞入全部资料）、压缩/摘要历史、用子 Agent 隔离上下文、把大段中间结果写入文件系统，只把引用放进上下文。
