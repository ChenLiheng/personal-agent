# LangGraph

LangGraph 是 LangChain 团队开发的底层 Agent 编排框架，以“有状态的图”建模 Agent：State（状态）、Node（节点函数）、Edge（固定边/条件边）。核心卖点是持久化（Checkpointer，每一步保存快照）、Human-in-the-loop（interrupt 暂停与 Command 恢复）、流式输出、时间旅行调试和长时间运行（durable execution）。2025 年 10 月发布 1.0，承诺 API 稳定。适合需要精细控制流程的生产级 Agent。

# LangChain 1.x 与 create_agent

LangChain 1.0 起大幅精简，核心入口是 create_agent：一行创建一个基于 LangGraph 的 ReAct Agent，并通过 Middleware（中间件）定制行为，例如 SummarizationMiddleware（历史摘要）、HumanInTheLoopMiddleware（人工审批）、ToolRetryMiddleware（工具重试）、ModelCallLimitMiddleware（调用上限）、PIIMiddleware（敏感信息处理）、TodoListMiddleware（任务规划）。经验法则：先用 create_agent 加中间件；需要自定义复杂流程时，再下沉到 LangGraph 的 StateGraph。1.4 版本起内置 langchain.mcp.MCPAdapter 用于接入 MCP。

# OpenAI Agents SDK

OpenAI 官方的轻量 Agent 框架，概念很少：Agent、Tool、Handoff（Agent 之间转交）、Guardrail（护栏）、Session（会话记忆）、Tracing（内置追踪）。上手最快，对 OpenAI 模型和 Responses API 的支持最好，也可通过 LiteLLM 接入其他模型。

# Claude Agent SDK

Anthropic 把 Claude Code 背后的 Agent 运行时开放成 SDK（Python/TypeScript）。内置文件读写、Shell、搜索等工具，以及子 Agent、Skills、Hooks、MCP 接入、权限控制和上下文自动压缩。适合构建“会操作电脑/代码库”的通用型 Agent。

# 其他主流框架

- Google ADK（Agent Development Kit）：Google 的多 Agent 框架，与 Gemini、Vertex AI 集成紧密，支持 A2A 协议。
- CrewAI：以“角色 + 任务 + 团队（Crew）”的方式组织多 Agent，抽象层级高，适合快速做多角色协作原型。
- AutoGen / Microsoft Agent Framework：微软的多 Agent 对话框架，后续与 Semantic Kernel 合并为 Microsoft Agent Framework。
- LlamaIndex：最擅长数据接入与 RAG（各种 Loader、索引、查询引擎），也提供 Agent 与 Workflow。
- PydanticAI：类型安全、写法 Pythonic，结构化输出体验好。
- Dify / Coze / n8n：低代码、可视化编排平台，适合非开发者或者快速搭建业务流程。

# 框架选型建议

学习阶段：先用原生 SDK 手写一次工具循环，理解本质，再学 LangGraph。生产阶段：需要精细控制、持久化、人工审批时选 LangGraph；只用 OpenAI 模型、追求简单时选 OpenAI Agents SDK；需要代码/文件操作型通用 Agent 时选 Claude Agent SDK；以 RAG 为主时选 LlamaIndex；业务人员搭建流程时选 Dify/n8n。
