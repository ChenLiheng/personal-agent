# Function Calling 原理

Function Calling（也叫 Tool Use）是模型厂商在 API 层面提供的能力：请求中带上工具列表（名称、描述、参数的 JSON Schema），模型在需要时不直接回答，而是返回结构化的 tool_calls（工具名 + JSON 参数 + 调用 id）。模型本身并不执行任何函数，由应用程序执行，然后把结果连同 id 作为工具消息发回，模型再基于结果继续推理。一次回复中可以包含多个调用（并行工具调用）。

# OpenAI 与 Anthropic 格式差异

OpenAI Chat Completions：工具定义为 {"type":"function","function":{name, description, parameters}}；模型返回 message.tool_calls，arguments 是 JSON 字符串；结果用 role="tool" 并带 tool_call_id 的消息返回。Anthropic Messages：工具定义为 {name, description, input_schema}；模型返回 content 中 type="tool_use" 的 block，input 已是对象；结果放在 role="user" 消息中 type="tool_result" 的 block 里，并带 tool_use_id。LangChain 等框架用统一的 AIMessage.tool_calls 与 ToolMessage 屏蔽了这些差异。

# 好工具的设计原则

1. 名称与描述清晰：说明工具做什么、什么时候该用、什么时候不该用，描述是写给模型看的文档。
2. 参数尽量少且类型明确，用枚举限制取值，并为每个参数写说明。
3. 返回精简、对模型有用的信息，避免返回巨大的原始 JSON；必要时分页或截断。
4. 出错时返回可读的错误信息，让模型可以自我修正，而不是让程序崩溃。
5. 工具数量不宜过多（几十个以上会降低选择准确率），可以按需动态加载工具（tool search）。
6. 有副作用的工具（写入、付款、发送）需要人工审批或权限控制，并保证幂等。

# MCP 是什么

MCP（Model Context Protocol，模型上下文协议）是 Anthropic 在 2024 年 11 月发布的开放协议，现已捐赠给 Linux 基金会下的 Agentic AI Foundation，OpenAI、Google、微软等都已支持。它标准化了 AI 应用（Host/Client）与外部能力提供方（Server）之间的通信，基于 JSON-RPC。写一次 MCP Server，所有支持 MCP 的客户端（Claude Desktop、Claude Code、Cursor、VS Code、ChatGPT、自研 Agent）都能使用，被称为“AI 的 USB-C 接口”。

# MCP 的三种原语与传输方式

Server 可以暴露三类能力：Tools（模型可调用的函数，最常用）、Resources（以 URI 标识的只读数据，由应用决定何时读取）、Prompts（预置的提示词模板，一般由用户主动选择）。传输方式有两种：stdio（客户端把服务器作为本地子进程启动，适合本地工具）与 Streamable HTTP（服务器独立部署，可远程、多客户端共享，支持 OAuth 2.1 授权）。旧的 HTTP+SSE 传输已被弃用。

# MCP 2026-07-28 版本的变化

2026-07-28 版本把 MCP 改成无状态协议：去掉 initialize 握手与会话 ID，每个请求在 _meta 中携带协议版本和客户端能力；新增 server/discover 接口用于发现服务器能力；用多轮请求模式（MRTR，返回 input_required）替代服务器主动发起的 sampling、elicitation 请求；tasks（长任务）移到官方扩展中；Roots、Sampling、Logging 三个功能被标记为弃用。无状态化让 MCP 服务器更容易水平扩展与部署在 Serverless 环境。

# Function Calling、MCP、A2A 的关系

Function Calling 是“模型与应用”之间的接口：模型告诉应用要调什么。MCP 是“应用与工具服务”之间的接口：应用从哪里拿到工具、怎么调用工具。应用从 MCP Server 拿到工具列表后，仍然转换成 function calling 格式交给模型。A2A（Agent2Agent，Google 发起）则是“Agent 与 Agent”之间的协议，用于不同团队、不同框架开发的 Agent 互相委派任务。三者是互补关系，不是替代关系。

# Skills

Agent Skills 是 Anthropic 在 2025 年推出的能力打包方式：一个文件夹包含 SKILL.md（说明何时使用与如何执行）以及脚本、模板等资源。Agent 启动时只加载每个 Skill 的简短描述，需要时再读取完整内容（渐进式披露），从而节省上下文。Skills 适合封装“流程性知识”，MCP 适合封装“外部系统连接”，两者经常搭配使用。
