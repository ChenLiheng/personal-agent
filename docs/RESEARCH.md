# AI Agent 技术调研报告（2026 年 11 月）

> 面向：想系统学习 Agent 开发的工程师
> 配套代码：本仓库 `src/pagent/`，每一节末尾标注了“对应代码”。
> 版本信息于 2026-11-07 在 PyPI、MCP 官方规范、LangChain 官方文档上核对过（见文末“资料来源与核实说明”）。

---

## 0. 一页结论

1. **Agent = LLM + 工具 + 循环。** 所有框架的内核都是同一个 while 循环：模型返回 `tool_calls`，程序执行，结果喂回去，直到模型不再要工具。先手写一遍这个循环，再学框架。
2. **框架格局已经收敛。** Python 生态里 **LangGraph（底层编排）+ LangChain 1.x `create_agent`（上层封装 + Middleware）** 是事实标准；厂商 SDK（OpenAI Agents SDK、Claude Agent SDK、Google ADK）各自绑定自家模型和生态；CrewAI、LlamaIndex、PydanticAI、Microsoft Agent Framework 各有所长。
3. **工具接入的三层协议：** Function Calling（模型 ↔ 应用）、**MCP**（应用 ↔ 工具服务，已成为行业标准，最新规范 2026-07-28 改成了无状态协议）、A2A（Agent ↔ Agent）。三者是互补关系。
4. **向量数据库只在“大量非结构化文本 + 语义检索”时才需要。** 结构化数据用 SQL/API 工具，精确匹配用关键词检索/grep，资料少就直接放进 prompt。在 Agent 里，检索通常做成工具（Agentic RAG）。
5. **当前的核心竞争力是“上下文工程”**：每一步给模型看什么。对应的技术手段有记忆分层、历史摘要、子 Agent 隔离上下文、按需加载工具或 Skills。
6. **生产化的关键在模型之外**：持久化/断点续跑、Human-in-the-loop、护栏、可观测性、评测集。

---

## 1. Agent 是什么，什么时候需要它

| 概念 | 说明 | 例子 |
|---|---|---|
| 单次 LLM 调用 | 输入 → 输出，无工具 | 翻译、摘要 |
| **Workflow（工作流）** | 步骤由代码预先编排，LLM 只在固定节点上被调用 | 先分类，再按类别走不同的 prompt |
| **Agent（智能体）** | LLM 在循环中**自主决定**下一步、调用什么工具、何时结束 | “帮我调研 X 并写报告” |

Anthropic 在《Building Effective Agents》中给出的建议现在仍然成立：**能用 Workflow 就别用 Agent**。常见的 Workflow 模式有 Prompt Chaining、Routing、Parallelization、Orchestrator-Workers、Evaluator-Optimizer。只有当步骤数量和顺序无法事先确定时，才值得付出 Agent 的成本（延迟、token、不可预测性）。

**ReAct 循环**（Reason + Act）是 Agent 的基本形态：思考 → 调用工具 → 观察 → 再思考。现代模型原生支持 tool calling，不再需要用 prompt 去解析 `Action:` 这类文本。

### Agent 的八个组件

```
┌──────────────────────────────────────────────────────────────┐
│ 8. 可观测 & 评估   LangSmith / Langfuse / OpenTelemetry / 评测集 │
│ 7. 护栏           HITL 审批 · 权限 · 调用上限 · PII · 注入防护   │
│ 6. 编排           循环 · 分支 · 并行 · 多 Agent  (LangGraph)    │
│ 5. 知识           RAG · 向量数据库 · 混合检索 · 重排            │
│ 4. 记忆           短期(Checkpointer) · 长期(Store)              │
│ 3. 工具           @tool 函数 · API · MCP Server · 子 Agent      │
│ 2. 指令           System Prompt · Skills                      │
│ 1. 模型           支持 tool calling 的 LLM                     │
└──────────────────────────────────────────────────────────────┘
```

对应代码：`data/knowledge/01-agent-basics.md`、`src/pagent/agent.py` 顶部架构图。

---

## 2. 框架全景与选型

### 2.1 主流框架对比（版本为 2026-11-07 PyPI 最新版）

| 框架 | 版本 | 定位 | 核心抽象 | 适合场景 |
|---|---|---|---|---|
| **LangGraph** | 1.2.20 | 底层有状态编排 | State / Node / Edge / Checkpointer / interrupt | 生产级、需要精细控制流程、持久化、人工审批 |
| **LangChain** | 1.4.9 | 上层 Agent 封装 + 集成 | `create_agent` + Middleware；`init_chat_model`；`langchain.mcp` | 快速搭建标准 Agent，再用中间件定制 |
| OpenAI Agents SDK | 0.24.0 | 轻量级、官方 | Agent / Tool / Handoff / Guardrail / Session / Tracing | 主要使用 OpenAI 模型，追求简单 |
| Claude Agent SDK | 0.2.179 | Claude Code 的运行时 | 内置文件/Shell/搜索工具、子 Agent、Skills、Hooks、MCP、权限 | 会操作代码库或电脑的通用 Agent |
| Google ADK | 2.14.0 | Google 多 Agent 框架 | Agent / Workflow Agent / A2A | Gemini、Vertex AI 生态 |
| CrewAI | 1.15.37 | 高层多角色协作 | Agent（角色）/ Task / Crew / Flow | 多角色协作原型 |
| LlamaIndex | 0.14.25 | 数据与 RAG 优先 | Loader / Index / QueryEngine / Workflow | 以知识库问答为主 |
| PydanticAI | 2.66.0 | 类型安全 | Agent[Deps, Output] | 重视类型与结构化输出 |
| AutoGen → Microsoft Agent Framework | 0.7.5（AutoGen） | 微软多 Agent | 对话式多 Agent | 微软 / Azure 生态 |
| Dify / Coze / n8n | — | 低代码可视化平台 | 拖拽式流程 | 业务人员、快速上线 |

### 2.2 LangChain 1.x 和 LangGraph 的关系（容易混淆）

- **LangGraph** 是“引擎”：只管图怎么执行、状态怎么保存、怎么暂停和恢复。
- **LangChain 1.x** 是“整车”：`create_agent()` 返回的就是一个编译好的 LangGraph 图（`model ⇄ tools` 循环），再通过 **Middleware** 在 `before_model / wrap_model_call / after_model / wrap_tool_call` 等位置挂钩子。
- 内置 Middleware（1.4.9 实测）：`SummarizationMiddleware`、`HumanInTheLoopMiddleware`、`ToolRetryMiddleware`、`ModelRetryMiddleware`、`ModelFallbackMiddleware`、`ModelCallLimitMiddleware`、`ToolCallLimitMiddleware`、`PIIMiddleware`、`TodoListMiddleware`、`LLMToolSelectorMiddleware`、`ContextEditingMiddleware`、`ShellToolMiddleware`、`FilesystemFileSearchMiddleware` 等。
- 1.4 起新增 `langchain.mcp.MCPAdapter`（基于 FastMCP，目前是 beta），取代原来的独立包 `langchain-mcp-adapters`。

**建议路径：** 先用 `create_agent` 加中间件；需要非标准流程（路由、并行、自定义审批节点、多 Agent 编排）时，再下沉到 `StateGraph`。

### 2.3 选型决策

```
只是调一次模型？ ─────────────────────────────► 直接用厂商 SDK
步骤固定？ ───────────────────────────────────► Workflow（LangGraph StateGraph / 普通代码）
需要 Agent：
  ├─ 只用 OpenAI 模型，追求简单 ───────────────► OpenAI Agents SDK
  ├─ 要操作文件、代码、Shell ───────────────────► Claude Agent SDK
  ├─ 要持久化、人工审批、复杂流程、换模型 ──────► LangGraph + LangChain   ← 本项目
  ├─ 以知识库问答为主 ───────────────────────────► LlamaIndex（或 LangChain + 向量库）
  └─ 非开发者搭流程 ─────────────────────────────► Dify / n8n
```

对应代码：`src/pagent/lessons/l02_langgraph_from_scratch.py`（手搭图）与 `src/pagent/agent.py`（`create_agent` + 7 个中间件）。

---

## 3. Function Calling（工具调用）

### 3.1 原理

1. 请求时带上工具列表：**名称 + 描述 + 参数 JSON Schema**。
2. 模型认为需要工具时，不直接回答，而是返回结构化的 `tool_calls`（工具名、参数、调用 id），一次可以返回多个（并行调用）。
3. **模型本身不执行任何代码**，由应用程序执行，并把结果带上 id 发回。
4. 模型读取结果后继续推理，直到给出最终回答。

### 3.2 两大协议格式对比

| | OpenAI Chat Completions（国产模型、Ollama、vLLM 基本都兼容） | Anthropic Messages |
|---|---|---|
| 工具定义 | `{"type":"function","function":{name, description, parameters}}` | `{name, description, input_schema}` |
| 模型请求 | `message.tool_calls[i].function.arguments`（**JSON 字符串**） | `content` 中 `type:"tool_use"` 的 block，`input` 已是对象 |
| 结果回传 | `{"role":"tool","tool_call_id":..., "content":...}` | `role:"user"` 消息里 `type:"tool_result"` 的 block，带 `tool_use_id` |
| 结束信号 | `finish_reason == "tool_calls"` | `stop_reason == "tool_use"` |

LangChain 用统一的 `AIMessage.tool_calls` 和 `ToolMessage` 屏蔽了这些差异，`init_chat_model("provider:model")` 一行就能切换厂商。

### 3.3 好工具的设计原则

- **描述是写给模型看的文档**：写清楚做什么、什么时候用、什么时候不该用。
- 参数少、类型明确，能用枚举就用枚举。
- 返回精简、有用的信息，不要直接扔回巨大的原始 JSON。
- 出错时**返回可读的错误信息，不要抛异常**，让模型可以自我纠正。
- 工具数量多了（几十个以上）选择准确率会下降，可以用 `LLMToolSelectorMiddleware` 或服务端 tool search 按需加载。
- 有副作用的工具（写入、付款、发送）要配合 HITL 审批，并保证幂等。

对应代码：`src/pagent/lessons/l01_raw_function_calling.py`（不用框架，**两种格式都实现了一遍**）、`src/pagent/tools.py`（`@tool` 由签名和 docstring 自动生成 schema）。

---

## 4. MCP（Model Context Protocol）

### 4.1 是什么

MCP 是 Anthropic 在 2024-11 发布的开放协议（JSON-RPC），后来捐赠给 Linux 基金会旗下的 Agentic AI Foundation 管理，OpenAI、Google、微软等主流厂商都已支持。它把“AI 应用（Host/Client）如何发现和调用外部能力（Server）”标准化了：**写一次 Server，Claude Desktop、Claude Code、Cursor、VS Code、ChatGPT 和你自己的 Agent 都能直接用。**

### 4.2 三种原语与两种传输

| 原语 | 谁来触发 | 用途 |
|---|---|---|
| **Tools** | 模型 | 可调用的函数（最常用），支持 `structuredContent` 结构化输出 |
| **Resources** | 应用 | 以 URI 标识的只读数据（文件、记录） |
| **Prompts** | 用户 | 预置的提示词模板（常做成斜杠命令） |

| 传输方式 | 说明 |
|---|---|
| **stdio** | Client 把 Server 作为子进程启动，本地工具首选（本项目的 todo server 就是这种） |
| **Streamable HTTP** | Server 独立部署，可远程、多客户端共享，用 OAuth 2.1 授权 |
| ~~HTTP+SSE~~ | 已弃用 |

### 4.3 最新规范 2026-07-28 的重要变化

当前（current）规范版本是 **2026-07-28**，相对 2025-11-25 版最大的变化是**无状态化**：

- **去掉 `initialize` 握手和 `Mcp-Session-Id`**：每个请求都在 `_meta` 里携带协议版本和客户端能力；新增必须实现的 `server/discover` 接口。
- **MRTR（多轮请求）模式**：服务器不再主动给客户端发请求（sampling、elicitation、roots），而是返回 `resultType: "input_required"`，客户端补充信息后重试原请求。
- `subscriptions/listen` 取代 GET 端点和 `resources/subscribe`。
- **tasks（长任务）移到官方扩展** `io.modelcontextprotocol/tasks`。
- **标记为弃用**：Roots、Sampling、Logging，HTTP+SSE 传输，以及 Dynamic Client Registration（改用 Client ID Metadata Documents）。
- `tools/list` 等结果加入 `ttlMs`/`cacheScope` 缓存提示；工具应按确定的顺序返回，以提高提示词缓存命中率。

对开发者的影响：**写 Server 的方式基本不变**（FastMCP 的 `@mcp.tool` 仍是主流写法），但更容易部署到 Serverless 或做水平扩展；新代码不要再依赖 sampling/roots。

### 4.4 Function Calling、MCP、A2A、Skills 的关系

```
            Function Calling                MCP                       A2A
  [ LLM ] ◀──────────────▶ [ 你的 Agent ] ◀────────▶ [ 工具服务 ]     [ Agent ] ◀──▶ [ 别人的 Agent ]
  模型告诉应用“要调什么”       应用从哪拿工具、怎么调用           Agent 之间互相委派任务
```

- 应用从 MCP Server 拿到工具列表后，**仍然转换成 function calling 格式交给模型**。
- **Agent Skills**（Anthropic，2025）：一个文件夹，包含 `SKILL.md` 和脚本、模板。启动时只加载简短描述，需要时再读全文（渐进式披露），用来封装“流程性知识”。MCP 封装“系统连接”，两者常搭配使用。

对应代码：`src/pagent/mcp_server.py`（FastMCP Server：3 个 Tool、1 个 Resource、1 个 Prompt）、`src/pagent/lessons/l04_mcp.py`（协议层逐个调用 `tools/list`、`tools/call`、`resources/read`、`prompts/get`）、`src/pagent/agent.py::default_mcp_config`（与 Claude Desktop 相同的 `mcpServers` 配置格式，可以放一个 `mcp_servers.json` 接入更多 Server）。

---

## 5. RAG 与向量数据库

### 5.1 什么时候用，什么时候别用

| ✅ 适合向量检索 | ❌ 不适合，应该改用 |
|---|---|
| 资料量大，塞不进上下文 | 只有几页 → 直接放进 system prompt |
| 非结构化文本（文档、手册、工单、聊天记录） | 结构化数据（订单、库存）→ SQL/API 工具 |
| 问法和原文措辞不同，需要语义匹配 | 精确匹配（编号、函数名）→ BM25 / grep |
| 资料常更新，不想微调 | 统计聚合（“一共几个”）→ SQL |
| Agent 长期记忆的语义召回 | |

> 趋势：编程类 Agent（Claude Code 等）大多**放弃了代码向量索引**，改用 grep/glob 这类“Agent 自主搜索”，因为代码需要精确匹配，而且变化快。向量检索在**文档类知识库**中仍是主流。

### 5.2 两条流水线

```
离线入库: 加载 → 清洗 → 切块(chunk) → Embedding → 写入向量库(+元数据: 来源/章节/时间/权限)
在线检索: 问题 → Embedding → ANN Top-K → (重排 Rerank) → 拼进 prompt → 带引用作答
```

### 5.3 两种使用方式

- **经典 RAG**：每次固定先检索再生成。简单、延迟稳定。
- **Agentic RAG**：检索作为工具，模型决定查不查、查几次、换什么关键词。更灵活，**本项目主体采用这种方式**。

### 5.4 提升效果的手段（按性价比排序）

1. **切块**：按文档结构（标题/段落）切，再按长度二次切分（300–800 token，重叠 10–20%）；**把标题拼进块内容**。
2. **混合检索**：向量 + BM25，用 RRF 融合。
3. **重排序**：先召回 Top-50，再用 Cross-Encoder（bge-reranker、Cohere Rerank）精选 Top-5。
4. **查询改写**：多查询扩展、HyDE、问题分解。
5. **元数据过滤**：时间、部门、权限（防止越权）。
6. **Contextual Retrieval**（Anthropic）：入库前用 LLM 给每个块生成一句上下文说明。
7. **GraphRAG**：需要跨文档关联推理时使用。

### 5.5 Embedding 模型与向量库选型

**Embedding（中文）：** BAAI bge 系列（bge-small-zh、bge-m3）、Qwen3-Embedding、阿里 text-embedding-v4、jina-embeddings。参考 C-MTEB 排行榜。**入库和查询必须用同一个模型，换模型就要重建索引**（本项目按模型名给集合命名，避免混用）。

| 向量库 | 类型 | 何时选 |
|---|---|---|
| **Chroma** | 嵌入式 | 学习、原型（本项目使用） |
| sqlite-vec / LanceDB | 嵌入式、文件级 | 端侧、轻量应用（本项目的长期记忆 Store 底层就是 sqlite-vec） |
| FAISS | 检索库（非数据库） | 离线批量检索 |
| **pgvector** | PostgreSQL 扩展 | 已在用 PG；需要事务和 SQL 过滤 |
| **Qdrant** | 独立服务 | 性能与过滤能力兼顾，中等规模首选 |
| **Milvus** / Zilliz | 分布式 | 亿级以上数据，国内使用广泛 |
| Elasticsearch / OpenSearch | 搜索引擎 | 已有搜索基础设施，天然支持混合检索 |
| Pinecone / Weaviate | 托管 / 独立服务 | 不想自己运维 |

### 5.6 评估

检索层看 Recall@K、MRR；生成层看忠实度、答案相关性。工具有 RAGAS、DeepEval、LangSmith Evaluation。**先建一个 50–200 条的评测集**，每次修改都跑一遍。

对应代码：`src/pagent/rag.py`（手写切块、Chroma、检索工具）、`src/pagent/config.py::FastEmbedEmbeddings`（本地 bge-small-zh，512 维）、`src/pagent/lessons/l03_rag.py`（经典 RAG 对比）。

---

## 6. 记忆

| | 短期记忆 | 长期记忆 |
|---|---|---|
| 内容 | 当前会话的消息历史 | 跨会话的用户事实、偏好、经验 |
| LangGraph 实现 | **Checkpointer**（InMemory / SQLite / Postgres / Redis），按 `thread_id` 区分 | **Store**（InMemory / SQLite / Postgres），按 namespace 组织，可开启向量索引 |
| 附带能力 | 断点续跑、HITL 暂停恢复、时间旅行调试 | 语义召回 |
| 太长怎么办 | 截断 / 摘要（`SummarizationMiddleware`）/ 清理旧工具结果（`ContextEditingMiddleware`） | 去重、合并、过期淘汰（TTL） |

长期记忆的三种类型：**语义记忆**（事实）、**情景记忆**（过往经历，可作为 few-shot 示例）、**程序性记忆**（做事方式，例如不断改进的 prompt）。
写入时机：**热路径**（模型调用 `remember` 工具，实时）或 **后台**（对话结束后由另一个 LLM 抽取，不影响延迟）。
专门的记忆服务：Mem0、Zep、Letta（MemGPT）、LangMem。

对应代码：`src/pagent/memory.py`（remember / recall / forget 工具，`ToolRuntime` 读取 store 和用户上下文）、`src/pagent/middleware.py::memory_aware_prompt`（每轮自动检索记忆并注入 system prompt）、`AsyncSqliteSaver` + `AsyncSqliteStore`（重启后记忆仍在，有测试覆盖）。

---

## 7. 上下文工程（Context Engineering）

模型每一步“看到什么”决定了它做得好不好。上下文窗口再大，塞得越多注意力越分散，成本也越高。常用手段：

- **写入（Write）**：把中间结果写到文件或 Store，上下文里只放引用。
- **选择（Select）**：按需检索知识、记忆，按需加载工具（tool search）或 Skills。
- **压缩（Compress）**：摘要旧消息、清理旧的工具结果、截断工具输出。
- **隔离（Isolate）**：用子 Agent 处理子任务，只把结论返回主 Agent。
- **缓存（Cache）**：保持 system prompt 和工具列表稳定，以命中提示词缓存（成本最多可降约 90%）。

对应代码：`agent.py` 中的 `SummarizationMiddleware`、`ask_researcher` 子 Agent、工具输出截断（`fetch_url` 的 `max_chars`）。

---

## 8. 多 Agent

| 模式 | 说明 | 本项目 |
|---|---|---|
| **Agents-as-tools / Supervisor** | 主 Agent 把子 Agent 当工具调用；子 Agent 上下文独立，只返回结论 | `agent.py::ask_researcher` |
| **Router** | 先分类，再分派给专家 | `lessons/l06_multi_agent.py` |
| Handoff / Swarm | Agent 之间转交控制权（OpenAI Agents SDK 的核心概念） | — |
| Plan-and-Execute | 先规划再逐步执行，可以重新规划 | `TodoListMiddleware` 可实现 |
| Hierarchical | 多层 Supervisor | — |

**经验法则：能用单 Agent + 好工具解决的，就不要上多 Agent。** 多 Agent 的价值在于上下文隔离与并行（Anthropic 的多 Agent 调研系统在广度类任务上效果显著提升，但 token 消耗是单 Agent 的数倍）。

---

## 9. 护栏、安全与 Human-in-the-loop

- **HITL**：高风险操作先暂停，等人批准、修改参数或拒绝。在 LangGraph 中底层是 `interrupt()` + `Command(resume=...)`，依赖 Checkpointer 保存暂停时的状态。
- **最小权限**：文件操作限定目录（本项目 `write_file` 只能写 `workspace/`）、只读的数据库账号、工具白名单。
- **调用上限**：`ModelCallLimitMiddleware`、`ToolCallLimitMiddleware`，防止死循环和成本失控。
- **输入/输出检查**：PII 脱敏（`PIIMiddleware`）、格式校验（结构化输出）、内容审核。
- **间接提示词注入**：网页、邮件、文档中可能藏着“指令”。**工具返回的内容是数据，不是命令**；高风险动作必须由人确认。
- **沙箱执行代码**：Docker、E2B、Modal 等。

对应代码：`agent.py`（HITL、PII、调用上限、工具重试）、`lessons/l02_langgraph_from_scratch.py`（手写 `interrupt` 审批节点）、`cli.py::ask_decisions`（approve / edit / reject 三种决策）。

---

## 10. 结构化输出

- `model.with_structured_output(PydanticModel)`：单次抽取。
- `create_agent(response_format=PydanticModel)`：Agent 先跑完工具循环，最后按 schema 输出到 `result["structured_response"]`。
- 实现策略：Tool 策略（把 schema 伪装成工具，兼容性最好）或 Provider 原生 JSON Schema 约束解码（更可靠）。

对应代码：`lessons/l05_structured_output.py`。

---

## 11. 可观测性与评估

**可观测性**：Agent 的执行路径不确定，必须能追踪每次模型调用（输入、输出、token、延迟）和每次工具调用。工具有 **LangSmith**（与 LangGraph 无缝集成，设置 `LANGSMITH_TRACING=true` 即可）、**Langfuse**（开源，可自托管）、Arize Phoenix、OpenTelemetry GenAI 语义约定。

**评估的三个层次**：
1. 单步：工具选对了吗？参数对吗？
2. 轨迹：调用顺序合理吗？有没有多余步骤？
3. 结果：任务完成了吗？（规则判断、LLM-as-judge 或人工）

公开基准：SWE-bench（代码）、τ-bench（客服工具调用）、GAIA（通用助手）、BrowseComp（浏览调研）、OSWorld（电脑操作）、Terminal-Bench（命令行）。

**测试技巧**：用“剧本模型”固定模型的决策，只测编排和工具逻辑，不花钱，也不受模型随机性影响。见 `tests/conftest.py::ScriptedChatModel`。

---

## 12. 部署与生产化清单

- [ ] 持久化：Postgres Checkpointer/Store（替换本项目的 SQLite）
- [ ] 长任务：异步执行 + 断点恢复（LangGraph durable execution）
- [ ] 流式输出到前端（`astream` 的 messages/updates 模式，或 LangChain 1.3+ 的 event streaming）
- [ ] 模型降级与重试（`ModelFallbackMiddleware`、`ModelRetryMiddleware`）
- [ ] 提示词缓存、成本监控、限流
- [ ] 密钥管理（不要写进代码或 URL）
- [ ] 评测集 + CI 回归
- 部署平台：LangSmith Deployment（原 LangGraph Platform）、自建 FastAPI、AWS Bedrock AgentCore、Google Vertex AI Agent Engine、Azure AI Foundry Agent Service、Claude Managed Agents。

---

## 13. 学习路线（对应本仓库）

| 阶段 | 学什么 | 文件 |
|---|---|---|
| 第 1 天 | 手写工具循环，理解 Function Calling 本质与两种协议格式 | `lessons/l01_raw_function_calling.py` |
| 第 2 天 | LangGraph：State/Node/Edge、条件边、ToolNode、Checkpointer、interrupt | `lessons/l02_langgraph_from_scratch.py` |
| 第 3 天 | RAG：切块、Embedding、Chroma、经典 RAG 与 Agentic RAG 对比 | `rag.py`、`lessons/l03_rag.py` |
| 第 4 天 | MCP：写 Server、协议层调用、接入 Agent | `mcp_server.py`、`lessons/l04_mcp.py` |
| 第 5 天 | 结构化输出、多 Agent（Router / Agents-as-tools） | `lessons/l05_*`、`lessons/l06_*` |
| 第 6 天 | 组装完整 Agent：create_agent + 7 个 Middleware + 记忆 + MCP + 子 Agent | `agent.py`、`middleware.py`、`memory.py` |
| 第 7 天 | 测试与可观测：剧本模型测试、LangSmith 追踪 | `tests/`、`.env.example` |

**进阶方向**：把 Chroma 换成 Qdrant/pgvector 并加入混合检索与重排；把 todo server 改成 Streamable HTTP 远程部署；接入现成的 MCP Server（GitHub、浏览器、数据库）；接入 LangSmith 并建立评测集；尝试 Claude Agent SDK 或 OpenAI Agents SDK，实现同样的功能做对比。

---

## 资料来源与核实说明

本次调研时，所在环境的联网搜索工具不可用，因此采用以下方式核实：

- **框架版本**：2026-11-07 直接查询 PyPI JSON API（`pypi.org/pypi/<包名>/json`）。
- **MCP 规范**：[Versioning](https://modelcontextprotocol.io/specification/versioning)、[2026-07-28 Changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)。
- **LangChain/LangGraph API**：官方文档 [Agents](https://docs.langchain.com/oss/python/langchain/agents)、[Built-in middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)、[MCP](https://docs.langchain.com/oss/python/langchain/mcp)、[Human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)、[Streaming](https://docs.langchain.com/oss/python/langchain/streaming)、[Long-term memory](https://docs.langchain.com/oss/python/langchain/long-term-memory)；并在安装好的包上用 `inspect.signature` 核对过所有用到的 API。
- **概念性内容**（Workflow 与 Agent 的取舍、上下文工程、多 Agent 经验、RAG 优化手段、框架定位）基于截至 2026 年中的公开资料（Anthropic《Building Effective Agents》、多 Agent 调研系统工程博客、Contextual Retrieval，LangChain 上下文工程系列等）。**这部分没有逐条联网核对最新进展**，用于学习足够，做技术决策前建议再查看各框架的最新官方文档。
