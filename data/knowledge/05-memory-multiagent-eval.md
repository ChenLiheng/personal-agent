# 短期记忆

短期记忆就是当前会话的消息历史。LangGraph 通过 Checkpointer（InMemorySaver、SqliteSaver、PostgresSaver、RedisSaver）在每一步之后保存状态快照，用 thread_id 区分会话。历史过长时的处理办法：截断最早的消息、对旧消息做摘要（SummarizationMiddleware）、清理旧的工具结果（ContextEditingMiddleware），以及使用模型厂商的服务端上下文压缩功能。

# 长期记忆

长期记忆是跨会话保存的信息。可以按认知科学分为三类：语义记忆（事实，例如“用户住在杭州”）、情景记忆（经历，例如过去成功解决问题的完整过程，可作为 few-shot 示例）、程序性记忆（做事方式，例如不断改进的系统提示词）。LangGraph 用 Store（InMemoryStore、PostgresStore、SqliteStore）保存，按 namespace（如 ("users", user_id, "memories")）组织，并可开启向量索引做语义搜索。写入时机：热路径（模型调用 remember 工具，实时但增加延迟）或后台（对话结束后由另一个 LLM 抽取，不影响响应速度）。专门的记忆服务有 Mem0、Zep、Letta（MemGPT）、LangMem。

# 多 Agent 架构

1. Supervisor / Agents-as-tools：主 Agent 将子 Agent 作为工具调用，子 Agent 拥有独立上下文，只返回结论。
2. Router：先分类，再分派给对应专家。
3. Handoff / Swarm：Agent 之间相互转交控制权，例如客服转给退款专员。
4. Plan-and-Execute：规划器先输出计划，执行器逐步执行，可以重新规划。
5. 层级式（Hierarchical）：多层 Supervisor 管理多个团队。
Anthropic 的多 Agent 调研系统显示，多 Agent 并行调研能显著提升广度类任务的效果，但 token 消耗约为单 Agent 的数倍到十几倍。经验法则：能用单 Agent 解决就不要使用多 Agent；多 Agent 的价值在于上下文隔离与并行。

# 护栏与安全

常见护栏：输入检查（提示词注入检测、敏感信息脱敏）、输出检查（格式校验、内容审核）、Human-in-the-loop（高风险操作需人工批准）、最小权限（工具只开放必要的能力，文件操作限定目录）、调用次数与成本上限、沙箱执行代码。要特别警惕间接提示词注入：网页、邮件、文档中的文字可能伪装成指令，工具返回的内容应视为数据而不是命令。

# 可观测性

Agent 的执行路径不确定，必须能追踪每一步：每次模型调用的输入输出、token、延迟，每次工具调用的参数与结果。常用工具：LangSmith、Langfuse（开源，可自托管）、Arize Phoenix、OpenTelemetry GenAI 语义约定，以及 OpenAI Agents SDK 的内置 tracing。

# Agent 评估

评估层次：单步评估（工具选择是否正确、参数是否正确）、轨迹评估（调用顺序是否合理、有没有多余步骤）、最终结果评估（任务是否完成，可用规则、人工或 LLM-as-judge 判断）。方法：构建评测数据集，离线回归测试，线上抽样标注。常用基准：SWE-bench（代码）、τ-bench（客服工具使用）、GAIA（通用助手）、BrowseComp（浏览调研）、OSWorld（电脑操作）、Terminal-Bench（命令行任务）。

# 部署与生产

生产化需要考虑：持久化（Postgres Checkpointer）、长任务的异步执行与断点恢复、流式输出到前端、并发与限流、模型降级（ModelFallbackMiddleware）、提示词缓存降低成本、密钥管理。部署选项：LangGraph Platform / LangSmith Deployment、自建 FastAPI 服务、各云厂商的 Agent 托管平台（AWS Bedrock AgentCore、Google Vertex AI Agent Engine、Azure AI Foundry Agent Service、Claude Managed Agents）。
