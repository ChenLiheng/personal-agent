# PAgent：一个用来学习 Agent 全套技术的个人助手

调研报告见 **[docs/RESEARCH.md](docs/RESEARCH.md)**，先读它，再对照代码学习。

## 这个 Agent 用到了哪些技术

| 技术 | 在哪里 |
|---|---|
| 手写 Function Calling 循环（OpenAI / Anthropic 两种格式） | `src/pagent/lessons/l01_raw_function_calling.py` |
| LangGraph：StateGraph、条件边、ToolNode、Checkpointer、`interrupt` | `src/pagent/lessons/l02_langgraph_from_scratch.py` |
| LangChain `create_agent` + 7 个 Middleware | `src/pagent/agent.py` |
| 自定义 Middleware（动态 prompt、工具调用日志） | `src/pagent/middleware.py` |
| `@tool` 本地工具（计算器、时间、抓网页、沙箱写文件） | `src/pagent/tools.py` |
| RAG + 向量数据库（Chroma + 本地 bge-small-zh） | `src/pagent/rag.py`、`lessons/l03_rag.py` |
| MCP Server（Tools / Resources / Prompts，stdio 与 HTTP） | `src/pagent/mcp_server.py` |
| MCP Client（协议层调用 + 转成 LangChain Tool） | `lessons/l04_mcp.py`、`agent.py::default_mcp_config` |
| 短期记忆（SQLite Checkpointer）+ 长期记忆（SQLite Store + 向量索引） | `src/pagent/memory.py` |
| Human-in-the-loop（批准 / 修改参数 / 拒绝） | `agent.py`、`cli.py` |
| 多 Agent：Agents-as-tools（调研员子 Agent）、Router | `agent.py::build_researcher`、`lessons/l06_multi_agent.py` |
| 结构化输出 | `lessons/l05_structured_output.py` |
| 流式输出 | `cli.py::stream_turn` |
| 护栏：PII 打码、调用上限、工具重试、写文件沙箱 | `agent.py`、`tools.py` |
| 离线测试（剧本模型） | `tests/` |
| 可观测性（LangSmith，可选） | `.env.example` |

## 快速开始

需要 [uv](https://docs.astral.sh/uv/)（会自动装好 Python 3.11 和所有依赖）。

```bash
uv sync
```

```bash
cp .env.example .env
```

编辑 `.env`，填入模型的 `LLM_PROVIDER`、`LLM_MODEL`、`LLM_API_KEY`（文件里有 Claude、DeepSeek、Ollama 的示例）。Embedding 默认用本地 fastembed，首次运行会自动下载约 100MB 的中文模型，不需要额外的 key。

先跑测试（不需要 API key，也不联网）：

```bash
uv run pytest -q
```

启动交互式 Agent：

```bash
uv run pagent
```

可以试试这些：

- `帮我算一下 2 的 32 次方减 1，再告诉我现在纽约几点`：并行调用工具
- `MCP 和 Function Calling 有什么区别？`：RAG 检索知识库
- `我主要用 Python，在杭州工作`，然后 `/new` 开新会话，问 `我在哪工作？`：长期记忆
- `加个待办：周五前看完 LangGraph 文档`，再问 `我还有哪些待办？`：MCP 工具
- `把我们刚才的讨论总结写到 notes/summary.md`：触发人工审批
- `帮我深入调研一下 RAG 的优化手段`：委派给调研员子 Agent

CLI 命令：`/new` 新会话、`/thread <id>` 切回旧会话、`/user <id>` 切换用户、`/memories` 查看长期记忆、`/reindex` 重建知识库索引、`/exit` 退出。

## 分课运行

每一课都是可以单独运行的脚本，建议按顺序学习：

```bash
uv run python -m pagent.lessons.l01_raw_function_calling
```

```bash
uv run python -m pagent.lessons.l02_langgraph_from_scratch
```

```bash
uv run python -m pagent.lessons.l03_rag
```

```bash
uv run python -m pagent.lessons.l04_mcp
```

```bash
uv run python -m pagent.lessons.l05_structured_output
```

```bash
uv run python -m pagent.lessons.l06_multi_agent
```

第 4 课不需要 LLM key。

## 扩展

- **知识库**：往 `data/knowledge/` 里放 `.md` / `.txt` 文件，然后在 CLI 里执行 `/reindex`。
- **接入更多 MCP Server**：在项目根目录创建 `mcp_servers.json`，格式与 Claude Desktop 相同：

  ```json
  {
    "mcpServers": {
      "fetch": { "command": "uvx", "args": ["mcp-server-fetch"] },
      "remote": { "url": "https://example.com/mcp" }
    }
  }
  ```
- **把 MCP Server 作为 HTTP 服务运行**：`uv run python -m pagent.mcp_server --http 8765`，然后用 `http://127.0.0.1:8765/mcp` 连接，也可以接到 Claude Desktop 或 Cursor 里。
- **看完整调用链**：在 `.env` 中打开 `LANGSMITH_TRACING`。

## 目录结构

```
├── docs/RESEARCH.md          调研报告
├── data/knowledge/           RAG 知识库（Agent 技术笔记）
├── src/pagent/
│   ├── config.py             模型 / Embedding 配置
│   ├── tools.py              本地工具
│   ├── rag.py                切块、向量库、检索工具
│   ├── memory.py             长期记忆工具
│   ├── mcp_server.py         MCP Server
│   ├── middleware.py         自定义中间件
│   ├── agent.py              组装完整 Agent
│   ├── cli.py                交互式命令行
│   └── lessons/              6 节分步课程
├── tests/                    离线测试
├── .runtime/                 运行时数据（向量库、SQLite、模型缓存），已被 gitignore
└── workspace/                Agent 唯一允许写文件的目录
```
