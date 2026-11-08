# 什么时候用向量数据库

适合使用向量数据库的情况：资料量大，无法全部放进上下文窗口；资料是非结构化文本（文档、手册、工单、聊天记录）；用户的问法与原文措辞不同，需要语义匹配；资料频繁更新，不想微调模型；需要给 Agent 做长期记忆的语义检索。不适合的情况：数据是结构化的（订单、库存、报表），应给 Agent 提供 SQL 或 API 工具；需要精确匹配（编号、代码符号），关键词检索或 grep 更合适；资料只有几页，直接放进提示词即可；需要全局统计或聚合（“一共有多少个”），向量检索做不到。

# RAG 流水线

离线入库：加载文档 -> 清洗 -> 切块（chunking）-> 用 Embedding 模型向量化 -> 写入向量库并保存元数据（来源、章节、时间、权限）。在线检索：问题向量化 -> 近似最近邻搜索 Top-K -> （可选）重排序 -> 把片段和来源拼进提示词 -> 模型基于资料作答并标注引用。

# 切块策略

块太大，检索不精确且浪费上下文；块太小，丢失上下文。常见做法：按文档结构（Markdown 标题、段落）切分，再按长度二次切分，每块约 300 到 800 个 token，相邻块重叠 10% 到 20%。把标题、文档名拼进块内容（contextual chunk headers）能显著提升召回。Anthropic 提出的 Contextual Retrieval 会用 LLM 为每个块生成一句上下文说明后再入库。

# 检索质量优化

1. 混合检索（Hybrid Search）：向量检索 + BM25 关键词检索，用 RRF 融合，兼顾语义与精确匹配。
2. 重排序（Rerank）：先召回 Top-50，再用 Cross-Encoder 重排模型（如 bge-reranker、Cohere Rerank）选出 Top-5。
3. 查询改写：多查询扩展、HyDE（先让模型生成假设答案再检索）、问题分解。
4. 元数据过滤：按时间、部门、权限过滤，防止越权访问。
5. Agentic RAG：把检索做成工具，由模型决定是否检索、检索几次、换什么关键词。
6. GraphRAG：抽取实体关系构建知识图谱，适合需要跨文档关联推理的问题。

# Embedding 模型选择

中文常用：BAAI bge 系列（bge-small-zh、bge-m3）、阿里 gte / text-embedding-v4、jina-embeddings-v3、Qwen3-Embedding；英文与多语言：OpenAI text-embedding-3、Cohere embed、Voyage。注意：入库与查询必须使用同一个模型；更换模型需要重建索引；查看 MTEB / C-MTEB 排行榜辅助选型。本地运行可用 fastembed、sentence-transformers 或 Ollama。

# 向量数据库选型

- Chroma：嵌入式，pip 安装即用，适合学习与原型。
- FAISS：Meta 的向量检索库，不是数据库，适合离线批量检索。
- pgvector：PostgreSQL 扩展，已有 PG 的团队首选，事务与 SQL 过滤能力强。
- Qdrant：Rust 实现，性能好，过滤能力强，支持本地与云。
- Milvus / Zilliz：面向十亿级大规模，分布式，国内使用广泛。
- Weaviate、Pinecone（全托管）、Elasticsearch / OpenSearch（已有搜索基础设施时，天然支持混合检索）。
- LanceDB、sqlite-vec：嵌入式、文件级，适合端侧与轻量应用。

# RAG 评估

检索层指标：召回率（Recall@K）、MRR、命中率。生成层指标：忠实度（答案是否都来自资料）、答案相关性、上下文精确度。常用工具：RAGAS、DeepEval、LangSmith 评估、TruLens。先建一个 50 到 200 条的问答评测集，每次修改切块、模型、提示词后都跑一遍。
