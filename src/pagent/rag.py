"""RAG（检索增强生成）+ 向量数据库。

什么时候需要向量数据库？
------------------------
- 资料太多，塞不进上下文窗口（或者塞进去太贵/太慢）。
- 资料是“非结构化文本”（文档、手册、聊天记录），用户的问法和原文措辞不一样，
  关键词搜索搜不到，需要按“语义”找。
- 资料经常更新，不想重新训练/微调模型。
不需要的情况：数据是结构化的（订单、库存）-> 直接给 Agent 一个 SQL/API 工具更准；
资料只有几页 -> 直接放进 system prompt 就行。

RAG 的两条流水线
----------------
离线入库（indexing）:  加载文档 -> 切块(chunk) -> 向量化(embedding) -> 写入向量库
在线检索（retrieval）: 用户问题 -> 向量化 -> 相似度 Top-K -> 把片段塞进上下文 -> 模型作答

在 Agent 里，RAG 通常被做成一个工具（"Agentic RAG"）：由模型自己决定
要不要查、查什么关键词、查一次不够再查一次。
"""

from __future__ import annotations

from pathlib import Path

from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from pagent.config import KNOWLEDGE_DIR, RUNTIME_DIR, get_embeddings

COLLECTION = "knowledge"


# --------------------------------------------------------------------------- #
# 切块：这里手写一个“按 Markdown 标题 + 长度”切分的简单实现，便于理解。
# 生产中常用 langchain-text-splitters 的 RecursiveCharacterTextSplitter /
# MarkdownHeaderTextSplitter，原理相同。
# --------------------------------------------------------------------------- #
def split_markdown(text: str, source: str, chunk_size: int = 500, overlap: int = 80) -> list[Document]:
    sections: list[tuple[str, str]] = []
    title, buf = "", []
    for line in text.splitlines():
        if line.startswith("#"):
            if buf:
                sections.append((title, "\n".join(buf).strip()))
            title, buf = line.lstrip("# ").strip(), []
        else:
            buf.append(line)
    if buf:
        sections.append((title, "\n".join(buf).strip()))

    docs: list[Document] = []
    for title, body in sections:
        if not body:
            continue
        start = 0
        while start < len(body):
            piece = body[start : start + chunk_size]
            # 把标题拼进内容里：检索时能匹配到标题语义，答案也更有上下文
            content = f"{title}\n{piece}" if title else piece
            docs.append(Document(page_content=content, metadata={"source": source, "section": title}))
            if start + chunk_size >= len(body):
                break
            start += chunk_size - overlap  # 重叠一部分，避免一句话被切成两半后都检索不到
    return docs


def load_documents(folder: Path = KNOWLEDGE_DIR) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            docs.extend(split_markdown(path.read_text(encoding="utf-8"), source=path.name))
    return docs


# --------------------------------------------------------------------------- #
# 向量库
# --------------------------------------------------------------------------- #
def collection_name_for(embeddings: Embeddings) -> str:
    """不同 Embedding 模型产生的向量不能混用（维度、语义空间都不同），
    所以按模型给集合命名：换模型 = 自动用一个新集合，需要重新建索引。"""
    model = getattr(embeddings, "model_name", None) or getattr(embeddings, "model", None) or type(embeddings).__name__
    slug = "".join(ch if ch.isalnum() else "_" for ch in str(model)).strip("_").lower()
    return f"{COLLECTION}_{slug}"[:60]


def get_vectorstore(embeddings: Embeddings | None = None, persist_dir: Path | None = None) -> Chroma:
    """Chroma：嵌入式向量库，数据存在本地目录，零运维，适合学习和中小规模。
    规模大了可以换 Qdrant / Milvus / pgvector / Elasticsearch，LangChain 接口基本一致。"""
    embeddings = embeddings or get_embeddings()
    return Chroma(
        collection_name=collection_name_for(embeddings),
        embedding_function=embeddings,
        persist_directory=str(persist_dir or RUNTIME_DIR / "chroma"),
        collection_metadata={"hnsw:space": "cosine"},  # 用余弦相似度
    )


def build_index(vs: Chroma | None = None, folder: Path = KNOWLEDGE_DIR) -> int:
    """（重新）构建索引。用确定性的 id，重复执行不会产生重复数据。"""
    vs = vs or get_vectorstore()
    docs = load_documents(folder)
    if not docs:
        return 0
    ids = [f"{d.metadata['source']}::{i}" for i, d in enumerate(docs)]
    vs.reset_collection()  # 简单起见：全量重建。增量更新可以按 source 删除后再写入
    vs.add_documents(docs, ids=ids)
    return len(docs)


def ensure_index(vs: Chroma) -> None:
    if not vs.get(limit=1)["ids"]:
        build_index(vs)


def make_search_tool(vs: Chroma | None = None, k: int = 4):
    """把检索包装成一个工具，交给 Agent 自己决定何时调用。"""
    store = vs or get_vectorstore()
    ensure_index(store)

    @tool
    def search_knowledge_base(query: str) -> str:
        """在本地知识库中做语义检索，知识库内容是关于 AI Agent 开发的学习笔记
        （框架对比、Function Calling、MCP、RAG、记忆、多 Agent、评估等）。
        回答这类问题前应先检索；query 用简洁的关键短语，可以多次调用、换关键词再查。"""
        hits = store.similarity_search_with_relevance_scores(query, k=k)
        if not hits:
            return "知识库中没有找到相关内容。"
        lines = []
        for doc, score in hits:
            src = f"{doc.metadata.get('source')}#{doc.metadata.get('section', '')}"
            lines.append(f"[{src} | 相关度 {score:.2f}]\n{doc.page_content}")
        return "\n\n---\n\n".join(lines)

    return search_knowledge_base


if __name__ == "__main__":
    n = build_index()
    print(f"已写入 {n} 个文档块到 {RUNTIME_DIR / 'chroma'}")
