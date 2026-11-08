"""第 3 课：RAG 与向量数据库。

两种用法对比：
  A. 经典 RAG（固定流水线）：每次都先检索 Top-K，再把结果塞进 prompt 生成答案。
     简单、可控、延迟稳定；但不管问什么都会检索，检索一次不够也没办法。
  B. Agentic RAG（检索作为工具）：模型自己决定要不要查、查几次、换什么关键词。
     更灵活，适合复杂问题；代价是多几次模型调用。PAgent 主体用的是 B。

运行：uv run python -m pagent.lessons.l03_rag
"""

from __future__ import annotations

from pagent.config import get_chat_model, get_embeddings
from pagent.rag import build_index, get_vectorstore, load_documents


def show_chunks() -> None:
    docs = load_documents()
    print(f"知识库切成了 {len(docs)} 块，前 2 块示例：")
    for d in docs[:2]:
        print(f"  - {d.metadata} | {d.page_content[:60]!r}…")


def show_embedding() -> None:
    emb = get_embeddings()
    v = emb.embed_query("什么是 MCP？")
    print(f"\n一句话被编码成 {len(v)} 维向量，前 5 维：{[round(x, 4) for x in v[:5]]}")


def classic_rag(question: str) -> str:
    vs = get_vectorstore()
    hits = vs.similarity_search_with_relevance_scores(question, k=4)
    print(f"\n检索 “{question}” 的 Top-4：")
    for doc, score in hits:
        print(f"  {score:.3f}  {doc.metadata['source']}#{doc.metadata['section']}")
    context = "\n\n".join(d.page_content for d, _ in hits)
    prompt = f"""仅根据下面的资料回答问题；资料里没有就说不知道。

<资料>
{context}
</资料>

问题：{question}"""
    return get_chat_model().invoke(prompt).text


if __name__ == "__main__":
    show_chunks()
    show_embedding()
    print(f"\n建立索引：{build_index()} 块写入 Chroma")
    print("\n回答：", classic_rag("什么时候应该用向量数据库，什么时候不该用？"))
