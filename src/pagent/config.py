"""统一配置：所有“换模型 / 换向量库后端”的地方都只改这里。

学习要点
--------
1. LangChain 的 `init_chat_model` 是一个工厂函数，用 "provider:model" 的方式
   屏蔽了各家 SDK 的差异，上层的 Agent 代码完全不用关心底层是哪家模型。
2. 国内大多数模型（DeepSeek、通义、Kimi、智谱、Ollama…）都提供 OpenAI 兼容接口，
   所以 provider 选 "openai" 再改 base_url 就能接入。
3. Embedding（向量模型）和对话模型是两回事，可以来自不同厂商。
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"  # RAG 的原始文档
RUNTIME_DIR = PROJECT_ROOT / ".runtime"  # 向量库、对话存档等运行时数据（已 gitignore）
WORKSPACE_DIR = PROJECT_ROOT / "workspace"  # Agent 允许写文件的沙箱目录

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")
    llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
    llm_api_key: str | None = os.getenv("LLM_API_KEY") or None
    llm_base_url: str | None = os.getenv("LLM_BASE_URL") or None

    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "fastembed")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    embedding_api_key: str | None = os.getenv("EMBEDDING_API_KEY") or None
    embedding_base_url: str | None = os.getenv("EMBEDDING_BASE_URL") or None


settings = Settings()


# --------------------------------------------------------------------------- #
# 对话模型
# --------------------------------------------------------------------------- #
def get_chat_model(temperature: float = 0.2) -> BaseChatModel:
    """返回一个支持 tool calling 的聊天模型。"""
    from langchain.chat_models import init_chat_model

    kwargs: dict = {"temperature": temperature}
    if settings.llm_api_key:
        kwargs["api_key"] = settings.llm_api_key
    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url
    return init_chat_model(settings.llm_model, model_provider=settings.llm_provider, **kwargs)


# --------------------------------------------------------------------------- #
# 向量模型
# --------------------------------------------------------------------------- #
class FastEmbedEmbeddings(Embeddings):
    """把 fastembed（本地 ONNX 推理）包装成 LangChain 的 Embeddings 接口。

    LangChain 的 Embeddings 只要求实现两个方法：
      - embed_documents: 批量把“文档块”变成向量（入库时用）
      - embed_query:     把“用户问题”变成向量（检索时用）
    有的模型（比如 bge、e5）对 query 和 document 用不同的前缀，所以接口分成两个。
    """

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(RUNTIME_DIR / "models"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


class HashingEmbeddings(Embeddings):
    """完全离线的“伪向量”：把字符二元组哈希到固定维度并归一化。

    它没有任何语义理解能力，只能做到“字面上相似的文本向量也相近”。
    用途：跑单元测试、没网时演示流程。真实使用请换 fastembed / openai。
    """

    def __init__(self, dims: int = 256):
        self.dims = dims

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        text = text.lower()
        grams = [text[i : i + 2] for i in range(max(len(text) - 1, 1))]
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            vec[h % self.dims] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    backend = settings.embedding_backend
    if backend == "fastembed":
        return FastEmbedEmbeddings(settings.embedding_model)
    if backend == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            # 很多兼容服务不支持 OpenAI 的 token 数组输入，关掉以直接发送字符串
            check_embedding_ctx_length=False,
        )
    if backend == "hash":
        return HashingEmbeddings()
    raise ValueError(f"未知的 EMBEDDING_BACKEND: {backend}")
