"""Sentence-BERT 句向量编码。

模型加载策略(为第二步 Docker 离线烘模型做准备):
  1. 若设置了环境变量 RESUME_MODEL_DIR 且该目录存在 → 直接从本地目录加载(完全离线)。
  2. 否则按模型名加载;此时设置 HF_ENDPOINT=https://hf-mirror.com 作为国内兜底,
     并尊重 SENTENCE_TRANSFORMERS_HOME 指定的缓存目录。

Docker 构建时会把模型预下载到镜像内固定目录(如 /models/all-MiniLM-L6-v2),
并设 RESUME_MODEL_DIR 指向它,运行时零联网。
"""

from __future__ import annotations

import os
from functools import lru_cache

# 在任何 sentence_transformers / huggingface 导入触发联网前,先挂好国内镜像兜底。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def resolve_model_source(model_name: str = DEFAULT_MODEL) -> str:
    """决定从哪里加载模型:优先本地目录,其次模型名(走网络/缓存)。"""
    local_dir = os.environ.get("RESUME_MODEL_DIR")
    if local_dir and os.path.isdir(local_dir):
        return local_dir
    return model_name


@lru_cache(maxsize=4)
def load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """加载并缓存 SentenceTransformer 模型(同名只加载一次)。"""
    source = resolve_model_source(model_name)
    return SentenceTransformer(source)


class Embedder:
    """句向量编码器:句子列表 → 归一化句向量矩阵。"""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = load_model(model_name)

    def encode(self, sentences: list[str], *, batch_size: int = 32) -> np.ndarray:
        """编码句子列表为向量矩阵 (n, dim)。空列表返回空数组。

        向量做 L2 归一化,后续余弦相似度可直接用点积。
        """
        if not sentences:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        return self.model.encode(
            sentences,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
