"""匹配打分:语义相似度 + 综合得分。

综合分 = 语义相似度 60% + 关键词匹配 40%(基础必做第5项)。

- 语义相似度:简历句向量的「文档向量」与 JD「文档向量」的余弦相似度。
  文档向量取句向量的均值(mean pooling),再算余弦。同时保留句×句相似度矩阵
  供热力图等可视化使用。
- 关键词匹配:由 skills.match_skills 给出命中比例 coverage。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .embedder import DEFAULT_MODEL, Embedder
from .skills import extract_skills, match_skills

SEMANTIC_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4

# 语义分重标定:Sentence-BERT 对「都是职场文本」的两段普遍给高余弦(常在 0.6~0.95),
# 尤其同属 IT 的不同岗位(前端/后端/数据)余弦能到 0.85+,直接当百分制会让错配也显得高。
# 故先减基线再拉伸:rescaled = (cos - BASELINE) / (1 - BASELINE),夹到 [0,1]。
# 基线取 0.65:把「近邻但不对口」(0.65~0.85)压到低区,对口(0.9+)仍保持高分,区分度最大。
SEMANTIC_BASELINE = 0.65


def rescale_semantic(cos: float) -> float:
    """把原始余弦相似度按基线重标定到 [0,1]。"""
    if cos <= SEMANTIC_BASELINE:
        return 0.0
    return (cos - SEMANTIC_BASELINE) / (1.0 - SEMANTIC_BASELINE)


@dataclass
class MatchResult:
    score: float                      # 综合分 0~100
    semantic_score: float             # 语义相似度 0~100
    keyword_score: float              # 关键词命中比例 0~100
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    sim_matrix: np.ndarray | None = None      # 简历句 × JD句 相似度矩阵(热力图用)
    resume_sentences: list[str] = field(default_factory=list)
    jd_sentences: list[str] = field(default_factory=list)
    resume_vecs: np.ndarray | None = None     # 简历句向量(雷达图复用,免重复编码)
    jd_doc_vec: np.ndarray | None = None      # JD 文档向量(雷达图复用)


def _doc_vector(sent_vecs: np.ndarray) -> np.ndarray:
    """句向量均值池化为单个文档向量(已是归一化句向量,均值后再归一化)。"""
    if sent_vecs.shape[0] == 0:
        return np.zeros((sent_vecs.shape[1],), dtype=np.float32)
    v = sent_vecs.mean(axis=0)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def semantic_similarity(resume_vecs: np.ndarray, jd_vecs: np.ndarray) -> tuple[float, np.ndarray]:
    """返回 (整体语义相似度 0~1, 句×句相似度矩阵)。"""
    if resume_vecs.shape[0] == 0 or jd_vecs.shape[0] == 0:
        return 0.0, np.zeros((resume_vecs.shape[0], jd_vecs.shape[0]))
    sim_matrix = cosine_similarity(resume_vecs, jd_vecs)
    rv = _doc_vector(resume_vecs)
    jv = _doc_vector(jd_vecs)
    overall = float(np.dot(rv, jv))
    overall = max(0.0, min(1.0, overall))  # 余弦可能略超界,夹到 [0,1]。
    # 重标定:压低职场文本普遍偏高的语义分,提升区分度。sim_matrix 保留原始余弦(可视化用真实值)。
    overall = rescale_semantic(overall)
    return overall, sim_matrix


def combined_score(semantic: float, coverage: float) -> float:
    """综合分:语义 60% + 关键词 40%,放大到 0~100。"""
    return (SEMANTIC_WEIGHT * semantic + KEYWORD_WEIGHT * coverage) * 100


def match(
    resume: dict,
    jd: dict,
    *,
    embedder: Embedder | None = None,
    top_n_skills: int = 15,
    model_name: str = DEFAULT_MODEL,
) -> MatchResult:
    """端到端匹配。

    Args:
        resume: parser.parse 的输出 {'text','sentences'}。
        jd: 同上(JD)。
        embedder: 复用已加载的 Embedder;为空则按 model_name 新建。
    """
    embedder = embedder or Embedder(model_name)

    r_sents = resume["sentences"]
    j_sents = jd["sentences"]
    r_vecs = embedder.encode(r_sents)
    j_vecs = embedder.encode(j_sents)

    semantic, sim_matrix = semantic_similarity(r_vecs, j_vecs)

    skills = extract_skills(jd["text"], top_n=top_n_skills, model_name=model_name)
    sk = match_skills(resume["text"], skills)

    score = combined_score(semantic, sk["coverage"])
    return MatchResult(
        score=round(score, 1),
        semantic_score=round(semantic * 100, 1),
        keyword_score=round(sk["coverage"] * 100, 1),
        matched_skills=sk["matched"],
        missing_skills=sk["missing"],
        sim_matrix=sim_matrix,
        resume_sentences=r_sents,
        jd_sentences=j_sents,
        resume_vecs=r_vecs,
        jd_doc_vec=_doc_vector(j_vecs),
    )
