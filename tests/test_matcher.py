"""matcher / skills / parser 的轻量单测,不依赖模型下载的部分尽量独立可跑。"""

from __future__ import annotations

import numpy as np

from src.matcher import combined_score, semantic_similarity
from src.parser import split_sentences
from src.skills import extract_skills, match_skills


def test_split_sentences_mixed():
    text = "我会 Python。也熟悉 Java!还有 SQL?"
    sents = split_sentences(text)
    assert len(sents) == 3
    assert "Python" in sents[0]


def test_split_sentences_filters_noise():
    text = "有效句子内容。\n-\n12\n"
    sents = split_sentences(text)
    assert sents == ["有效句子内容。"]


def test_combined_score_weights():
    # 语义=1.0, 关键词=0.0 → 60 分;反之 → 40 分。
    assert combined_score(1.0, 0.0) == 60.0
    assert combined_score(0.0, 1.0) == 40.0
    assert combined_score(1.0, 1.0) == 100.0


def test_rescale_semantic_baseline():
    from src.matcher import SEMANTIC_BASELINE, rescale_semantic
    # 基线以下夹到 0,满分仍是 1,基线处为 0,区分度被拉开。
    assert rescale_semantic(0.0) == 0.0
    assert rescale_semantic(SEMANTIC_BASELINE) == 0.0
    assert rescale_semantic(1.0) == 1.0
    mid = rescale_semantic((SEMANTIC_BASELINE + 1.0) / 2)
    assert 0.45 < mid < 0.55  # 基线与满分中点 → 约 0.5


def test_semantic_similarity_identical():
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    overall, mat = semantic_similarity(vecs, vecs)
    assert mat.shape == (2, 2)
    assert overall > 0.99


def test_semantic_similarity_empty():
    overall, mat = semantic_similarity(np.empty((0, 2)), np.array([[1.0, 0.0]]))
    assert overall == 0.0
    assert mat.shape == (0, 1)


def test_match_skills_hit_and_miss():
    resume = "熟练使用 Python 和 Docker,了解 Kubernetes。"
    skills = ["Python", "Docker", "Kafka", "机器学习"]
    res = match_skills(resume, skills)
    assert "Python" in res["matched"]
    assert "Docker" in res["matched"]
    assert "Kafka" in res["missing"]
    assert 0 < res["coverage"] < 1


def test_match_skills_java_not_in_javascript():
    # 词边界匹配:Java 的别名 "java" 不应命中 "javascript"。
    resume = "I write JavaScript every day."
    res = match_skills(resume, ["Java"])
    assert "Java" in res["missing"]


def test_extract_skills_chinese_jd_no_noise():
    # 回归 Bug #9:中文 JD 提取应是干净的规范技能名,不带中文噪声短语。
    jd = "熟练掌握 Java 或 Python,熟悉 Docker、Kubernetes 容器化技术,了解 Kafka。"
    skills = extract_skills(jd)
    assert "Java" in skills
    assert "Python" in skills
    assert "Docker" in skills
    assert "Kubernetes" in skills
    # 不应混入带中文动词的长短语。
    assert all("掌握" not in s and "熟悉" not in s for s in skills)


def test_extract_then_match_consistent():
    # 提取与匹配共用别名规则:JD 识别到的技能,简历含同名就应全部命中。
    jd = "需要 Python、Docker、Kubernetes 经验。"
    resume = "精通 Python,熟练 Docker 与 Kubernetes。"
    skills = extract_skills(jd)
    res = match_skills(resume, skills)
    assert res["coverage"] == 1.0
    assert not res["missing"]


def test_extract_skills_only_from_requirement_section():
    # 回归 Bug #14:职责段的描述性噪声(客户成功案例/配合销售团队)不应被当技能,
    # 只从「任职要求」段提取。
    jd = (
        "岗位职责:\n配合销售团队完成客户成功案例沉淀。\n"
        "任职要求:\n掌握 SQL,具备需求分析能力。"
    )
    skills = extract_skills(jd)
    assert "SQL" in skills
    assert "需求分析" in skills
    assert "客户成功" not in skills  # 职责段噪声,不应入选
    assert "销售" not in skills
