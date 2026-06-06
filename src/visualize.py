"""可视化加分项:句×句相似度热力图 + 四维度匹配雷达图。

两者都只做「展示」,不影响核心打分;由 config 的 HEATMAP / RADAR flag 控制是否在 UI 渲染。
复用 matcher 已算好的数据(sim_matrix),不重复编码、不另拉模型。
"""

from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")  # 无显示环境也能出图。
import matplotlib.pyplot as plt
import numpy as np

# 中文显示:指定常见中文字体并修正负号。
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 联系方式/纯个人标识:无条件过滤(无论句子长短)。
_CONTACT_PAT = re.compile(r"(电话|邮箱|手机|微信|@|\d{6,})")
# 章节标题/字段名:仅当整句很短时过滤(避免误伤含这些词的正文)。
_SECTION_PAT = re.compile(
    r"(姓名|地址|所在地|个人信息|联系方式|求职意向"
    r"|教育背景|工作经历|项目经历|专业技能|自我评价|岗位职责|任职要求|招聘)"
)


def _is_noise(s: str) -> bool:
    """判断句子是否为噪声(联系方式/章节标题/过短/纯符号)。"""
    s = s.strip()
    if len(s) < 6:  # 过短的碎片(如「万,月活跃」)。
        return True
    if _CONTACT_PAT.search(s):  # 联系方式无条件滤掉。
        return True
    if _SECTION_PAT.search(s) and len(s) < 14:  # 短标题句。
        return True
    return bool(re.fullmatch(r"[\W_]+", s))


def _filter_indices(sentences: list[str]) -> list[int]:
    """返回过滤掉噪声后的句子下标(保序)。全被过滤则退回原始全部。"""
    keep = [i for i, s in enumerate(sentences) if not _is_noise(s)]
    return keep or list(range(len(sentences)))


def _wrap(label: str, width: int = 14, max_lines: int = 3) -> str:
    """把长句按宽度折行(而非省略号截断),保留更多信息且不溢出。"""
    s = label.strip()
    lines = [s[i:i + width] for i in range(0, len(s), width)]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:width - 1] + "…"
    return "\n".join(lines)


def _short_label(s: str, width: int = 22) -> str:
    """把 JD 要求句压成短标签:去掉常见前缀虚词,超长再截断。"""
    s = s.strip().lstrip("·-•　 ")
    s = re.sub(r"^(负责|具备|熟练使用|熟练掌握|掌握|熟悉|了解|能|有|优秀的|良好的|参与|制定)", "", s)
    s = s.strip("，。、 ")
    return s if len(s) <= width else s[:width] + "…"


def coverage_bar_figure(
    sim_matrix: np.ndarray,
    resume_sentences: list[str],
    jd_sentences: list[str],
    *,
    max_n: int = 8,
):
    """JD 每条要求 → 简历最佳匹配的相似度,横向条形图。返回 (Figure, 明细)。

    回答「JD 每条要求,简历满足得怎样」:对每条 JD 要求,在简历句里找最相似的一句,
    用其相似度作为「满足度」。比句×句矩阵更聚焦、更贴合简历匹配场景。

    Returns:
        (fig, details):details = [(JD要求, 最佳匹配简历句, 相似度0~1), ...] 按满足度降序
    """
    if sim_matrix is None or sim_matrix.size == 0:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, "无可用相似度数据", ha="center", va="center", color="#d0d4e0")
        ax.axis("off")
        return fig, []

    r_keep = _filter_indices(resume_sentences)
    j_keep = _filter_indices(jd_sentences)
    sub = sim_matrix[np.ix_(r_keep, j_keep)]  # (简历, JD)

    best_idx = sub.argmax(axis=0)
    best_val = sub.max(axis=0)
    order = np.argsort(-best_val)[:max_n]

    details = []
    for c in order:
        details.append((
            jd_sentences[j_keep[c]],
            resume_sentences[r_keep[best_idx[c]]],
            float(best_val[c]),
        ))

    labels = [_short_label(d[0]) for d in details]
    vals = [d[2] * 100 for d in details]

    fig, ax = plt.subplots(figsize=(9, max(3, len(labels) * 0.62)))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    fg = "#d0d4e0"

    y = np.arange(len(labels))[::-1]  # 满足度最高在最上。
    colors = [plt.cm.RdYlGn(v / 100) for v in vals]
    bars = ax.barh(y, vals, color=colors, edgecolor="#3a3f4c")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10, color=fg)
    ax.set_xlim(0, 100)
    ax.set_xlabel("简历对该要求的满足度(最佳匹配句相似度 ×100)", fontsize=10, color=fg)
    ax.set_title("JD 各项要求的简历满足度(越绿越满足)", fontsize=13, color=fg, pad=12)
    ax.tick_params(colors=fg)
    for sp in ax.spines.values():
        sp.set_color("#5a5f6c")
    for b, v in zip(bars, vals):
        ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.0f}",
                va="center", fontsize=9, color=fg)
    fig.tight_layout()
    return fig, details


# PLACEHOLDER_RADAR

# ---- 四维度雷达图 ----
# 每个维度用一句「维度描述」与简历做语义匹配。不再依赖简历句必须含某锚词
# (锚词法太脆,一个维度匹配不到就 0 分),改为语义相似度,任何维度都有合理分值。
RADAR_DIMENSIONS: dict[str, str] = {
    "技能": "专业技能、工具、编程语言、熟练掌握的技术能力",
    "经验": "工作经验、任职经历、负责的工作职责、从业年限",
    "学历": "教育背景、毕业院校、学历学位、所学专业",
    "项目": "项目经历、主导或参与的项目、产品、系统、业务成果",
}


def dimension_scores(
    resume_sentences: list[str],
    resume_vecs: np.ndarray,
    jd_doc_vec: np.ndarray,
    *,
    skill_coverage: float,
    embedder=None,
) -> dict[str, float]:
    """计算四维度匹配度(0~100),简单可解释。

    - 技能维:直接用关键词命中比例 coverage(与综合分口径一致,最硬的指标)。
    - 其余三维:用「维度描述句」与简历各句算余弦,取最高的若干句的均值。
      这样每个维度都有合理分值,不会因简历没用特定词而掉到 0。

    Args:
        resume_vecs: 简历句向量矩阵 (n, dim),已 L2 归一化。
        jd_doc_vec: JD 文档向量(保留参数,兼容旧调用)。
        skill_coverage: matcher 给出的关键词命中比例 0~1。
        embedder: 用于编码维度描述句;为空则懒加载默认模型。
    """
    scores: dict[str, float] = {"技能": round(skill_coverage * 100, 1)}

    other = {k: v for k, v in RADAR_DIMENSIONS.items() if k != "技能"}
    if resume_vecs is None or resume_vecs.shape[0] == 0:
        for k in other:
            scores[k] = 0.0
        return scores

    if embedder is None:
        from .embedder import Embedder
        embedder = Embedder()
    dim_vecs = embedder.encode(list(other.values()))  # (3, dim),已归一化。

    for (dim, _desc), dv in zip(other.items(), dim_vecs):
        sims = resume_vecs @ dv  # 与简历每句的余弦。
        k = min(3, len(sims))
        top = np.sort(sims)[-k:]  # 最相关的 k 句。
        score = float(np.mean(top))
        scores[dim] = round(max(0.0, min(1.0, score)) * 100, 1)
    return scores


def radar_figure(scores: dict[str, float]):
    """用 matplotlib 画四维度雷达图,返回 Figure。

    改 matplotlib(不依赖 plotly+kaleido,可服务端导出自测;配色适配深色主题):
    透明背景 + 浅色文字 + 满分基准虚线环 + 各维数值标注。
    """
    dims = list(scores.keys())
    vals = [scores[d] for d in dims]
    avg = round(sum(vals) / len(vals), 1) if vals else 0.0

    n = len(dims)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    vals_c = vals + vals[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_alpha(0)        # 透明背景,融入深色页面。
    ax.set_facecolor("none")
    fg = "#d0d4e0"                # 浅色前景,深底可读。

    # 满分基准环。
    ax.plot(angles, [100] * (n + 1), linestyle=":", color="#8a8f9c", linewidth=1.2)
    # 实际匹配度。
    ax.plot(angles, vals_c, color="#5b7cfa", linewidth=2.2, marker="o")
    ax.fill(angles, vals_c, color="#5b7cfa", alpha=0.28)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"{d}\n{scores[d]:.0f}" for d in dims], fontsize=12, color=fg)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8, color=fg)
    ax.tick_params(colors=fg)
    ax.spines["polar"].set_color("#5a5f6c")
    ax.grid(color="#5a5f6c", alpha=0.5)
    ax.set_title(f"四维度匹配雷达图　平均 {avg} / 100", fontsize=13, color=fg, pad=18)
    fig.tight_layout()
    return fig


def radar_interpretation(scores: dict[str, float]) -> str:
    """根据四维分生成一段文字解读(Markdown),给雷达图配说明。"""
    if not scores:
        return ""
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_dim, best_val = items[0]
    worst_dim, worst_val = items[-1]
    avg = sum(scores.values()) / len(scores)

    def level(v: float) -> str:
        return "强" if v >= 70 else ("中等" if v >= 50 else "偏弱")

    lines = [
        f"- **整体匹配**:平均 {avg:.0f}/100,综合处于「{level(avg)}」水平。",
        f"- **最强维度**:{best_dim}({best_val:.0f}),是该简历相对这个岗位最突出的部分。",
        f"- **最弱维度**:{worst_dim}({worst_val:.0f}),是相对短板,建议在简历中补充{worst_dim}相关的具体描述。",
    ]
    gap = best_val - worst_val
    if gap >= 25:
        lines.append(f"- **均衡性**:各维差距较大(最高与最低相差 {gap:.0f} 分),能力结构不够均衡。")
    else:
        lines.append(f"- **均衡性**:四个维度较为均衡(极差 {gap:.0f} 分),无明显短板。")
    return "\n".join(lines)


def rank_bar_figure(names: list[str], scores: list[float]):
    """批量排序的横向条形图:按分数渐变着色,深色主题友好,返回 Figure。"""
    fig, ax = plt.subplots(figsize=(8, max(2.5, len(names) * 0.6)))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    fg = "#d0d4e0"

    y = np.arange(len(names))[::-1]  # 第一名在最上。
    colors = [plt.cm.RdYlGn(s / 100) for s in scores]
    bars = ax.barh(y, scores, color=colors, edgecolor="#3a3f4c")

    ax.set_yticks(y)
    ax.set_yticklabels([_wrap(n, 16, 2) for n in names], fontsize=9, color=fg)
    ax.set_xlim(0, 100)
    ax.set_xlabel("综合匹配分", color=fg)
    ax.tick_params(colors=fg)
    for sp in ax.spines.values():
        sp.set_color("#5a5f6c")
    ax.set_title("候选简历综合分排名", fontsize=12, color=fg)
    for b, s in zip(bars, scores):
        ax.text(s + 1, b.get_y() + b.get_height() / 2, f"{s:.1f}",
                va="center", fontsize=9, color=fg)
    fig.tight_layout()
    return fig

