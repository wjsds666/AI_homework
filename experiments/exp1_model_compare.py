"""对比实验 1:嵌入模型对比 (all-MiniLM-L6-v2 vs all-mpnet-base-v2)。

在同一批(简历, JD)配对上,比较两个 Sentence-BERT 模型的:
  - 匹配分(综合分)
  - 编码 + 匹配耗时

输出:
  - experiments/results/exp1_model_compare.csv   (明细表)
  - experiments/results/exp1_score_compare.png   (匹配分对比柱状图)
  - experiments/results/exp1_time_compare.png    (耗时对比柱状图)

运行:
    python -m experiments.exp1_model_compare
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # 无显示环境也能出图。
import matplotlib.pyplot as plt

from src.embedder import Embedder
from src.matcher import match
from src.parser import parse

# 中文显示(matplotlib 默认字体不含中文,这里指定常见中文字体并修正负号)。
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "experiments" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

MODELS = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]

# 测试配对:(简历文件, JD文件, 关系标签)。
PAIRS = [
    ("resume_backend.txt", "jd_backend.txt", "后端→后端(匹配)"),
    ("resume_data_analyst.txt", "jd_data_analyst.txt", "数据→数据(匹配)"),
    ("resume_frontend.txt", "jd_backend.txt", "前端→后端(不匹配)"),
    ("resume_data_analyst.txt", "jd_backend.txt", "数据→后端(不匹配)"),
]


def run() -> pd.DataFrame:
    rows = []
    for model_name in MODELS:
        print(f"\n=== 加载模型 {model_name} ===")
        embedder = Embedder(model_name)
        for resume_f, jd_f, label in PAIRS:
            resume = parse(DATA / "resumes" / resume_f)
            jd = parse(DATA / "jds" / jd_f)
            t0 = time.perf_counter()
            res = match(resume, jd, embedder=embedder, model_name=model_name)
            elapsed = time.perf_counter() - t0
            rows.append({
                "模型": model_name,
                "配对": label,
                "综合分": res.score,
                "语义分": res.semantic_score,
                "关键词分": res.keyword_score,
                "耗时(秒)": round(elapsed, 3),
            })
            print(f"  {label:18} 综合分={res.score:5}  耗时={elapsed:.3f}s")

    df = pd.DataFrame(rows)
    csv_path = RESULTS / "exp1_model_compare.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n明细表已存:{csv_path}")
    return df


def plot_score(df: pd.DataFrame) -> None:
    """分组柱状图:每个配对下两模型的综合分。"""
    pivot = df.pivot(index="配对", columns="模型", values="综合分")
    ax = pivot.plot(kind="bar", figsize=(10, 5), rot=15)
    ax.set_ylabel("综合匹配分")
    ax.set_title("图1:不同嵌入模型在各配对上的匹配分对比")
    ax.legend(title="模型")
    plt.tight_layout()
    out = RESULTS / "exp1_score_compare.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"匹配分对比图已存:{out}")


def plot_time(df: pd.DataFrame) -> None:
    """柱状图:两模型平均单次匹配耗时。"""
    avg = df.groupby("模型")["耗时(秒)"].mean()
    ax = avg.plot(kind="bar", figsize=(6, 5), rot=0, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("平均耗时(秒)")
    ax.set_title("图2:不同嵌入模型平均单次匹配耗时")
    for i, v in enumerate(avg.values):
        ax.text(i, v, f"{v:.3f}s", ha="center", va="bottom")
    plt.tight_layout()
    out = RESULTS / "exp1_time_compare.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"耗时对比图已存:{out}")


if __name__ == "__main__":
    df = run()
    plot_score(df)
    plot_time(df)
    print("\n=== 实验 1 完成 ===")
    print(df.to_string(index=False))
