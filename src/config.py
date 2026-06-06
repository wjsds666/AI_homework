"""集中管理 feature flags 与运行时配置。

策略(见 CLAUDE.md 决策记录 2026-06-06):
  四个加分项全部实现并入库,但 UI 默认关闭,Demo 只跑稳定核心链路。
  每个 flag 可被环境变量覆盖,便于 Demo 时临时开启或在 app 内勾选。

环境变量(值为 1/true/yes/on 视为开启,大小写不敏感):
  ENABLE_HEATMAP            句×句相似度热力图
  ENABLE_RADAR             四维度匹配雷达图
  ENABLE_BATCH_RANK        批量简历排序
  ENABLE_INDUSTRY_CLASSIFY 行业分类

其它:
  TORCH_NUM_THREADS  限制 torch 线程数(默认 2,4G 内存服务器友好)
"""

from __future__ import annotations

import os

_TRUTHY = {"1", "true", "yes", "on"}


def _env_flag(name: str, default: bool = False) -> bool:
    """读取环境变量并解析为布尔。未设置时用 default。"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


# ---- Feature flags(默认全关,Demo 干净)----
HEATMAP = _env_flag("ENABLE_HEATMAP")
RADAR = _env_flag("ENABLE_RADAR")
BATCH_RANK = _env_flag("ENABLE_BATCH_RANK")
INDUSTRY_CLASSIFY = _env_flag("ENABLE_INDUSTRY_CLASSIFY")

# flag 名 → 中文展示名,供 app sidebar 渲染勾选框。
FEATURE_LABELS: dict[str, str] = {
    "HEATMAP": "JD要求满足度分析",
    "RADAR": "四维度匹配雷达图",
    "BATCH_RANK": "批量简历排序",
    "INDUSTRY_CLASSIFY": "行业分类",
}


def default_flags() -> dict[str, bool]:
    """返回当前(由环境变量决定的)各 flag 默认值,供 UI 初始化勾选框。"""
    return {
        "HEATMAP": HEATMAP,
        "RADAR": RADAR,
        "BATCH_RANK": BATCH_RANK,
        "INDUSTRY_CLASSIFY": INDUSTRY_CLASSIFY,
    }


def apply_torch_threads() -> None:
    """限制 torch 线程数,降低多核机器上的内存/CPU 抖动(4G 服务器友好)。

    需在加载模型前调用。torch 未安装时静默跳过。
    """
    try:
        import torch

        n = int(os.environ.get("TORCH_NUM_THREADS", "2"))
        torch.set_num_threads(max(1, n))
    except Exception:
        pass
