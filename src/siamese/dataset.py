"""孪生网络数据集(骨架,暂不训练)。

数据形态:(句子A, 句子B, label),label=1 相似 / 0 不相似。
来源设想:
  - 正样本:同一岗位的 简历句 ↔ JD句、或语义改写对。
  - 负样本:不同岗位的句子随机配对。

⚠️ 骨架:接口已定,具体构造在 Colab 训练时实现。
"""

from __future__ import annotations


class SiamesePairDataset:
    """句子对数据集(骨架)。

    TODO(Colab):
        - 继承 torch.utils.data.Dataset。
        - __init__(pairs): pairs 为 [(sentA, sentB, label), ...]。
        - __getitem__: 返回两句的句向量(用 all-MiniLM-L6-v2 预编码)与 label。
        - __len__。
    """

    def __init__(self, pairs: list[tuple[str, str, int]] | None = None):
        self.pairs = pairs or []

    def __len__(self) -> int:
        return len(self.pairs)


def build_pairs_from_data(resume_dir: str, jd_dir: str) -> list[tuple[str, str, int]]:
    """从 data/ 构造训练句对(骨架)。

    TODO(Colab):读取简历与 JD,按「同岗位=正、跨岗位=负」生成句对。
    """
    raise NotImplementedError("句对构造在 Colab 训练时实现。")
