"""孪生网络模型定义(骨架,暂不训练)。

孪生网络原理:两路**共享权重**的编码器分别把句子 A、句子 B 映射成向量,
再用距离/相似度(如余弦)度量二者语义接近程度。训练目标是让语义相似的对距离近、
不相似的对距离远(对比损失 / 三元组损失)。

本项目用法:以预训练 Sentence-BERT 句向量为输入,接一个小投影头微调,
学习「简历句 ↔ JD 句」是否匹配,作为对预训练模型的对比基线(第3组实验)。

⚠️ 骨架:结构已设计,forward/损失留 TODO,训练在 Colab 完成,本机不跑。
"""

from __future__ import annotations

# NOTE: 训练时再 import torch;骨架阶段不强依赖,避免本机无 GPU 误跑。
# import torch
# import torch.nn as nn


class SiameseHead:
    """孪生投影头(骨架)。

    设计:
        input_dim: 输入句向量维度(all-MiniLM-L6-v2 为 384)。
        proj_dim:  投影后维度。
        结构:Linear(input_dim -> proj_dim) -> ReLU -> Linear(proj_dim -> proj_dim)。
        两路共享同一投影头(孪生 = 权重共享)。

    TODO(Colab 训练时实现):
        - 继承 nn.Module,在 __init__ 里建上述层。
        - forward_once(x): 单路前向,返回投影向量。
        - forward(a, b): 两路前向,返回 (va, vb)。
        - 相似度:F.cosine_similarity(va, vb)。
    """

    def __init__(self, input_dim: int = 384, proj_dim: int = 128):
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        # TODO: 定义 nn 层(Colab)。

    def forward_once(self, x):  # noqa: ANN001
        """单路前向(骨架)。TODO: Linear->ReLU->Linear。"""
        raise NotImplementedError("Siamese 训练在 Colab 完成,本机骨架不实现 forward。")

    def forward(self, a, b):  # noqa: ANN001
        """两路共享权重前向(骨架)。"""
        raise NotImplementedError("Siamese 训练在 Colab 完成,本机骨架不实现 forward。")


def contrastive_loss(dist, label, margin: float = 1.0):  # noqa: ANN001
    """对比损失(骨架)。

    label=1 表示相似对(拉近),label=0 表示不相似对(推远到 margin 外)。
    公式:L = y * d^2 + (1-y) * max(0, margin - d)^2。
    TODO: Colab 用 torch 实现。
    """
    raise NotImplementedError("Siamese 训练在 Colab 完成,本机骨架不实现损失。")
