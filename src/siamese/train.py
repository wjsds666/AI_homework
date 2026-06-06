"""孪生网络训练脚本(骨架,在 Colab/Kaggle GPU 上跑,本机不执行)。

预期流程(Colab):
  1. 设 HF_ENDPOINT=hf-mirror,加载 all-MiniLM-L6-v2 预编码句对。
  2. 构造 SiamesePairDataset + DataLoader。
  3. SiameseHead + 对比损失,Adam 优化,训练若干 epoch。
  4. 保存投影头权重,回填本项目做「自训练 Siamese vs 预训练」对比(第3组实验)。

⚠️ 骨架:main 留 TODO,不在本机训练(无 GPU)。今天仅占位。
"""

from __future__ import annotations


def train(epochs: int = 10, lr: float = 1e-3, batch_size: int = 32) -> None:
    """训练入口(骨架)。

    TODO(Colab):实现上面注释的完整训练循环并保存权重。
    """
    raise NotImplementedError(
        "Siamese 训练需 GPU,在 Colab/Kaggle 执行;本机仅保留骨架,今天不训练。"
    )


if __name__ == "__main__":
    print("Siamese 训练骨架:请在 Colab/Kaggle GPU 环境实现 train() 后运行。")
