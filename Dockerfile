# 简历智能匹配系统 (T2-2) — 生产镜像
#
# 目标部署:腾讯云 x86_64 服务器。
# ⚠️ 在 Apple Silicon (M1/M2) 上构建必须强制 amd64 架构,否则产物无法在 x86 服务器运行:
#     docker buildx build --platform linux/amd64 -t resume-matcher .
#   (M1 上靠 QEMU 模拟 x86,构建较慢但产物可在腾讯云直接跑;compose 已设 platform。)
#
# 模型策略:构建阶段把 all-MiniLM-L6-v2 预下载并 save 到镜像内 /models,
#   运行时 embedder.py 通过 RESUME_MODEL_DIR 优先读该本地目录,做到零联网。

FROM python:3.11-slim

# --- 系统依赖 ---
# 直接使用 Debian 官方源,避免某些网络环境下第三方镜像返回 403。
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python 依赖 ---
# torch 在 Linux 默认轮子自带 CUDA(体积大),这里强制走 PyTorch 官方 CPU 源装 torch==2.12.0+cpu。
# 安装其余依赖时用 grep 跳过 requirements.txt 里的 torch 行,
# 避免 PyPI 默认轮子(带 CUDA)把已装好的 CPU 版覆盖掉。
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu \
    && grep -v '^torch==' requirements.txt > /tmp/requirements-no-torch.txt \
    && pip install -r /tmp/requirements-no-torch.txt

# --- 构建阶段预下载模型,烘进镜像 ---
# HF_ENDPOINT 设为构建参数:默认走官方 Hugging Face,避免 hf-mirror
# 在某些 huggingface_hub 版本上返回不到元数据而导致构建失败。
# 若部署环境只能走镜像站,可在构建时覆盖:
#   docker buildx build --platform linux/amd64 --build-arg HF_ENDPOINT=https://hf-mirror.com -t resume-matcher .
ARG HF_ENDPOINT=https://huggingface.co
ENV HF_ENDPOINT=${HF_ENDPOINT} \
    RESUME_MODEL_DIR=/models/all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2').save('/models/all-MiniLM-L6-v2')"

# --- 应用代码 ---
COPY src/ ./src/
COPY app.py ./

EXPOSE 8501

# Streamlit 健康检查端点。
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
