# 简历智能匹配系统 (T2-2)

上传**简历(PDF/TXT)**和**岗位 JD**,系统输出**匹配分数(0-100)** + **已匹配技能清单** + **缺失技能清单**。

- 课题:T2-2 简历智能匹配系统(Tier 2,难度系数 ×1.15)
- 核心技术:Sentence-BERT 句向量、余弦相似度、KeyBERT 关键词提取、pdfplumber 解析
- 界面:Streamlit
- 综合打分 = 语义相似度 60% + 关键词匹配 40%

---

## 功能

| 模块 | 文件 | 说明 |
|---|---|---|
| 文本解析 | `src/parser.py` | PDF(pdfplumber)/ TXT,输出纯文本 + 按句切分 |
| 句向量 | `src/embedder.py` | all-MiniLM-L6-v2,本地目录优先 + hf-mirror 兜底 |
| 匹配打分 | `src/matcher.py` | 整体余弦相似度 + 综合打分(60/40) |
| 技能匹配 | `src/skills.py` | KeyBERT 提取 JD 技能词 → 已匹配/缺失清单 |
| 界面 | `app.py` | Streamlit 上传 + 展示 |
| 对比实验 | `experiments/exp1_model_compare.py` | MiniLM vs mpnet 匹配分/耗时 |

---

## 本地运行(macOS / Linux)

> 实测环境:macOS(Apple Silicon)+ Python 3.13。以下命令规避了开发中踩到的坑(详见 CLAUDE.md 第8节)。

```bash
# 1) 建虚拟环境(macOS Homebrew Python 受 PEP 668 限制,必须用 venv)
python3 -m venv venv
source venv/bin/activate

# 2) 装依赖(国内源;若代理拦截先 unset 代理)
#    unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
#    若 pip 报 SSL 证书错误,临时加:
#      --trusted-host pypi.tuna.tsinghua.edu.cn --trusted-host pypi.org --trusted-host files.pythonhosted.org

# 3) 跑单测(纯逻辑,不下模型)
python -m pytest tests/ -v

# 4) 启动界面(首次会下载模型,设国内镜像兜底)
export HF_ENDPOINT=https://hf-mirror.com
streamlit run app.py --server.port=8501
# 浏览器打开 http://localhost:8501

# 5) 跑对比实验1(出表 + 图到 experiments/results/)
python -m experiments.exp1_model_compare
```

> 注:`torch` 在 Apple Silicon 上 pip 装的即原生 CPU/MPS 版,不会拉 CUDA。
> Python 3.13 需 torch ≥ 2.6.0(本项目固定 2.12.0)。

---

## Docker 部署

目标环境:**腾讯云 x86_64 服务器**。模型在**构建阶段烘进镜像**(`/models/all-MiniLM-L6-v2`),
运行时通过 `RESUME_MODEL_DIR` 优先读本地目录,**完全离线**,Demo 现场不依赖网络。

### 构建并启动

> 目录名含中文时 compose 可能报 `project name must not be empty`,命令统一带 `-p resume-matcher`(见 CLAUDE.md Bug #8)。

```bash
# 目标是 x86 服务器。在服务器本机(x86)直接:
docker compose -p resume-matcher up -d --build
```

**在 Apple Silicon (M1/M2) 上构建给 x86 服务器用**,必须强制 amd64 架构,否则产物跑不起来:

```bash
# 方式一:compose 已锁 platform: linux/amd64,直接:
docker compose -p resume-matcher up -d --build

# 方式二:单独 buildx(更显式)
docker buildx build --platform linux/amd64 -t resume-matcher:latest .
```

> M1 上构建 amd64 镜像靠 QEMU 模拟,**构建较慢**(预下载 torch + 模型,首次约 10~20 分钟);
> 若就在本机 arm64 演示,把 `docker-compose.yml` 里的 `platform: linux/amd64` 注释掉即可大幅提速。
>
> 实测(2026-06-06,M1 本机 arm64 构建):容器 `Up (healthy)`,`/_stcore/health` 返回 `ok`。

### 访问

构建启动后,浏览器打开:

```
http://<服务器IP>:8501
```

上传 `data/resumes/` 下的样例简历,JD 粘贴 `data/jds/` 下内容,即可出分。

### 构建期模型下载失败的兜底

已知坑(见 CLAUDE.md Bug #5):`hf-mirror` 对某些 `huggingface_hub` 版本不返回模型元数据,
会导致构建期 bake 模型这步失败。此时改用官方源重建:

```bash
docker buildx build --platform linux/amd64 \
  --build-arg HF_ENDPOINT=https://huggingface.co \
  -t resume-matcher:latest .
```

### 资源与体积参考

| 项目 | 量级 |
|---|---|
| 首次构建耗时 | x86 本机约 5~10 分钟;M1 模拟 amd64 约 10~20 分钟 |
| 镜像大小 | 内容约 **750 MB**(`docker image ls` CONTENT SIZE);Docker Desktop 含层/缓存磁盘占用约 **3.16 GB**(实测) |
| 运行内存 | 约 1~2 GB(compose 已设 `mem_limit: 2g` 防 OOM) |
| torch | **CPU 版**(Dockerfile 走 PyTorch 官方 CPU 源装 `torch==2.12.0+cpu`,不带 CUDA) |

### 验证「真离线」

证明模型确实烘进了镜像、运行时不联网。**推荐用 `--network none` 直接断网起容器**,
比断本机网更干净、不影响其它服务:

```bash
# 镜像已构建好(模型已烘入)。用 no-network 起一个临时容器:
docker run --rm --network none -p 8501:8501 resume-matcher:latest
# 浏览器开 http://localhost:8501,上传 data/resumes/resume_backend.txt
# + 粘贴 data/jds/jd_backend.txt → 仍能出分 = 容器全程零联网,离线成立 ✅
```

或用 compose 起好后断开网络再访问:

```bash
docker compose -p resume-matcher up -d --build          # 先构建好(此时已下完模型)
docker network disconnect bridge resume-matcher-app-1   # 断开容器网络(或直接断本机网)
# 再访问 http://localhost:8501 上传简历+JD,仍能正常出分 → 证明离线可用
```

### ⚠️ 安全提示(务必看)

本部署是**裸端口 8501、无鉴权、无反向代理**,仅适用于作业 Demo / 内网场景。
**公网部署前请自行加防火墙 / 反向代理(Nginx)/ 鉴权**,不要把无保护的端口直接暴露到公网。

---

## 测试数据说明

`data/` 下均为**脱敏假数据**(虚构姓名、虚构联系方式),仅供测试与 Demo,不含任何真实个人信息。

