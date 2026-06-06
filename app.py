"""简历智能匹配系统 — Streamlit 界面 (T2-2)。

运行:
    streamlit run app.py --server.port=8501

功能(基础必做):
    上传简历(PDF/TXT)+ 粘贴或上传 JD → 输出匹配分数 + 已匹配技能 + 缺失技能。

本文件仅负责界面与交互层(布局/样式/文案/状态);评分、技能、可视化逻辑全在 src/。
"""

from __future__ import annotations

import streamlit as st

from src import config
from src.classifier import classify_industry
from src.embedder import DEFAULT_MODEL, Embedder
from src.matcher import match
from src.parser import parse, split_sentences
from src.visualize import (
    coverage_bar_figure,
    dimension_scores,
    radar_figure,
    radar_interpretation,
    rank_bar_figure,
)

config.apply_torch_threads()  # 限制 torch 线程(4G 内存友好),需在加载模型前。

st.set_page_config(
    page_title="AI 简历匹配分析系统",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============ 设计 token + 全局样式 ============
st.markdown(
    """
    <style>
      :root {
        --primary:#5b7cfa; --primary-2:#7c5cff;
        --ok:#2ecc71; --warn:#e67e22; --danger:#e74c3c;
        --card:rgba(255,255,255,.045); --card-2:rgba(255,255,255,.07);
        --bd:rgba(255,255,255,.09); --muted:#9aa3b2;
        --r:14px; --shadow:0 6px 22px rgba(0,0,0,.28);
      }
      /* 主背景:深色 + 极轻径向渐变,有质感不花哨 */
      .stApp {
        background:
          radial-gradient(1200px 600px at 85% -5%, rgba(124,92,255,.10), transparent 60%),
          radial-gradient(900px 500px at -5% 110%, rgba(91,124,250,.08), transparent 55%),
          #0e1117;
      }
      #MainMenu, footer, header [data-testid="stToolbar"] {visibility:hidden;}
      .block-container {padding-top:2.2rem; max-width:1180px;}

      /* 侧栏缩窄、弱化 */
      [data-testid="stSidebar"] {min-width:262px; max-width:262px; background:#10131b;}
      [data-testid="stSidebar"] .block-container {padding-top:1.5rem;}

      /* 顶部品牌条 */
      .brand {display:flex; align-items:center; gap:12px; margin-bottom:4px;}
      .brand .logo {
        width:42px; height:42px; border-radius:12px; flex:none;
        background:linear-gradient(135deg,var(--primary),var(--primary-2));
        display:flex; align-items:center; justify-content:center; font-size:22px;
        box-shadow:0 4px 14px rgba(91,124,250,.4);
      }
      .brand h1 {margin:0; font-size:23px; font-weight:750; letter-spacing:.3px;}
      .brand .sub {color:var(--muted); font-size:13px; margin-top:1px;}

      /* 通用卡片 */
      .card {
        background:var(--card); border:1px solid var(--bd); border-radius:var(--r);
        padding:18px 20px; box-shadow:var(--shadow);
      }
      .step-tag {
        display:inline-block; font-size:12px; font-weight:600; color:var(--primary);
        background:rgba(91,124,250,.12); border:1px solid rgba(91,124,250,.3);
        border-radius:999px; padding:3px 12px; margin-bottom:10px; letter-spacing:.4px;
      }

      /* 结果摘要 Hero */
      .hero {
        background:linear-gradient(135deg, rgba(91,124,250,.16), rgba(124,92,255,.10));
        border:1px solid var(--bd); border-radius:18px; padding:22px 26px;
        display:flex; align-items:center; justify-content:space-between; gap:24px;
        box-shadow:var(--shadow);
      }
      .hero .meta {flex:1; min-width:0;}
      .hero .io {color:var(--muted); font-size:13px; margin-bottom:6px;}
      .hero .io b {color:#cdd5e3; font-weight:600;}
      .hero .verdict {font-size:21px; font-weight:750; line-height:1.35;}
      .hero .ring {
        flex:none; width:128px; height:128px; border-radius:50%;
        display:flex; flex-direction:column; align-items:center; justify-content:center;
        color:#fff; box-shadow:0 8px 24px rgba(0,0,0,.32);
      }
      .hero .ring .n {font-size:40px; font-weight:850; line-height:1;}
      .hero .ring .u {font-size:12px; opacity:.9; margin-top:2px;}

      /* 指标卡 */
      .metric {
        background:var(--card); border:1px solid var(--bd); border-radius:var(--r);
        padding:16px 18px; text-align:center;
      }
      .metric .v {font-size:30px; font-weight:800; line-height:1;}
      .metric .l {font-size:13px; color:var(--muted); margin-top:6px;}
      .metric .bar {height:6px; border-radius:6px; background:rgba(255,255,255,.08); margin-top:12px; overflow:hidden;}
      .metric .bar > span {display:block; height:100%; border-radius:6px;}

      /* 技能卡 */
      .skill-box {
        background:var(--card); border:1px solid var(--bd); border-radius:var(--r);
        padding:16px 18px; height:100%;
      }
      .skill-head {display:flex; align-items:center; gap:8px; font-weight:700; font-size:15px; margin-bottom:12px;}
      .skill-head .badge {
        font-size:12px; font-weight:700; padding:2px 10px; border-radius:999px;
      }
      .badge-ok {background:rgba(46,204,113,.16); color:#5fe39b; border:1px solid rgba(46,204,113,.35);}
      .badge-miss {background:rgba(231,76,60,.16); color:#ff8b7e; border:1px solid rgba(231,76,60,.35);}
      .chip {display:inline-block; padding:6px 13px; margin:4px 6px 4px 0; border-radius:10px; font-size:13.5px; font-weight:500;}
      .chip-ok {background:rgba(46,204,113,.13); color:#7ef0ad; border:1px solid rgba(46,204,113,.28);}
      .chip-miss {background:rgba(231,76,60,.12); color:#ff9d92; border:1px solid rgba(231,76,60,.28);}
      .chip-core {box-shadow:0 0 0 1px rgba(255,255,255,.12) inset; font-weight:650;}

      /* 总结条 */
      .summary {
        background:var(--card-2); border-left:3px solid var(--primary); border-radius:10px;
        padding:14px 18px; font-size:15px; line-height:1.7; color:#d7dce8;
      }
      .sec {font-size:18px; font-weight:750; margin:6px 0 2px;}
      .sec-sub {color:var(--muted); font-size:13px; margin-bottom:8px;}
      h3 {letter-spacing:.3px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="正在加载语义模型(首次较慢)...")
def get_embedder(model_name: str) -> Embedder:
    """缓存 Embedder,避免每次交互都重新加载模型。"""
    return Embedder(model_name)


def read_upload(uploaded) -> dict:
    """把 Streamlit 上传对象解析成 {'text','sentences'}。"""
    return parse(uploaded.getvalue(), filename=uploaded.name)


def score_color(score: float) -> str:
    """按分数高低返回主色:高分绿、中分橙、低分红。"""
    if score >= 75:
        return "#2ecc71"
    if score >= 50:
        return "#e6a020"
    return "#e74c3c"


def score_gradient(score: float) -> str:
    """分数环背景渐变。"""
    if score >= 75:
        return "linear-gradient(135deg,#27c08a,#1aa06f)"
    if score >= 50:
        return "linear-gradient(135deg,#f0a93b,#d98324)"
    return "linear-gradient(135deg,#e74c3c,#c0392b)"


def verdict_text(score: float) -> str:
    """综合分 → 一句话结论(纯展示,不改评分)。"""
    if score >= 80:
        return "高度匹配:简历与岗位高度契合,推荐进入下一轮"
    if score >= 65:
        return "较为匹配:整体契合度较好,建议结合细节进一步评估"
    if score >= 45:
        return "中度匹配:部分契合,但存在明显能力缺口"
    return "匹配度较低:简历与岗位方向差异较大"


# 硬技能(具体工具/语言)优先于软技能展示,视觉更聚焦。
_SOFT_SKILLS = {"沟通协调", "团队协作", "抗压能力", "学习能力", "执行力", "责任心", "逻辑思维"}


def sort_skills(skills: list[str]) -> list[str]:
    """技能排序:硬技能在前、软技能在后(顺序内保持原相对次序)。"""
    hard = [s for s in skills if s not in _SOFT_SKILLS]
    soft = [s for s in skills if s in _SOFT_SKILLS]
    return hard + soft


def render_chips(items: list[str], kind: str) -> str:
    """技能列表 → 彩色 chip HTML。硬技能加高亮边。"""
    cls = "chip-ok" if kind == "ok" else "chip-miss"
    if not items:
        return "<span style='color:var(--muted);'>—</span>"
    out = []
    for it in sort_skills(items):
        core = " chip-core" if it not in _SOFT_SKILLS else ""
        out.append(f"<span class='chip {cls}{core}'>{it}</span>")
    return "".join(out)


def summary_text(result) -> str:
    """根据结果生成一句自然语言总结(纯展示)。"""
    lvl = ("高度" if result.score >= 80 else "较好" if result.score >= 65
           else "中度" if result.score >= 45 else "较低")
    sem = "语义贴合度高" if result.semantic_score >= 65 else (
        "语义贴合度中等" if result.semantic_score >= 40 else "语义方向差异较大")
    parts = [f"该简历与岗位整体呈**{lvl}匹配**(综合 {result.score}/100),{sem}"]
    miss = sort_skills(result.missing_skills)
    if miss:
        gap = "、".join(miss[:4])
        more = f" 等 {len(miss)} 项" if len(miss) > 4 else ""
        parts.append(f"在 **{gap}**{more}方面仍有缺口")
    else:
        parts.append("JD 要求的技能基本已覆盖")
    return ",".join(parts) + "。"


# PLACEHOLDER_BODY

# ---- 顶部品牌条 ----
st.markdown(
    """
    <div class="brand">
      <div class="logo">🎯</div>
      <div>
        <h1>AI 简历匹配分析系统</h1>
        <div class="sub">Sentence-BERT 语义匹配 · 技能词典命中 · 综合评分 = 语义 60% + 关键词 40%</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("#### ⚙️ 分析设置")
    model_name = st.selectbox(
        "嵌入模型",
        [DEFAULT_MODEL, "all-mpnet-base-v2"],
        index=0,
        help="默认 all-MiniLM-L6-v2(轻量、CPU 友好)。",
    )
    top_n = st.slider("技能识别上限", 10, 50, 30,
                      help="从 JD 任职要求中识别的技能数量上限。")
    st.markdown("---")

    # 高级功能:默认折叠,文案精简。
    defaults = config.default_flags()
    flags: dict[str, bool] = {}
    with st.expander("深度分析模块", expanded=any(defaults.values())):
        for key, label in config.FEATURE_LABELS.items():
            flags[key] = st.checkbox(label, value=defaults[key], key=f"flag_{key}")

    st.markdown("---")
    st.caption("🔒 本地运行,数据不出本机。")

st.write("")

# 批量排序(加分项)开启时,提供模式切换;否则只走单份匹配。
mode = "单份匹配"
if flags.get("BATCH_RANK"):
    mode = st.radio("分析模式", ["单份匹配", "批量简历排序"], horizontal=True)

if mode == "单份匹配":
    st.markdown("<span class='step-tag'>STEP 1 · 输入</span>", unsafe_allow_html=True)
    col_resume, col_jd = st.columns(2, gap="large")
    with col_resume:
        st.markdown("<div class='sec'>📄 候选人简历</div>", unsafe_allow_html=True)
        st.markdown("<div class='sec-sub'>支持 PDF / TXT,自动解析正文</div>", unsafe_allow_html=True)
        resume_file = st.file_uploader("简历文件", type=["pdf", "txt"], key="resume",
                                       label_visibility="collapsed")
    with col_jd:
        st.markdown("<div class='sec'>🧭 岗位 JD</div>", unsafe_allow_html=True)
        jd_mode = st.radio("JD 输入方式", ["粘贴文本", "上传文件"], horizontal=True,
                           label_visibility="collapsed")
        jd_text_input = ""
        jd_file = None
        if jd_mode == "粘贴文本":
            jd_text_input = st.text_area("粘贴 JD 文本", height=180,
                                         placeholder="粘贴岗位职责与任职要求…",
                                         label_visibility="collapsed")
        else:
            jd_file = st.file_uploader("JD 文件", type=["pdf", "txt"], key="jd",
                                       label_visibility="collapsed")

    # 当前输入指纹:用于判断「上传/修改后是否需要重新匹配」。
    def _fp(f):
        return (f.name, f.size) if f is not None else None
    cur_fp = (_fp(resume_file), jd_mode,
              jd_text_input.strip() if jd_mode == "粘贴文本" else _fp(jd_file))

    st.write("")
    run = st.button("🚀 开始智能分析", type="primary", use_container_width=True)

    # 输入相对上次匹配是否已变更(用于让旧结果失效)。
    saved = st.session_state.get("match")
    input_changed = saved is not None and saved["fp"] != cur_fp

    if run:
        if resume_file is None:
            st.error("请先上传简历。")
            st.stop()
        if jd_mode == "粘贴文本" and not jd_text_input.strip():
            st.error("请粘贴 JD 文本,或切换到上传文件。")
            st.stop()
        if jd_mode == "上传文件" and jd_file is None:
            st.error("请上传 JD 文件,或切换到粘贴文本。")
            st.stop()

        with st.status("正在进行智能分析…", expanded=True) as status:
            st.write("① 解析简历与 JD 文档…")
            resume = read_upload(resume_file)
            if jd_mode == "粘贴文本":
                jd = {"text": jd_text_input, "sentences": split_sentences(jd_text_input)}
                jd_title = jd_text_input.strip().splitlines()[0][:40] if jd_text_input.strip() else "粘贴的 JD"
            else:
                jd = read_upload(jd_file)
                jd_title = jd_file.name
            st.write("② 加载语义模型并编码句向量…")
            embedder = get_embedder(model_name)
            st.write("③ 计算语义相似度与技能匹配…")
            result = match(resume, jd, embedder=embedder, top_n_skills=top_n, model_name=model_name)
            status.update(label=f"✅ 分析完成 · 综合匹配分 {result.score}/100",
                          state="complete", expanded=False)

        # 结果与输入指纹一起存入 session_state,刷新/换文件时据此判断是否过期。
        st.session_state["match"] = {
            "result": result,
            "resume_text": resume["text"],
            "summary": {"resume": resume_file.name, "jd": jd_title},
            "fp": cur_fp,
        }
        saved = st.session_state["match"]
        input_changed = False

    # ---- 渲染:输入已变更则让旧结果失效;否则展示已存结果 ----
    if input_changed:
        st.warning("🔄 输入已更新(简历或 JD 有改动),下方仍是上一次的结果。请点击「🚀 开始智能分析」重新计算。")

    if saved is not None:
        result = saved["result"]
        summary = saved["summary"]
        resume_text = saved["resume_text"]
        stale = "　·　🔴 结果已过期" if input_changed else ""

        st.write("")
        st.markdown("<span class='step-tag'>STEP 2 · 分析报告</span>", unsafe_allow_html=True)

        # ---- 结果摘要 Hero:输入 + 结论 + 综合分主视觉 ----
        st.markdown(
            f"""
            <div class="hero">
              <div class="meta">
                <div class="io">📄 简历 <b>{summary['resume']}</b>　🧭 岗位 <b>{summary['jd']}</b>{stale}</div>
                <div class="verdict">{verdict_text(result.score)}</div>
              </div>
              <div class="ring" style="background:{score_gradient(result.score)}">
                <div class="n">{result.score}</div><div class="u">综合匹配分 / 100</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- 自然语言总结 ----
        st.write("")
        st.markdown(f"<div class='summary'>📝 {summary_text(result)}</div>", unsafe_allow_html=True)

        # ---- 三指标卡:综合分主、语义/关键词辅 ----
        st.write("")
        c1, c2, c3 = st.columns(3)

        def _metric(col, val, label, color):
            col.markdown(
                f"<div class='metric'><div class='v' style='color:{color}'>{val}</div>"
                f"<div class='l'>{label}</div>"
                f"<div class='bar'><span style='width:{min(val,100)}%;background:{color}'></span></div></div>",
                unsafe_allow_html=True,
            )
        _metric(c1, result.score, "综合匹配分", score_color(result.score))
        _metric(c2, result.semantic_score, "语义相似度", "#5b7cfa")
        _metric(c3, result.keyword_score, "关键词命中率", "#7c5cff")
        st.caption("综合分 = 语义相似度 ×60% + 关键词命中 ×40%　·　语义=整体语义贴合度,关键词=JD 技能在简历中的命中比例")

        # ---- 技能卡:已匹配 / 缺失,硬技能优先 ----
        st.write("")
        s1, s2 = st.columns(2, gap="large")
        with s1:
            st.markdown(
                f"<div class='skill-box'><div class='skill-head'>✅ 已具备技能"
                f"<span class='badge badge-ok'>{len(result.matched_skills)}</span></div>"
                f"{render_chips(result.matched_skills, 'ok')}</div>",
                unsafe_allow_html=True,
            )
        with s2:
            if result.missing_skills:
                inner = render_chips(result.missing_skills, "miss")
            else:
                inner = "<span style='color:var(--ok);'>🎉 JD 要求的技能均已覆盖</span>"
            st.markdown(
                f"<div class='skill-box'><div class='skill-head'>⚠️ 待补强技能"
                f"<span class='badge badge-miss'>{len(result.missing_skills)}</span></div>"
                f"{inner}</div>",
                unsafe_allow_html=True,
            )

        # ---- 深度分析(二级模块,受 flag 控制;均基于已存结果重算)----
        deep_on = any(flags.get(k) for k in ("INDUSTRY_CLASSIFY", "RADAR", "HEATMAP"))
        if deep_on:
            st.write("")
            st.markdown("---")
            st.markdown("<span class='step-tag'>STEP 3 · 深度分析</span>", unsafe_allow_html=True)

        if flags.get("INDUSTRY_CLASSIFY"):
            st.write("")
            ind = classify_industry(resume_text)
            st.markdown("<div class='sec'>🏷️ 简历行业归类</div>", unsafe_allow_html=True)
            st.info(f"判定行业:**{ind['industry']}**　·　置信度 {ind['confidence']:.0%}")
            if ind["ranking"]:
                st.caption("命中分布:" + " · ".join(f"{n}({c})" for n, c in ind["ranking"]))

        if flags.get("RADAR"):
            st.write("")
            st.markdown("<div class='sec'>📡 四维度能力匹配</div>", unsafe_allow_html=True)
            st.markdown("<div class='sec-sub'>从技能 / 经验 / 学历 / 项目四个维度评估简历与岗位的契合度</div>",
                        unsafe_allow_html=True)
            scores = dimension_scores(
                result.resume_sentences, result.resume_vecs, result.jd_doc_vec,
                skill_coverage=result.keyword_score / 100,
                embedder=get_embedder(model_name),
            )
            rc1, rc2 = st.columns([1.1, 1], gap="large")
            with rc1:
                st.pyplot(radar_figure(scores))
            with rc2:
                st.markdown("**📋 维度解读**")
                st.markdown(radar_interpretation(scores))

        if flags.get("HEATMAP"):
            st.write("")
            st.markdown("<div class='sec'>🎯 JD 各项要求满足度</div>", unsafe_allow_html=True)
            st.markdown("<div class='sec-sub'>逐条检查 JD 要求,在简历中匹配最贴合的内容并给出满足度</div>",
                        unsafe_allow_html=True)
            cov_fig, cov_details = coverage_bar_figure(
                result.sim_matrix, result.resume_sentences, result.jd_sentences)
            st.pyplot(cov_fig)
            with st.expander("查看每条 JD 要求对应的简历原句"):
                for jd_s, res_s, sim in cov_details:
                    st.markdown(f"- **JD**:{jd_s}　→　**简历**:{res_s}　(满足度 {sim*100:.0f})")

        st.write("")
        with st.expander("🔧 解析明细(调试用)"):
            st.write("**简历句子**", result.resume_sentences)
            st.write("**JD 句子**", result.jd_sentences)
    elif not run:
        st.info("👆 上传简历并提供岗位 JD,点击「开始智能分析」即可生成匹配报告。")


else:
    # ---- 批量简历排序(加分项 BATCH_RANK)----
    st.markdown("<span class='step-tag'>批量模式</span>", unsafe_allow_html=True)
    st.markdown("<div class='sec'>📄 上传多份简历</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>对同一岗位 JD 逐份打分并按综合分排名,快速筛出最匹配的候选人</div>",
                unsafe_allow_html=True)
    resume_files = st.file_uploader(
        "简历(可多选,PDF 或 TXT)", type=["pdf", "txt"], key="batch", accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.markdown("<div class='sec'>🧭 岗位 JD</div>", unsafe_allow_html=True)
    jd_text_batch = st.text_area("粘贴 JD 文本", height=160, placeholder="粘贴岗位职责与任职要求…",
                                 label_visibility="collapsed")

    st.write("")
    run_batch = st.button("🚀 批量分析并排名", type="primary", use_container_width=True)

    if run_batch:
        if not resume_files:
            st.error("请至少上传一份简历。")
            st.stop()
        if not jd_text_batch.strip():
            st.error("请粘贴 JD 文本。")
            st.stop()
        if len(resume_files) < 2:
            st.warning("只传了 1 份简历,批量排序在多份对比时才有意义。建议传 2 份以上不同简历。")

        jd = {"text": jd_text_batch, "sentences": split_sentences(jd_text_batch)}
        embedder = get_embedder(model_name)
        rows = []
        prog = st.progress(0.0, text="逐份处理中…")
        # 逐份处理(不一次性 load 全部),省内存。
        for i, f in enumerate(resume_files):
            resume = read_upload(f)
            r = match(resume, jd, embedder=embedder, top_n_skills=top_n, model_name=model_name)
            rows.append({
                "简历": f.name,
                "综合分": r.score,
                "语义分": r.semantic_score,
                "关键词分": r.keyword_score,
                "命中技能数": len(r.matched_skills),
            })
            prog.progress((i + 1) / len(resume_files), text=f"已处理 {i + 1}/{len(resume_files)}")
        prog.empty()

        import pandas as pd

        df = pd.DataFrame(rows).sort_values("综合分", ascending=False).reset_index(drop=True)
        # 同名文件加序号区分,避免多行同名看不出谁是谁。
        if df["简历"].duplicated().any():
            df["简历"] = [f"{n} #{i+1}" for i, n in enumerate(df["简历"])]
        medals = {0: "🥇", 1: "🥈", 2: "🥉"}
        df.insert(0, "排名", [f"{medals.get(i, '')} {i+1}" for i in range(len(df))])

        st.markdown("### 🏆 批量排序结果")
        # 带颜色条的表格:综合分用进度条列,分数高低一目了然。
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "综合分": st.column_config.ProgressColumn(
                    "综合分", min_value=0, max_value=100, format="%.1f"),
                "语义分": st.column_config.ProgressColumn(
                    "语义分", min_value=0, max_value=100, format="%.1f"),
                "关键词分": st.column_config.ProgressColumn(
                    "关键词分", min_value=0, max_value=100, format="%.1f"),
            },
        )
        st.pyplot(rank_bar_figure(df["简历"].tolist(), df["综合分"].tolist()))
