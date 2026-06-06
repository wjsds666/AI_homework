"""技能词提取与匹配。

设计(为何用词典而非纯 KeyBERT):
  KeyBERT 在中英混排、无分词的中文 JD 上会抽出「熟练掌握 java」这类中文动词+英文
  技能的污染短语(已记 CLAUDE.md Bug #9),导致与简历整串子串匹配恒为 0。
  简历匹配里「技能」本是有限可枚举的专有名词,故改用「技能词典扫描」为主:
    - extract_skills:在 JD 文本中扫词典命中的技能(规范名)。
    - match_skills:在简历中用同一套别名规则检测命中。
  提取与匹配共用别名规则 → JD 识别到的技能,只要简历里出现就一定能匹配上。
  KeyBERT 作为可选增强保留(extract_keyphrases),供报告演示,不参与主链路。
"""

from __future__ import annotations

import re

# 技能词典:规范名 -> 匹配别名(全部小写)。别名命中即算该技能出现。
# 覆盖测试数据涉及的 IT 技能;可按需扩充。
SKILL_DB: dict[str, list[str]] = {
    "Python": ["python"],
    "Java": ["java"],
    "JavaScript": ["javascript", "js"],
    "TypeScript": ["typescript", "ts"],
    "C++": ["c++"],
    "C#": ["c#"],
    "Go": ["golang", "go语言"],
    "SQL": ["sql"],
    "MySQL": ["mysql"],
    "Redis": ["redis"],
    "MongoDB": ["mongodb", "mongo"],
    "Hive": ["hive"],
    "Spring Boot": ["spring boot", "springboot", "spring"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Kafka": ["kafka"],
    "RabbitMQ": ["rabbitmq"],
    "RESTful API": ["restful", "rest api", "restful api"],
    "微服务": ["微服务", "microservice", "microservices"],
    "Linux": ["linux"],
    "Git": ["git"],
    "React": ["react"],
    "Vue": ["vue"],
    "Node.js": ["node.js", "nodejs", "node"],
    "HTML": ["html", "html5"],
    "CSS": ["css", "css3"],
    "Webpack": ["webpack"],
    "Vite": ["vite"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "XGBoost": ["xgboost"],
    "机器学习": ["机器学习", "machine learning"],
    "Tableau": ["tableau"],
    "Power BI": ["power bi", "powerbi"],
    "matplotlib": ["matplotlib"],
    "ECharts": ["echarts"],
    "A/B测试": ["a/b test", "a/b 测试", "ab test", "abtest", "a/b测试"],
    "数据分析": ["数据分析", "data analysis"],
    "数据可视化": ["数据可视化", "data visualization"],
    # ---- 产品 / 项目管理 ----
    "产品经理": ["产品经理", "product manager", "产品策划"],
    "需求分析": ["需求分析"],
    "产品规划": ["产品规划"],
    "需求文档": ["prd", "产品需求文档", "需求文档"],
    "原型设计": ["原型设计", "原型", "prototype"],
    "高保真原型": ["高保真"],
    "竞品分析": ["竞品分析", "竞品"],
    "用户画像": ["用户画像", "user persona"],
    "用户调研": ["用户调研", "用户研究", "user research"],
    "产品路线图": ["产品路线图", "路线图", "roadmap"],
    "Axure": ["axure"],
    "Figma": ["figma"],
    "Sketch": ["sketch"],
    "敏捷开发": ["敏捷", "scrum", "agile"],
    "项目管理": ["项目管理", "project management", "项目推进"],
    "商业化": ["商业化", "变现"],
    "定价策略": ["定价"],
    "数据驱动": ["数据驱动", "data driven"],
    "留存率": ["留存率", "留存"],
    "转化率": ["转化率"],
    "NPS": ["nps"],
    # ---- 运营 / 市场 ----
    "用户增长": ["用户增长", "增长", "growth"],
    "运营": ["运营", "operation"],
    "市场营销": ["市场营销", "marketing"],
    "客户成功": ["客户成功", "customer success"],
    "SaaS": ["saas"],
    "B端": ["b端", "to b", "tob", "企业客户"],
    "C端": ["c端", "to c", "toc"],
    # ---- 通用软技能 ----
    "沟通协调": ["沟通协调", "沟通能力", "沟通"],
    "团队协作": ["团队协作", "跨职能", "跨部门", "协作"],
    "抗压能力": ["抗压", "抗压能力"],
    "学习能力": ["学习能力", "快速学习"],
    "执行力": ["执行力"],
    "责任心": ["责任心", "责任感"],
    "逻辑思维": ["逻辑思维", "逻辑能力"],
    "英语": ["英语", "english", "cet-6", "cet6", "六级", "雅思", "ielts", "托福", "toefl"],
    # ---- 扩展:编程语言 / 后端 ----
    "C语言": ["c语言"],
    "PHP": ["php"],
    "Ruby": ["ruby"],
    "Rust": ["rust"],
    "Scala": ["scala"],
    "Kotlin": ["kotlin"],
    "Swift": ["swift"],
    "Shell": ["shell", "bash"],
    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi"],
    "Spring Cloud": ["spring cloud", "springcloud"],
    "MyBatis": ["mybatis"],
    "GraphQL": ["graphql"],
    "gRPC": ["grpc"],
    "Nginx": ["nginx"],
    "PostgreSQL": ["postgresql", "postgres"],
    "Oracle": ["oracle"],
    "Elasticsearch": ["elasticsearch", "es搜索"],
    # ---- 扩展:前端 / 移动端 ----
    "Angular": ["angular"],
    "Next.js": ["next.js", "nextjs"],
    "小程序": ["小程序", "微信小程序"],
    "Flutter": ["flutter"],
    "Android": ["android", "安卓"],
    "iOS": ["ios"],
    "UniApp": ["uniapp", "uni-app"],
    "Sass": ["sass", "scss", "less"],
    # ---- 扩展:AI / 大数据 ----
    "深度学习": ["深度学习", "deep learning"],
    "PyTorch": ["pytorch"],
    "TensorFlow": ["tensorflow"],
    "自然语言处理": ["nlp", "自然语言处理"],
    "计算机视觉": ["计算机视觉", "cv", "computer vision"],
    "大模型": ["大模型", "llm", "大语言模型", "gpt"],
    "Spark": ["spark"],
    "Hadoop": ["hadoop"],
    "Flink": ["flink"],
    "数据仓库": ["数据仓库", "数仓", "data warehouse"],
    "ETL": ["etl"],
    "数据挖掘": ["数据挖掘", "data mining"],
    "推荐系统": ["推荐系统", "recommendation"],
    # ---- 测试 / 运维 / 云 ----
    "自动化测试": ["自动化测试", "automation test"],
    "Selenium": ["selenium"],
    "性能测试": ["性能测试", "jmeter", "loadrunner"],
    "测试用例": ["测试用例", "test case"],
    "CI/CD": ["ci/cd", "cicd", "持续集成", "持续交付"],
    "Jenkins": ["jenkins"],
    "DevOps": ["devops"],
    "Prometheus": ["prometheus"],
    "Grafana": ["grafana"],
    "AWS": ["aws", "亚马逊云"],
    "阿里云": ["阿里云", "aliyun"],
    "腾讯云": ["腾讯云"],
    "Terraform": ["terraform"],
    "Ansible": ["ansible"],
    # ---- 设计 ----
    "UI设计": ["ui设计", "ui design"],
    "UX设计": ["ux", "用户体验", "交互设计"],
    "Photoshop": ["photoshop", "ps设计"],
    "Illustrator": ["illustrator", "ai设计"],
    "C4D": ["c4d", "cinema 4d"],
    "视觉设计": ["视觉设计", "平面设计"],
    "三维建模": ["三维建模", "3d建模", "建模"],
    # ---- 财务 / 金融 ----
    "财务分析": ["财务分析", "financial analysis"],
    "会计": ["会计", "accounting"],
    "审计": ["审计", "audit"],
    "税务": ["税务", "纳税"],
    "成本控制": ["成本控制", "成本核算"],
    "Excel": ["excel", "vlookup", "数据透视"],
    "财务报表": ["财务报表", "三大报表"],
    "CPA": ["cpa", "注册会计师"],
    "风险控制": ["风险控制", "风控", "risk control"],
    "投资分析": ["投资分析", "投研", "估值"],
    # ---- 人力 / 行政 ----
    "招聘": ["招聘管理", "招聘渠道", "招聘工作", "负责招聘", "recruitment", "recruiting", "人才招聘", "校招", "社招"],
    "绩效管理": ["绩效管理", "绩效考核", "kpi", "okr"],
    "薪酬福利": ["薪酬", "薪酬福利", "compensation"],
    "员工关系": ["员工关系", "employee relation"],
    "组织发展": ["组织发展", "od"],
    "培训": ["培训", "training"],
    # ---- 法律 ----
    "合同审查": ["合同审查", "合同管理"],
    "合规": ["合规", "compliance"],
    "知识产权": ["知识产权", "专利", "商标"],
    "诉讼": ["诉讼", "仲裁", "litigation"],
    "法律顾问": ["法律顾问", "法务"],
    # ---- 医疗 ----
    "临床": ["临床", "clinical"],
    "护理": ["护理", "nursing"],
    "药学": ["药学", "药剂", "pharmacy"],
    "医学影像": ["医学影像", "影像诊断"],
    "病历管理": ["病历", "电子病历"],
    # ---- 教育 ----
    "教学设计": ["教学设计", "课程设计"],
    "课程开发": ["课程开发", "教研"],
    "教师资格": ["教师资格", "教资"],
    "班级管理": ["班级管理", "班主任"],
    # ---- 销售 / 客服 ----
    "销售": ["销售", "sales"],
    "商务拓展": ["商务拓展", "bd", "business development"],
    "客户关系": ["客户关系", "crm", "客户维护"],
    "渠道管理": ["渠道管理", "渠道开发"],
    "客户服务": ["客户服务", "客服"],
    "谈判": ["谈判", "negotiation"],
}

# 别名是否「纯 ASCII 单词」(用词边界匹配)还是「含特殊符号/中文」(用子串匹配)。
_ASCII_WORD = re.compile(r"^[a-z0-9]+$")


def _alias_hit(text_low: str, alias: str) -> bool:
    """判断别名是否命中文本。

    - 纯 ASCII 单词(如 java)用词边界,避免 java 命中 javascript。
    - 含特殊符号(c++/c#/node.js)或中文别名用子串匹配。
    """
    if _ASCII_WORD.match(alias):
        return re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text_low) is not None
    return alias in text_low


# JD 中「任职要求」类小标题(技能要求集中在此段;职责段只描述岗位干啥,噪声多)。
_REQ_HEADER = re.compile(
    r"(任职要求|任职资格|岗位要求|应聘条件|招聘条件|任职条件|岗位资格"
    r"|requirements|qualifications|我们希望你|你需要具备|加分项)"
)
# 「岗位职责」类小标题:出现在它之后、要求段之前的内容视为职责描述(可能含噪声)。
_DUTY_HEADER = re.compile(r"(岗位职责|工作职责|职位描述|工作内容|岗位描述|responsibilities|job\s*description)")


def _requirement_section(jd_text: str) -> str:
    """截取 JD 的「任职要求」段。

    技能要求集中在要求段;职责段(如「客户成功案例沉淀」「配合销售团队」)是岗位
    描述、不是对候选人的技能要求,从源头排除可避免这类误命中。
    找不到要求小标题时回退全文(保证不漏)。
    """
    m = _REQ_HEADER.search(jd_text)
    if m:
        return jd_text[m.start():]
    return jd_text  # 无明确分段,回退扫全文。


def _detect(text: str) -> list[str]:
    """扫描文本,返回命中的规范技能名(按词典顺序,去重)。"""
    text_low = text.lower()
    hits: list[str] = []
    for canonical, aliases in SKILL_DB.items():
        if any(_alias_hit(text_low, a) for a in aliases):
            hits.append(canonical)
    return hits


def extract_skills(jd_text: str, *, top_n: int = 30, **_ignored) -> list[str]:
    """从 JD 提取技能词(词典命中的规范名)。

    只从「任职要求」段提取,避开职责段的描述性噪声(如「客户成功案例」误命中
    「客户成功」、「配合销售团队」误命中「销售」)。无要求段时回退全文。
    top_n 仅作上限;**_ignored 兼容旧调用里传的 model_name 等参数。
    """
    if not jd_text.strip():
        return []
    return _detect(_requirement_section(jd_text))[:top_n]


def match_skills(resume_text: str, skills: list[str]) -> dict:
    """在简历中检测给定技能是否命中。

    技能为 SKILL_DB 的规范名;用其别名在简历里匹配,与提取共用规则。

    Returns:
        {'matched': [...], 'missing': [...], 'coverage': 命中比例 0~1}
    """
    resume_low = resume_text.lower()
    matched, missing = [], []
    for skill in skills:
        aliases = SKILL_DB.get(skill, [skill.lower()])
        if any(_alias_hit(resume_low, a) for a in aliases):
            matched.append(skill)
        else:
            missing.append(skill)
    coverage = len(matched) / len(skills) if skills else 0.0
    return {"matched": matched, "missing": missing, "coverage": coverage}


def extract_keyphrases(jd_text: str, *, top_n: int = 10, model_name: str | None = None) -> list[tuple[str, float]]:
    """[可选/报告演示] 用 KeyBERT 提取关键短语,返回 (短语, 分数)。

    仅供报告展示「KeyBERT 关键词提取」用,不参与主匹配链路。
    懒加载 KeyBERT,避免无谓的模型加载。
    """
    from keybert import KeyBERT

    from .embedder import DEFAULT_MODEL, load_model

    kw = KeyBERT(model=load_model(model_name or DEFAULT_MODEL))
    return kw.extract_keywords(
        jd_text,
        keyphrase_ngram_range=(1, 1),
        stop_words=None,
        top_n=top_n,
    )
