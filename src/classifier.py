"""行业分类(加分项):判断简历/文本属于哪个行业。

方法:关键词词典打分法(简单、可解释、零额外模型)。
  为每个行业维护一组特征关键词,统计文本命中数,取得分最高的行业。
  命中数全为 0 时返回「通用/未知」。

受 config 的 INDUSTRY_CLASSIFY flag 控制是否在 UI 展示。
"""

from __future__ import annotations

# 行业 → 特征关键词(小写匹配)。覆盖常见行业,可按需扩充。
INDUSTRY_KEYWORDS: dict[str, list[str]] = {
    "IT/互联网": [
        "python", "java", "javascript", "docker", "kubernetes", "微服务",
        "后端", "前端", "算法", "数据库", "redis", "mysql", "spring",
        "react", "vue", "linux", "git", "api", "服务器",
    ],
    "金融": [
        "财务", "会计", "审计", "税务", "cpa", "财务报表", "成本控制",
        "风险控制", "风控", "投资", "估值", "金融", "银行", "证券", "基金",
    ],
    "教育": [
        "教学", "课程", "教研", "教师", "教资", "班级", "学生", "培训",
        "教育", "授课", "备课", "教学设计",
    ],
    "医疗": [
        "临床", "护理", "药学", "药剂", "医学", "病历", "诊断", "医院",
        "患者", "医疗", "影像",
    ],
    "产品/运营": [
        "产品经理", "需求分析", "原型", "axure", "figma", "竞品", "用户画像",
        "留存率", "转化率", "nps", "运营", "增长", "saas",
    ],
    "人力/行政": [
        "招聘", "绩效", "薪酬", "员工关系", "组织发展", "人力资源", "hr",
        "okr", "kpi", "行政",
    ],
    "市场/销售": [
        "销售", "市场营销", "marketing", "商务拓展", "客户关系", "渠道",
        "客户服务", "谈判", "品牌", "推广",
    ],
    "设计": [
        "ui设计", "ux", "视觉设计", "平面设计", "photoshop", "illustrator",
        "c4d", "三维建模", "交互设计",
    ],
}


def classify_industry(text: str, *, top_k: int = 3) -> dict:
    """对文本做行业分类。

    Returns:
        {
          'industry': 最佳行业名(或「通用/未知」),
          'confidence': 该行业命中占总命中的比例 0~1,
          'ranking': [(行业, 命中数), ...] 前 top_k(命中数降序)
        }
    """
    text_low = text.lower()
    counts: dict[str, int] = {}
    for industry, kws in INDUSTRY_KEYWORDS.items():
        counts[industry] = sum(1 for kw in kws if kw in text_low)

    total = sum(counts.values())
    ranking = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    ranking = [(name, c) for name, c in ranking if c > 0][:top_k]

    if total == 0:
        return {"industry": "通用/未知", "confidence": 0.0, "ranking": []}

    best_name, best_count = ranking[0]
    return {
        "industry": best_name,
        "confidence": round(best_count / total, 2),
        "ranking": ranking,
    }
