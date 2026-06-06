"""简历 / JD 文本解析:支持 PDF(pdfplumber)与 TXT,输出纯文本 + 按句切分。

设计要点:
- PDF 用 pdfplumber 逐页抽取;TXT 直接读。
- 中英文混排,句子切分同时按中文句末标点(。!?;)和英文标点(.!?;)切。
- 切分后做基本清洗:去空白、过滤过短碎片(如孤立的页码、符号)。
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pdfplumber

# 句末标点:中英文统一处理。保留分号便于把简历里分号分隔的技能拆开。
_SENT_SPLIT = re.compile(r"(?<=[。!?；!?;])\s*|\n+")
# 用于判断一段文本是否“信息量太低”(纯符号 / 纯页码)。
_MEANINGFUL = re.compile(r"[A-Za-z一-鿿0-9]")
_HAS_LETTER_OR_CJK = re.compile(r"[A-Za-z一-鿿]")


def extract_text_from_pdf(data: bytes) -> str:
    """从 PDF 字节流抽取纯文本。逐页提取后用换行拼接。"""
    parts: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt:
                parts.append(txt)
    return "\n".join(parts)


def extract_text_from_txt(data: bytes) -> str:
    """从 TXT 字节流解码文本,优先 UTF-8,兜底 GBK。"""
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    # 最后兜底:忽略无法解码的字节,保证不崩。
    return data.decode("utf-8", errors="ignore")


def load_text(source: str | Path | bytes, *, filename: str | None = None) -> str:
    """统一入口:接受文件路径或字节流,按扩展名 / 内容选择解析器。

    Args:
        source: 文件路径(str/Path)或文件字节内容(bytes)。
        filename: 当 source 为 bytes 时,用文件名判断类型(.pdf / .txt)。
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        data = path.read_bytes()
        name = path.name.lower()
    else:
        data = source
        name = (filename or "").lower()

    if name.endswith(".pdf") or (not name and _looks_like_pdf(data)):
        return extract_text_from_pdf(data)
    return extract_text_from_txt(data)


def _looks_like_pdf(data: bytes) -> bool:
    """无文件名时,用 PDF 魔数 %PDF 粗判。"""
    return data[:4] == b"%PDF"


def split_sentences(text: str, *, min_len: int = 2) -> list[str]:
    """把整段文本切成句子列表。

    Args:
        text: 原始纯文本。
        min_len: 过滤掉有意义字符数少于该值的碎片。
    """
    raw = _SENT_SPLIT.split(text)
    sentences: list[str] = []
    for seg in raw:
        seg = seg.strip()
        if not seg:
            continue
        # 过滤纯符号 / 过短碎片(页码、分隔线等)。
        if len(_MEANINGFUL.findall(seg)) < min_len:
            continue
        # 进一步过滤纯数字碎片,如页码、编号。
        if not _HAS_LETTER_OR_CJK.search(seg):
            continue
        sentences.append(seg)
    return sentences


def parse(source: str | Path | bytes, *, filename: str | None = None) -> dict:
    """解析入口:返回 {'text': 纯文本, 'sentences': 句子列表}。"""
    text = load_text(source, filename=filename)
    return {"text": text, "sentences": split_sentences(text)}
