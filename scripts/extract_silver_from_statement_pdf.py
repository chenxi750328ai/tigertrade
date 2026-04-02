#!/usr/bin/env python3
"""
从老虎「今年以来账单报表」PDF 中提取白银相关行，用于与白银持仓分析报告对账、发现遗漏、刷新报告。
用法：
  python scripts/extract_silver_from_statement_pdf.py
  python scripts/extract_silver_from_statement_pdf.py --pdf /path/to/今年以来账单报表0301.pdf --out docs/0301_白银提取.txt
输出：包含 SLV、SIL、白银、微白银、相关日期与数量的行，按页与行号排列，便于与 PDF 对照。
"""

import argparse
import re
import sys
from pathlib import Path

# 尝试 PyPDF2，若无则 pdfminer
try:
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False
try:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer
    HAS_PDFMINER = True
except ImportError:
    HAS_PDFMINER = False


def extract_text_pypdf2(pdf_path: str) -> list[tuple[int, str]]:
    """(页码 1-based, 行文本)"""
    reader = PdfReader(pdf_path)
    out = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        for line in text.splitlines():
            line = line.strip()
            if line:
                out.append((i + 1, line))
    return out


def extract_text_pdfminer(pdf_path: str) -> list[tuple[int, str]]:
    out = []
    for page_num, page in enumerate(extract_pages(pdf_path)):
        for el in page:
            if isinstance(el, LTTextContainer):
                for line in el.get_text().splitlines():
                    line = line.strip()
                    if line:
                        out.append((page_num + 1, line))
    return out


def extract_text(pdf_path: str) -> list[tuple[int, str]]:
    if HAS_PYPDF2:
        return extract_text_pypdf2(pdf_path)
    if HAS_PDFMINER:
        return extract_text_pdfminer(pdf_path)
    raise RuntimeError("需要安装 PyPDF2 或 pdfminer.six: pip install PyPDF2 或 pip install pdfminer.six")


# 白银相关：标的、常见行权价/数量、关键日期
SILVER_KEYWORDS = re.compile(
    r"SLV|SIL2603|SIL2605|SIL2604|SIL2606|微白银|白银|"
    r"67\.0|67\.5|72\.5|75\.0|88\.0|90\.0|100\.0|105\.0|120\.0|70\.0|76\.0|71\.0|60\.0|65\.0|"
    r"2026-02-26|2026-02-27|02-26|02-27|"
    r"500\s|500\.|,500\b|\b500\b",
    re.I
)


def main():
    parser = argparse.ArgumentParser(description="从账单 PDF 提取白银相关行")
    parser.add_argument("--pdf", default=None, help="PDF 路径，默认 docs/今年以来账单报表0301.pdf")
    parser.add_argument("--out", default=None, help="输出 txt 路径，默认 docs/0301_白银提取.txt")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    pdf_path = args.pdf or str(base / "docs" / "今年以来账单报表0301.pdf")
    out_path = args.out or str(base / "docs" / "0301_白银提取.txt")

    if not Path(pdf_path).exists():
        print(f"未找到 PDF: {pdf_path}", file=sys.stderr)
        print("请将 今年以来账单报表0301.pdf 放到 docs/ 或使用 --pdf 指定路径。", file=sys.stderr)
        sys.exit(1)

    print(f"正在提取: {pdf_path}")
    lines = extract_text(pdf_path)
    silver = []
    for page, line in lines:
        if SILVER_KEYWORDS.search(line):
            silver.append((page, line))

    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# 白银相关行（来源: {Path(pdf_path).name}）\n")
        f.write(f"# 共 {len(silver)} 行（总行数 {len(lines)}）\n\n")
        for page, line in silver:
            f.write(f"p{page}\t{line}\n")
    print(f"已写入 {len(silver)} 行 -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
