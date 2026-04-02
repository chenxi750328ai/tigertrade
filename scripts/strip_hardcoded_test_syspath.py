#!/usr/bin/env python3
"""维护：去掉 tests 里硬编码 /home/cx/tigertrade 的 sys.path 块（正常运行依赖 pytest.ini pythonpath + conftest）。"""
import re
from pathlib import Path

TESTS = Path(__file__).resolve().parent.parent / "tests"

BLOCK = re.compile(
    r"(?:#[^\n]*\n)*"
    r"tigertrade_dir = ['\"]/home/cx/tigertrade['\"]\s*\n"
    r"if tigertrade_dir not in sys\.path:\s*\n"
    r"\s*sys\.path\.insert\(0, tigertrade_dir\)\s*\n",
    re.MULTILINE,
)

SINGLE_PATTERNS = [
    re.compile(r"^\s*sys\.path\.insert\(0,\s*['\"]/home/cx/tigertrade['\"]\)\s*\n", re.MULTILINE),
    re.compile(r"^\s*_sys\.path\.insert\(0,\s*['\"]/home/cx/tigertrade['\"]\)\s*\n", re.MULTILINE),
    re.compile(r"^\s*sys\.path\.insert\(0,\s*['\"]/home/cx/tigertrade1['\"]\)\s*\n", re.MULTILINE),
]


def clean_text(text: str) -> str:
    text = BLOCK.sub("", text)
    for p in SINGLE_PATTERNS:
        text = p.sub("", text)
    return text


def main():
    n = 0
    for path in sorted(TESTS.rglob("*.py")):
        raw = path.read_text(encoding="utf-8")
        new = clean_text(raw)
        if new != raw:
            path.write_text(new, encoding="utf-8")
            n += 1
            print("updated", path.relative_to(TESTS.parent))
    print("files changed:", n)


if __name__ == "__main__":
    main()
