# -*- coding: utf-8 -*-
"""
设计完备性 + DFX 文档门禁（与 docs/设计完备性_业界方案与DFX.md §3.4/3.7 对齐）。

删除或改名关键设计文档时 CI 必须失败，避免「例行写了对照」却无物可对照。
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# 与 shared_rag/best_practices/例行工作清单_agent必读.md 中「设计完备性」引用一致
REQUIRED_DESIGN_DOCS = (
    "docs/设计完备性_业界方案与DFX.md",
    "docs/需求分析和Feature测试设计.md",
    "docs/reports/日志文件在哪儿.md",
    "docs/reports/止损未触发_取证与归因.md",
)


@pytest.mark.parametrize("rel", REQUIRED_DESIGN_DOCS)
def test_required_design_document_exists(rel):
    p = ROOT / rel
    assert p.is_file(), f"缺失设计/DFX 相关文档（例行与门禁依赖）: {rel}"


def test_order_log_module_defines_dfx_sink():
    """实现侧必须保留 DFX 落盘常量与 log_dfx（可服务性）。"""
    from src import order_log

    assert hasattr(order_log, "DFX_EXECUTION_FILE")
    assert hasattr(order_log, "log_dfx")
    assert callable(order_log.log_dfx)
    src = Path(order_log.__file__).read_text(encoding="utf-8")
    assert "dfx_execution.jsonl" in src
    assert "def log_dfx" in src
