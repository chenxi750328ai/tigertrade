#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设计完备性 + DFX 门禁脚本（落地例行「对照设计完备性」的可执行验收）。

依据：
- docs/设计完备性_业界方案与DFX.md（可服务性 §3.4、可测试性 §3.7）
- shared_rag/best_practices/例行工作清单_agent必读.md（设计完备性项）

用法（仓库 tigertrade 根目录）：
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python scripts/verify_design_completeness_and_dfx.py

退出码：0 通过，非 0 失败（CI / 推前可调用）。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_DOCS = (
    "docs/设计完备性_业界方案与DFX.md",
    "docs/需求分析和Feature测试设计.md",
    "docs/reports/日志文件在哪儿.md",
    "docs/reports/止损未触发_取证与归因.md",
)

# 与 tests/test_design_completeness_dfx_gate.py、CI 步骤保持一致
DFX_GATE_TESTS = (
    "tests/test_design_completeness_dfx_gate.py",
    "tests/test_executor_dfx_emission.py",
    "tests/test_order_log_dfx.py",
    "tests/test_api_adapter_bracket_and_fill.py",
)


def main() -> int:
    os.chdir(ROOT)
    missing = [rel for rel in REQUIRED_DOCS if not (ROOT / rel).is_file()]
    if missing:
        print("[FAIL] 缺失设计/DFX 文档:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    cmd = [sys.executable, "-m", "pytest", *DFX_GATE_TESTS, "-v", "--tb=short"]
    print("[verify_design_completeness_and_dfx] Running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if r.returncode != 0:
        print("[FAIL] DFX/设计完备性回归用例未通过", file=sys.stderr)
    else:
        print("[OK] 文档齐全 + DFX/组合单回归用例通过")
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
