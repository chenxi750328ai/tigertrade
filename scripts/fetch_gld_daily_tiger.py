#!/usr/bin/env python3
"""从老虎 QuoteClient 拉 GLD 日线，写入 CSV（供 backtest/optimize 使用）。

环境变量（可选）::
    TIGEROPEN_PROPS  或  TIGER_PROPS  — properties 文件路径

示例::

    export TIGEROPEN_PROPS=$HOME/tiger_openapi_config.properties
    python scripts/fetch_gld_daily_tiger.py -o data/gld_daily.csv --start 2010-01-01
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser(description="老虎 API 拉 GLD 日线 → CSV")
    ap.add_argument(
        "-o",
        "--output",
        required=True,
        help="输出 CSV 路径（Date,Close）",
    )
    ap.add_argument(
        "--tiger-props",
        default=os.environ.get("TIGEROPEN_PROPS", os.environ.get("TIGER_PROPS", "")),
        help="properties 路径，默认读环境变量 TIGEROPEN_PROPS / TIGER_PROPS",
    )
    ap.add_argument("--tiger-symbol", default="GLD", help="标的代码")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None, help="默认今天")
    args = ap.parse_args()

    props = (args.tiger_props or "").strip()
    if not props:
        print("请设置 --tiger-props 或环境变量 TIGEROPEN_PROPS", file=sys.stderr)
        sys.exit(2)

    # scripts/ 下运行：从 scripts 包导入 backtest 文件
    import importlib.util

    p = Path(__file__).resolve().parent / "backtest_gld_options_futures_proxy.py"
    spec = importlib.util.spec_from_file_location("_gld_bt", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["_gld_bt"] = mod
    spec.loader.exec_module(mod)

    import pandas as pd

    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    df = mod.download_gld_tiger(props, args.tiger_symbol, args.start, end)
    out = df.reset_index()
    out = out.rename(columns={out.columns[0]: "Date"})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"写入 {args.output}，共 {len(out)} 行")


if __name__ == "__main__":
    main()
