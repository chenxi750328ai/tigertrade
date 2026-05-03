#!/usr/bin/env python3
"""
基于历史 GLD 日线，对「三档 moneyness 各占 1/3 张数」的期权腿做网格搜索，选出 Sharpe / Calmar / CAGR 较优的参数。

约定（与 backtest_gld_options_futures_proxy 一致）::
    Short Call: K = S * (1 + m)，m 越 **负** 越深度实值，m 越 **正** 越虚值。
    Long  Put: K = S * (1 - m)，m 越 **负** 越深度实值（Put 价内=高行权价），m 越 **正** 越虚值。

三腿张数：每桶 ``--contracts-per-bucket`` 张（默认 2 → 每档 2 张，共 6 张 Short Call；
若 ``--with-put-thirds`` 则再加 6 张 Long Put 三档）。

依赖与回测脚本相同: pip install yfinance scipy pandas numpy tqdm

示例::

    python scripts/optimize_gld_options_moneyness.py --csv data/gld.csv \\
        --objective calmar --top 15

    python scripts/optimize_gld_options_moneyness.py --csv data/gld.csv \\
        --align-15d-cycle --include-430-ladder --strict-order --objective calmar --top 10
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import itertools
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore


def _load_backtest_module():
    p = Path(__file__).resolve().parent / "backtest_gld_options_futures_proxy.py"
    name = "_gld_backtest_mod"
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载回测模块")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bt = _load_backtest_module()
Leg = bt.Leg
_run_backtest_clean = bt._run_backtest_clean
download_gld = bt.download_gld
download_gld_tiger = getattr(bt, "download_gld_tiger", None)
load_close_csv = bt.load_close_csv


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def build_legs_short_call_thirds(
    n_per: int, m_deep: float, m_mid: float, m_otm: float, with_put_thirds: Optional[Tuple[float, float, float]] = None
) -> List[Leg]:
    """三档 Short Call 各 n_per 张；可选三档 Long Put（m_put_deep, m_put_mid, m_put_otm）各 n_per。"""
    legs = [
        Leg("SC", m_deep, n_per),
        Leg("SC", m_mid, n_per),
        Leg("SC", m_otm, n_per),
    ]
    if with_put_thirds is not None:
        pd_, pm, po = with_put_thirds
        legs.extend(
            [
                Leg("LP", pd_, n_per),
                Leg("LP", pm, n_per),
                Leg("LP", po, n_per),
            ]
        )
    return legs


def calmar(summary: Dict[str, Any]) -> float:
    cagr = float(summary.get("cagr", float("nan")))
    mdd = float(summary.get("max_drawdown", 0.0))
    if mdd >= -1e-9:
        return float("nan")
    return cagr / abs(mdd)


def score(summary: Dict[str, Any], objective: str) -> float:
    if objective == "sharpe":
        return float(summary.get("sharpe", float("nan")))
    if objective == "cagr":
        return float(summary.get("cagr", float("nan")))
    if objective == "calmar":
        return calmar(summary)
    raise ValueError(objective)


def load_close_series(args: argparse.Namespace) -> pd.Series:
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    if args.csv:
        df = load_close_csv(args.csv)
        close = df["Close"].astype(float).sort_index()
        if args.start:
            close = close[close.index >= pd.Timestamp(args.start)]
        if args.end:
            close = close[close.index <= pd.Timestamp(args.end)]
    elif (getattr(args, "tiger_props", "") or "").strip() and download_gld_tiger is not None:
        df = download_gld_tiger(
            args.tiger_props.strip(),
            getattr(args, "tiger_symbol", "GLD").strip() or "GLD",
            args.start,
            end,
        )
        close = df["Close"].astype(float).sort_index()
        if args.start:
            close = close[close.index >= pd.Timestamp(args.start)]
        if args.end:
            close = close[close.index <= pd.Timestamp(args.end)]
    else:
        df = download_gld(args.start, end)
        close = df["Close"].astype(float)
    if len(close) < 50:
        raise SystemExit(f"有效收盘价不足 50 根（当前 {len(close)}）")
    return close


@dataclass
class ComboResult:
    m_deep: float
    m_mid: float
    m_otm: float
    put_triple: Optional[Tuple[float, float, float]]
    summary: Dict[str, Any]

    def row(self) -> Dict[str, Any]:
        d = {
            "sc_m_deep": self.m_deep,
            "sc_m_mid": self.m_mid,
            "sc_m_otm": self.m_otm,
            **self.summary,
        }
        if self.put_triple is not None:
            d["lp_m_deep"] = self.put_triple[0]
            d["lp_m_mid"] = self.put_triple[1]
            d["lp_m_otm"] = self.put_triple[2]
        return d


def main() -> None:
    ap = argparse.ArgumentParser(
        description="GLD 三档 moneyness 1/3 张数 网格优化",
        epilog="含负数的 grid 请写成 --sc-deep-grid=-0.1,-0.08（等号），否则 argparse 会误解析。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--csv", default=None)
    ap.add_argument(
        "--tiger-props",
        default=os.environ.get("TIGEROPEN_PROPS", os.environ.get("TIGER_PROPS", "")),
        help="老虎 properties；若设置且未指定 --csv，则拉 GLD 日线",
    )
    ap.add_argument("--tiger-symbol", default="GLD")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument(
        "--sc-deep-grid",
        default="-0.14,-0.11,-0.08,-0.06",
        help="Short Call 深度实值档 m 列表（K=S(1+m)，负得越多越深实值）",
    )
    ap.add_argument(
        "--sc-mid-grid",
        default="-0.025,-0.015,-0.008,0.0,0.012",
        help="Short Call 实值/近档 m 列表",
    )
    ap.add_argument(
        "--sc-otm-grid",
        default="0.028,0.038,0.048,0.062,0.080",
        help="Short Call 虚值档 m 列表",
    )
    ap.add_argument(
        "--lp-deep-grid",
        default="-0.11,-0.08,-0.05",
        help="Long Put 深度实值档 m（LP: K=S(1-m)，负=高行权价=Put价内）",
    )
    ap.add_argument(
        "--lp-mid-grid",
        default="-0.035,-0.02,-0.01",
        help="Long Put 中间档 m",
    )
    ap.add_argument(
        "--lp-otm-grid",
        default="0.02,0.035,0.05",
        help="Long Put 虚值档 m（正=低行权价=Put虚值）",
    )
    ap.add_argument("--with-put-thirds", action="store_true", help="同时搜索三档 Long Put（各占 1/3 张）")
    ap.add_argument("--contracts-per-bucket", type=int, default=2, help="每一 moneyness 档的张数")
    ap.add_argument("--rebalance-every", type=int, default=5)
    ap.add_argument("--dte-days", type=int, default=7)
    ap.add_argument(
        "--align-15d-cycle",
        action="store_true",
        help="dte=15、rebalance=10，与两周持有期一致",
    )
    ap.add_argument(
        "--include-430-ladder",
        action="store_true",
        help="在网格最前强制加入锚定价 400/430/450 三档 m（S=430 时等价）",
    )
    ap.add_argument(
        "--anchor-for-seed",
        type=float,
        default=430.0,
        help="与 --include-430-ladder 搭配，换算 m=K/anchor-1",
    )
    ap.add_argument("--vol-window", type=int, default=20)
    ap.add_argument("--r-free", type=float, default=0.04)
    ap.add_argument("--fee-per-contract", type=float, default=0.65)
    ap.add_argument("--basis-drag-bps", type=float, default=25.0)
    ap.add_argument("--gap-slippage-bps", type=float, default=5.0)
    ap.add_argument("--share-hedge-ratio", type=float, default=1.0)
    ap.add_argument("--initial-capital", type=float, default=100_000.0)
    ap.add_argument("--objective", choices=("sharpe", "calmar", "cagr"), default="calmar")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--max-combos", type=int, default=0, help=">0 时随机子采样这么多组（大网格用）")
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--out-csv", default=None, help="结果写 CSV")
    ap.add_argument(
        "--strict-order",
        action="store_true",
        help="仅保留 m_deep < m_mid < m_otm（Short Call 行权价递增）",
    )
    args = ap.parse_args()

    if args.align_15d_cycle:
        args.dte_days = 15
        args.rebalance_every = 10

    g_deep = _parse_float_list(args.sc_deep_grid)
    g_mid = _parse_float_list(args.sc_mid_grid)
    g_otm = _parse_float_list(args.sc_otm_grid)
    lp_d = _parse_float_list(args.lp_deep_grid)
    lp_m = _parse_float_list(args.lp_mid_grid)
    lp_o = _parse_float_list(args.lp_otm_grid)

    close = load_close_series(args)

    sc_combos = list(itertools.product(g_deep, g_mid, g_otm))
    if args.strict_order:
        sc_combos = [(a, b, c) for a, b, c in sc_combos if a < b < c]

    seed_sc_430: Optional[Tuple[float, float, float]] = None
    if args.include_430_ladder:
        a = args.anchor_for_seed
        seed_sc_430 = (400.0 / a - 1.0, 0.0, 450.0 / a - 1.0)
        if seed_sc_430 not in sc_combos:
            sc_combos.insert(0, seed_sc_430)

    if args.with_put_thirds:
        lp_combos = list(itertools.product(lp_d, lp_m, lp_o))
        lp_combos = [(a, b, c) for a, b, c in lp_combos if a < b < c]
        full = list(itertools.product(sc_combos, lp_combos))
    else:
        full = [(sc, None) for sc in sc_combos]

    if args.max_combos > 0 and len(full) > args.max_combos:
        rnd = random.Random(args.random_seed)
        full = rnd.sample(full, args.max_combos)

    if seed_sc_430 is not None and not args.with_put_thirds:
        if (seed_sc_430, None) not in full:
            full.insert(0, (seed_sc_430, None))

    results: List[ComboResult] = []
    iterator = full
    if tqdm is not None:
        iterator = tqdm(full, desc="grid", unit="combo")

    for item in iterator:
        (m_deep, m_mid, m_otm), put_t = item

        legs = build_legs_short_call_thirds(
            args.contracts_per_bucket, m_deep, m_mid, m_otm, with_put_thirds=put_t
        )
        try:
            _, summary = _run_backtest_clean(
                close,
                legs,
                args.dte_days,
                args.rebalance_every,
                args.r_free,
                args.vol_window,
                args.fee_per_contract,
                args.basis_drag_bps,
                args.gap_slippage_bps,
                args.share_hedge_ratio,
                args.initial_capital,
            )
        except Exception as e:
            if tqdm is None:
                print(f"skip {m_deep},{m_mid},{m_otm},{put_t}: {e}", file=sys.stderr)
            continue

        results.append(ComboResult(m_deep, m_mid, m_otm, put_t, summary))

    if not results:
        raise SystemExit("无有效回测结果")

    obj = args.objective

    def key_fn(cr: ComboResult) -> float:
        v = score(cr.summary, obj)
        if v != v:  # nan
            return float("-inf")
        return v

    results.sort(key=key_fn, reverse=True)
    top = results[: args.top]

    rows = []
    for rank, cr in enumerate(top, 1):
        s = cr.summary
        row = {
            "rank": rank,
            "sc_m_deep": cr.m_deep,
            "sc_m_mid": cr.m_mid,
            "sc_m_otm": cr.m_otm,
            "lp_m_deep": cr.put_triple[0] if cr.put_triple else None,
            "lp_m_mid": cr.put_triple[1] if cr.put_triple else None,
            "lp_m_otm": cr.put_triple[2] if cr.put_triple else None,
            "cagr": s.get("cagr"),
            "ann_vol": s.get("ann_vol"),
            "sharpe": s.get("sharpe"),
            "calmar": calmar(s),
            "max_drawdown": s.get("max_drawdown"),
            "end_equity": s.get("end_equity"),
            "years": s.get("years"),
        }
        rows.append(row)

    df_out = pd.DataFrame(rows)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    print(f"=== 目标: {obj}（越大越好）| 组合数 evaluated: {len(results)} ===")
    print(df_out.to_string(index=False))

    print(
        "\n说明: 使用与回测脚本相同的简化模型（欧式期末、历史σ、无提前指派）。"
        "「最佳」仅在该模型与样本内有效，实盘需再校验。"
    )

    if args.out_csv:
        df_out.to_csv(args.out_csv, index=False)
        print(f"\n已写入 {args.out_csv}")


if __name__ == "__main__":
    main()
