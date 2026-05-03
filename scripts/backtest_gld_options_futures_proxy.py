#!/usr/bin/env python3
"""
回测「证券户 GLD 期权腿 + 期货户黄金多头」的**简化代理模型**（长期年化等统计）。

重要简化（读后再用）：
- 标的：**GLD 复权收盘价**代理金价敞口；**不**单独拉 MGC 连续合约（避免换月拼接复杂度）。
  用 ``--basis-drag-bps`` 近似 GLD↔期货 基差 + 换月摩擦。
- 期权：按 **欧式** 在持有期末结算内在价值；**不**模拟提前指派与跳空成交滑点。
  用 ``--gap-slippage-bps`` 做一次性保守扣减（可调大模拟尾部）。
- 权利金：用 **Black-Scholes**，sigma 用过去 ``--vol-window`` 日对数收益年化波动率（历史波动，非市场 IV）。

依赖: pip install yfinance scipy；老虎拉线另需 ``tigeropen`` 与有效 ``openapicfg`` properties。

示例（接近你口述结构，按「行权价相对现价比例」每 5 个交易日重开）::

    python scripts/backtest_gld_options_futures_proxy.py \\
        --start 2010-01-01 \\
        --legs "SC:0.0465:3,SC:0.0233:3,SC:-0.0116:4,LP:0.0233:3" \\
        --rebalance-every 5 --vol-window 20 --r-free 0.04 \\
        --basis-drag-bps 25 --fee-per-contract 0.65 --gap-slippage-bps 5

legs 格式（逗号分隔）::
    SC:<m>:<n>  卖 n 张 Call，行权价 K = S*(1+m)，m 可负（价内短 Call）
    LP:<m>:<n>  买 n 张 Put，  行权价 K = S*(1-m)，m>0 表示价外幅度

期货多头代理：持有 ``--share-hedge-ratio * 100 * 总期权张数`` 的 GLD 等价份额（默认 1.0 = 与「张数×100 股」同量级名义）。

三档 moneyness（深实值/实值/虚值）各占 1/3 张数的**参数搜索**：见 ``scripts/optimize_gld_options_moneyness.py``。

**三档卖 Call 预设（口头行权价 400/430/450 在 S_ref≈430 时）**（各占相同张数）::

    # 数据：二选一 —— 本地 CSV，或老虎（需 TIGEROPEN_PROPS）
    python scripts/backtest_gld_options_futures_proxy.py --preset-near430 --tiger-props \"$TIGEROPEN_PROPS\" \\
        --align-15d-cycle --print-anchor-extrinsic --calibrate-sigma-for-extrinsic \\
        --extrinsic-target 4 --extrinsic-horizon-days 15 --contracts-per-bucket 2

**滚动说明（重要）**：``m`` 只由「你当时报价用的美元行权价 ÷ 报价参考价 S_ref」换算一次并**固定**；**每期重开**时用**当日 GLD 收盘价** ``S0`` 算 ``K=S0*(1+m)``，因此行权价**随 GLD 涨跌而滚动**，**不是**把 GLD 钉在 430。``--anchor-spot`` 即该 S_ref，仅影响 m 的换算与外在价值打印，**不影响**「用当日 S 定价」。

若你观察的期权净收约 **$4/股/15 天**，可 ``--align-15d-cycle``，并用 ``--calibrate-sigma-for-extrinsic`` 在 S_ref 下反推 σ（仅打印；回测仍用历史 σ）。

先拉 CSV：``python scripts/fetch_gld_daily_tiger.py -o data/gld_daily.csv``。
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("请先安装: pip install yfinance", file=sys.stderr)
    sys.exit(1)

from scipy.stats import norm


def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def _bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


@dataclass
class Leg:
    kind: str  # "SC" short call, "LP" long put
    moneyness: float  # SC: K=S*(1+m), LP: K=S*(1-m) with m>0 OTM put
    n: int


_LEG_RE = re.compile(r"^(SC|LP):([+-]?\d*\.?\d+):(\d+)$")


def legs_from_anchor_short_calls(
    anchor_s: float, strikes: Tuple[float, float, float], n_per_bucket: int
) -> List[Leg]:
    """三档卖 Call：固定 m_i = K_i/S_ref - 1（S_ref=报价参考价）；回测里每期用当日 S 算 K=S*(1+m)。"""
    if anchor_s <= 0:
        raise ValueError("S_ref 必须 > 0")
    return [Leg("SC", k / anchor_s - 1.0, n_per_bucket) for k in strikes]


def legs_from_anchor_with_long_put(
    anchor_s: float,
    strikes_sc: Tuple[float, float, float],
    n_sc_per: int,
    put_strike: float,
    n_put: int,
) -> List[Leg]:
    """三档 Short Call + Long Put。LP: m_put=1-K_put/S_ref；行权价仍按每期收盘 S 为 K=S*(1-m)。"""
    legs = legs_from_anchor_short_calls(anchor_s, strikes_sc, n_sc_per)
    m_put = 1.0 - put_strike / anchor_s
    legs.append(Leg("LP", m_put, n_put))
    return legs


def _extrinsic_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return max(_bs_call(S, K, T, r, sigma) - max(S - K, 0.0), 0.0)


def _extrinsic_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return max(_bs_put(S, K, T, r, sigma) - max(K - S, 0.0), 0.0)


def extrinsic_total_usd(legs: List[Leg], S: float, d_calendar: int, r_free: float, sigma: float) -> float:
    """各腿外在价值（美元）合计（含张数×100 股）。"""
    T = max(int(d_calendar), 1) / 365.0
    tot = 0.0
    for leg in legs:
        K = strike_for_leg(S, leg)
        if leg.kind == "SC":
            tot += _extrinsic_call(S, K, T, r_free, sigma) * 100.0 * leg.n
        else:
            tot += _extrinsic_put(S, K, T, r_free, sigma) * 100.0 * leg.n
    return tot


def extrinsic_per_share_usd(
    legs: List[Leg], S: float, d_calendar: int, r_free: float, sigma: float
) -> float:
    shares = sum(100.0 * leg.n for leg in legs)
    if shares <= 0:
        return 0.0
    return extrinsic_total_usd(legs, S, d_calendar, r_free, sigma) / shares


def calibrate_sigma_for_extrinsic_per_share(
    legs: List[Leg],
    S: float,
    d_calendar: int,
    r_free: float,
    target_per_share: float,
    lo: float = 0.06,
    hi: float = 1.8,
    iterations: int = 60,
) -> float:
    """二分 σ，使 extrinsic_per_share ≈ target（通常随 σ 单调增）。"""

    def g(sig: float) -> float:
        return extrinsic_per_share_usd(legs, S, d_calendar, r_free, sig) - target_per_share

    a, b = lo, hi
    ga, gb = g(a), g(b)
    if ga > 0:
        a = 0.03
        ga = g(a)
    if gb < 0:
        while gb < 0 and b < 3.0:
            b += 0.2
            gb = g(b)
    if ga > 0 or gb < 0:
        return float("nan")

    for _ in range(iterations):
        m = 0.5 * (a + b)
        gm = g(m)
        if abs(gm) < 1e-4:
            return m
        if gm < 0:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def print_anchor_extrinsic(
    legs: List[Leg],
    anchor_s: float,
    dte_days: int,
    r_free: float,
    sigma: float,
    *,
    extrinsic_target_per_share: float = 4.0,
    extrinsic_compare_horizon_days: int = 15,
) -> None:
    """在锚定价下打印各腿外在价值（时间价值）及合计（每股标的近似）；默认同时给 7d 与 15d 便于对照。"""
    horizons = sorted({7, 15, max(1, int(dte_days)), int(extrinsic_compare_horizon_days)})
    print("\n=== 锚定价外在价值（欧式 BS，仅作量级参考）===")
    print(f"  S_ref（外在打印参考价）={anchor_s:.2f}, sigma={sigma:.3f}, r={r_free:.3f}")
    print(
        "  说明：深实值卖 Call 外在价值很小；近价档贡献大。"
        + f"经验目标约 **${extrinsic_target_per_share:.1f}/股/{extrinsic_compare_horizon_days}天**"
        + "（卖方净收与下表纯外在值会因 IV、价差、是否带 Put 等有偏差）。"
    )
    for d in horizons:
        T = d / 365.0
        print(f"\n  --- 持有期 {d} 日历天 ---")
        tot_ext = 0.0
        shares_covered = 0.0
        for leg in legs:
            K = strike_for_leg(anchor_s, leg)
            if leg.kind == "SC":
                ext = _extrinsic_call(anchor_s, K, T, r_free, sigma) * 100.0 * leg.n
                tag = "卖Call"
            else:
                ext = _extrinsic_put(anchor_s, K, T, r_free, sigma) * 100.0 * leg.n
                tag = "买Put"
            tot_ext += ext
            shares_covered += 100.0 * leg.n
            print(f"    {tag} K={K:.1f} x{leg.n}: 外在价值合计 ${ext:,.2f}")
        if shares_covered > 0:
            per_share = tot_ext / shares_covered
            print(f"    合计 ${tot_ext:,.2f} → 摊薄 ≈ ${per_share:.2f}/股（标的）")
            if d == extrinsic_compare_horizon_days:
                print(
                    f"    与经验目标「约 ${extrinsic_target_per_share:.1f}/股/{d}天」对照："
                    f"差 {per_share - extrinsic_target_per_share:+.2f} $/股"
                    + "（可用 --calibrate-sigma-for-extrinsic 反推 σ，或调 --anchor-sigma）"
                )


def parse_legs(s: str) -> List[Leg]:
    out: List[Leg] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        m = _LEG_RE.match(part)
        if not m:
            raise ValueError(f"无法解析 leg: {part!r}，期望如 SC:0.05:3 或 LP:0.02:3")
        knd, mny, n = m.group(1), float(m.group(2)), int(m.group(3))
        if n <= 0:
            raise ValueError(f"张数必须>0: {part}")
        out.append(Leg(kind=knd, moneyness=mny, n=n))
    if not out:
        raise ValueError("至少一条 leg")
    return out


def strike_for_leg(S: float, leg: Leg) -> float:
    if leg.kind == "SC":
        return S * (1.0 + leg.moneyness)
    if leg.kind == "LP":
        return S * (1.0 - leg.moneyness)
    raise ValueError(leg.kind)


def download_gld(start: str, end: str) -> pd.DataFrame:
    try:
        df = yf.download("GLD", start=start, end=end, progress=False, auto_adjust=True)
    except Exception as e:
        raise RuntimeError(f"yfinance 下载失败: {e}。可改用 --csv 指定本地 OHLC。") from e
    if df.empty:
        raise RuntimeError(
            "GLD 数据为空（可能被限流或日期无数据）。请稍后重试或使用 --csv 本地文件。"
        )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str.capitalize)
    if "Close" not in df.columns:
        df = df.rename(columns={"Adj close": "Close"})
    return df[["Close"]].dropna()


def _normalize_tiger_bars_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise RuntimeError("老虎 API 返回空 K 线（检查标的、权限、properties）")
    d = df.copy()
    if "time" in d.columns:
        dt = pd.to_datetime(d["time"], unit="ms", utc=True).dt.tz_convert(None)
        d = d.assign(_dt=dt).set_index("_dt")
    elif isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors="coerce")
    else:
        d.index = pd.to_datetime(d.index, errors="coerce")
    d = d[~d.index.duplicated(keep="first")].sort_index()
    cc = "close" if "close" in d.columns else "Close"
    if cc not in d.columns:
        raise RuntimeError(f"老虎 K 线缺少 close，列名为: {list(d.columns)}")
    out = pd.DataFrame({"Close": pd.to_numeric(d[cc], errors="coerce")})
    return out.dropna()


def download_gld_tiger(
    props_path: str,
    symbol: str = "GLD",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """使用老虎 ``QuoteClient.get_bars``（必要时 ``get_bars_by_page``）拉日线收盘价。"""
    from tigeropen.common.consts import BarPeriod, QuoteRight
    from tigeropen.quote.quote_client import QuoteClient
    from tigeropen.tiger_open_config import TigerOpenClientConfig

    path = os.path.expanduser(props_path)
    if not path or not os.path.isfile(path):
        raise RuntimeError(
            f"老虎配置文件不存在: {path!r}。请设置 TIGEROPEN_PROPS 或传入 --tiger-props。"
        )
    cfg = TigerOpenClientConfig(props_path=path)
    qc = QuoteClient(cfg)
    begin = start if start else -1
    fin = end if end else -1
    df = qc.get_bars(
        symbol,
        period=BarPeriod.DAY,
        begin_time=begin,
        end_time=fin,
        limit=12000,
        right=QuoteRight.BR,
    )
    if df is None or len(df) < 80:
        try:
            df = qc.get_bars_by_page(
                symbol,
                period=BarPeriod.DAY,
                begin_time=begin,
                end_time=fin,
                total=20000,
                page_size=500,
                right=QuoteRight.BR,
            )
        except Exception as e:
            raise RuntimeError(f"老虎分页拉线失败: {e}") from e
    return _normalize_tiger_bars_df(df)


def load_close_csv(path: str) -> pd.DataFrame:
    """本地 CSV：需含 Date 与 Close（或 Adj Close），Date 解析为索引。"""
    raw = pd.read_csv(path)
    cols = {c.lower(): c for c in raw.columns}
    date_col = cols.get("date") or cols.get("datetime")
    close_col = cols.get("close") or cols.get("adj close") or cols.get("adj_close")
    if not date_col or not close_col:
        raise ValueError("CSV 需包含 Date 与 Close（或 Adj Close）列")
    idx = pd.to_datetime(raw[date_col])
    vals = pd.to_numeric(raw[close_col], errors="coerce").to_numpy()
    # 必须用 .values 对齐索引；否则 pandas 按标签对齐会把整列对齐成 NaN
    out = pd.DataFrame({"Close": vals}, index=idx)
    return out.dropna()


def annualized_vol(close: pd.Series, window: int, asof_idx: int) -> float:
    if asof_idx < window + 1:
        return 0.20
    sl = close.iloc[asof_idx - window : asof_idx + 1].astype(float)
    r = np.log(sl / sl.shift(1)).dropna()
    if len(r) < 5:
        return 0.20
    return float(r.std() * math.sqrt(252.0))


def _run_backtest_clean(
    close: pd.Series,
    legs: List[Leg],
    dte_days: int,
    rebalance_every: int,
    r_free: float,
    vol_window: int,
    fee_per_contract: float,
    basis_drag_bps: float,
    gap_slippage_bps: float,
    share_hedge_ratio: float,
    initial_capital: float,
) -> Tuple[pd.Series, dict]:
    idx = close.index
    # BS 到期：日历天（默认 7）；持有期按 rebalance_every 个交易日结算内在价值
    T_year = max(dte_days, 1) / 365.0
    daily_basis = basis_drag_bps / 10000.0 / 252.0
    daily_gap = gap_slippage_bps / 10000.0 / 252.0

    total_contracts = sum(leg.n for leg in legs)
    stock_shares = 100.0 * total_contracts * share_hedge_ratio

    i0 = vol_window + 2
    if i0 >= len(close):
        raise ValueError("数据太短")

    S = float(close.iloc[i0])
    cash = float(initial_capital) - stock_shares * S

    dates = []
    equities = []

    i = i0
    n = len(close)
    rebalance_count = 0

    # 期初净值点
    dates.append(idx[i0])
    equities.append(float(initial_capital))

    while i < n - 1:
        S0 = float(close.iloc[i])
        sigma = max(0.08, annualized_vol(close, vol_window, i))
        fees_open = fee_per_contract * total_contracts
        premium_net = 0.0
        strikes: List[Tuple[str, float, int]] = []
        for leg in legs:
            K = strike_for_leg(S0, leg)
            strikes.append((leg.kind, K, leg.n))
            if leg.kind == "SC":
                premium_net += _bs_call(S0, K, T_year, r_free, sigma) * 100 * leg.n
            else:
                premium_net -= _bs_put(S0, K, T_year, r_free, sigma) * 100 * leg.n

        cash += premium_net - fees_open
        rebalance_count += 1

        j_end = min(i + rebalance_every, n - 1)
        for j in range(i + 1, j_end + 1):
            S_prev = float(close.iloc[j - 1])
            S_j = float(close.iloc[j])
            notional = stock_shares * S_prev
            cash -= notional * (daily_basis + daily_gap)
            mv_stock = stock_shares * S_j
            eq = cash + mv_stock
            dates.append(idx[j])
            equities.append(eq)

        S_T = float(close.iloc[j_end])
        settle = 0.0
        for kind, K, nc in strikes:
            if kind == "SC":
                settle -= max(S_T - K, 0.0) * 100 * nc
            else:
                settle += max(K - S_T, 0.0) * 100 * nc
        cash += settle - fee_per_contract * total_contracts

        # 股票部分：持有 stock_shares 不变（对冲规模固定）；现金吸收期权现金流
        i = j_end

    ser = pd.Series(equities, index=pd.DatetimeIndex(dates))
    total_return = ser.iloc[-1] / ser.iloc[0] - 1.0 if len(ser) > 1 else 0.0
    years = (ser.index[-1] - ser.index[0]).days / 365.25
    cagr = (ser.iloc[-1] / ser.iloc[0]) ** (1.0 / years) - 1.0 if years > 0.1 else float("nan")
    daily = ser.pct_change().dropna()
    vol = float(daily.std() * math.sqrt(252)) if len(daily) > 5 else float("nan")
    sharpe = float((daily.mean() * 252 - r_free) / (daily.std() * math.sqrt(252))) if daily.std() > 1e-12 else float("nan")
    roll_max = ser.cummax()
    dd = (ser / roll_max - 1.0).min()

    summary = {
        "rebalances": rebalance_count,
        "years": years,
        "total_return": float(total_return),
        "cagr": float(cagr),
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": float(dd),
        "end_equity": float(ser.iloc[-1]),
    }
    return ser, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="GLD 期权+期货代理 简化回测")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None, help="默认今天（仅 yfinance 下载时过滤）")
    ap.add_argument(
        "--csv",
        default=None,
        help="本地 GLD 日线 CSV（含 Date,Close）；若指定则优先于老虎/yfinance",
    )
    ap.add_argument(
        "--tiger-props",
        default=os.environ.get("TIGEROPEN_PROPS", os.environ.get("TIGER_PROPS", "")),
        help="老虎 OpenAPI properties 路径；若给出且未用 --csv，则用 QuoteClient 拉日线",
    )
    ap.add_argument(
        "--tiger-symbol",
        default="GLD",
        help="老虎标的（默认 GLD）；无权限时可按券商文档改如带市场前缀",
    )
    ap.add_argument(
        "--legs",
        default=None,
        help='手工腿：如 SC:0.0465:3,SC:0.0233:3,LP:0.0233:3。与 --preset-near430 二选一',
    )
    ap.add_argument(
        "--preset-near430",
        action="store_true",
        help="三档卖Call：默认美元行权 400/430/450 按 --anchor-spot(S_ref) 换成固定 m；每期用**当日收盘**S 算 K=S(1+m) 滚动",
    )
    ap.add_argument(
        "--anchor-spot",
        type=float,
        default=430.0,
        help="报价参考价 S_ref：只用于「口头美元行权价→m」及外在打印；**每期重开仍用当日 GLD 收盘 S**",
    )
    ap.add_argument(
        "--preset-strikes-sc",
        default="400,430,450",
        help="预设三档 Short Call 行权价（美元），逗号分隔",
    )
    ap.add_argument(
        "--contracts-per-bucket",
        type=int,
        default=2,
        help="--preset-near430 下每档 Short Call 张数（三档共 3×该数）",
    )
    ap.add_argument(
        "--preset-long-put-strike",
        type=float,
        default=None,
        help="可选：同时买该行权价的 Put（张数=--put-contracts；m 用 S_ref 换算，K 仍按每期 S）",
    )
    ap.add_argument("--put-contracts", type=int, default=2, help="与 --preset-long-put-strike 搭配")
    ap.add_argument(
        "--print-anchor-extrinsic",
        action="store_true",
        help="在锚定价下用 BS 打印各腿外在价值（时间价值）合计",
    )
    ap.add_argument(
        "--anchor-sigma",
        type=float,
        default=0.22,
        help="打印外在价值时用的年化波动率（非回测路径上的历史 σ）",
    )
    ap.add_argument(
        "--align-15d-cycle",
        action="store_true",
        help="将 --dte-days 设为 15、--rebalance-every 设为 10（约两周持有+重开）",
    )
    ap.add_argument(
        "--calibrate-sigma-for-extrinsic",
        action="store_true",
        help="按当前腿在锚定价下反推 σ，使外在价值摊薄接近 --extrinsic-target（用于 --print-anchor-extrinsic）",
    )
    ap.add_argument(
        "--extrinsic-target",
        type=float,
        default=4.0,
        help="与 --calibrate-sigma-for-extrinsic 搭配：目标 $/股",
    )
    ap.add_argument(
        "--extrinsic-horizon-days",
        type=int,
        default=15,
        help="校准与对照用的日历天（默认 15）",
    )
    ap.add_argument(
        "--dte-days",
        type=int,
        default=7,
        help="Black-Scholes 用的年化时间：日历天（可与 --rebalance-every 不同）",
    )
    ap.add_argument("--rebalance-every", type=int, default=5, help="每多少**交易日**重开期权腿")
    ap.add_argument("--vol-window", type=int, default=20)
    ap.add_argument("--r-free", type=float, default=0.04)
    ap.add_argument("--fee-per-contract", type=float, default=0.65, help="每张单边美元，开+平各扣一次（简化）")
    ap.add_argument("--basis-drag-bps", type=float, default=25.0, help="GLD↔期货+换月 年化扣减 bps")
    ap.add_argument("--gap-slippage-bps", type=float, default=5.0, help="指派/跳空保守年化扣减 bps")
    ap.add_argument(
        "--share-hedge-ratio",
        type=float,
        default=1.0,
        help="GLD 股数 = 100*总张数*ratio；1.0 表示与期权张数同量级名义",
    )
    ap.add_argument("--initial-capital", type=float, default=100_000.0)
    ap.add_argument(
        "--compare-bnh",
        action="store_true",
        help="同时打印同股数纯持有 GLD 的期末净值（不含期权、不含费用）",
    )
    args = ap.parse_args()

    if args.align_15d_cycle:
        args.dte_days = 15
        args.rebalance_every = 10

    if args.preset_near430:
        strikes_sc = tuple(float(x.strip()) for x in args.preset_strikes_sc.split(",") if x.strip())
        if len(strikes_sc) != 3:
            raise SystemExit("--preset-strikes-sc 必须恰好 3 个行权价")
        if args.preset_long_put_strike is not None:
            legs = legs_from_anchor_with_long_put(
                args.anchor_spot,
                strikes_sc,  # type: ignore[arg-type]
                args.contracts_per_bucket,
                args.preset_long_put_strike,
                args.put_contracts,
            )
        else:
            legs = legs_from_anchor_short_calls(
                args.anchor_spot,
                strikes_sc,  # type: ignore[arg-type]
                args.contracts_per_bucket,
            )
    elif args.legs:
        legs = parse_legs(args.legs)
    else:
        raise SystemExit("请指定 --legs 或 --preset-near430")
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    if args.csv:
        df = load_close_csv(args.csv)
        close = df["Close"].astype(float)
        close = close.sort_index()
        if args.start:
            close = close[close.index >= pd.Timestamp(args.start)]
        if args.end:
            close = close[close.index <= pd.Timestamp(args.end)]
    elif (args.tiger_props or "").strip():
        try:
            df = download_gld_tiger(
                (args.tiger_props or "").strip(),
                args.tiger_symbol.strip() or "GLD",
                args.start,
                end,
            )
        except Exception as e:
            raise SystemExit(f"老虎拉线失败: {e}") from e
        close = df["Close"].astype(float).sort_index()
        if args.start:
            close = close[close.index >= pd.Timestamp(args.start)]
        if args.end:
            close = close[close.index <= pd.Timestamp(args.end)]
    else:
        df = download_gld(args.start, end)
        close = df["Close"].astype(float)

    if len(close) < 50:
        raise SystemExit(f"有效收盘价不足 50 根（当前 {len(close)}），请拉长样本或换 --csv")

    if args.calibrate_sigma_for_extrinsic:
        sig_star = calibrate_sigma_for_extrinsic_per_share(
            legs,
            args.anchor_spot,
            args.extrinsic_horizon_days,
            args.r_free,
            args.extrinsic_target,
        )
        if math.isnan(sig_star):
            print(
                "\n⚠️ --calibrate-sigma-for-extrinsic 未收敛（尝试放宽腿或检查目标）；"
                "仍使用 --anchor-sigma",
                file=sys.stderr,
            )
        else:
            print(
                f"\n>>> 校准 σ：使 {args.extrinsic_horizon_days}d 外在摊薄 ≈ ${args.extrinsic_target:.2f}/股 → "
                f"anchor_sigma = {sig_star:.4f}\n"
            )
            args.anchor_sigma = sig_star

    if args.print_anchor_extrinsic:
        print_anchor_extrinsic(
            legs,
            args.anchor_spot,
            args.dte_days,
            args.r_free,
            args.anchor_sigma,
            extrinsic_target_per_share=args.extrinsic_target,
            extrinsic_compare_horizon_days=args.extrinsic_horizon_days,
        )

    ser, summary = _run_backtest_clean(
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

    print("=== GLD 期权+期货代理 回测摘要（简化模型，见脚本顶部说明）===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("\n注: CAGR/波动等为历史模拟，非预测；未含税、未含提前指派。")

    if args.compare_bnh:
        i0 = args.vol_window + 2
        sh = 100.0 * sum(leg.n for leg in legs) * args.share_hedge_ratio
        s0 = float(close.iloc[i0])
        s1 = float(close.iloc[-1])
        bnh = args.initial_capital - sh * s0 + sh * s1
        years = summary["years"]
        bnh_cagr = (bnh / args.initial_capital) ** (1.0 / years) - 1.0 if years > 0.1 else float("nan")
        print("\n=== 对照: 同股数买入持有 GLD（无期权腿）===")
        print(f"  end_equity: {bnh:.2f}")
        print(f"  cagr: {bnh_cagr:.4f}")


if __name__ == "__main__":
    main()
