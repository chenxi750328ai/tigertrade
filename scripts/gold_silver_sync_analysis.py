#!/usr/bin/env python3
"""
黄金与白银期货同步性对比分析
数据源：yfinance (GC=F 黄金, SI=F 白银)；请求间隔 15 秒避免限流。
分析：日线/分钟线收益相关、高低点同步、领先滞后。
若仍限流：可稍后重试或从其它环境运行；成功一次后可用 output/gold_silver_daily_aligned.csv 做离线分析。
"""
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    print("请安装: pip install yfinance")
    raise

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _normalize(df: pd.DataFrame, symbol_label: str = "") -> pd.DataFrame:
    """统一索引与列名"""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    if df.index.nlevels > 1:
        df.index = df.index.get_level_values(0)
    # 多标的时列为 MultiIndex (Open, Close, ...) 或 (symbol, Open) 等，压平为 symbol_open 或 open
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(x).lower() for x in c).replace(" ", "_") for c in df.columns]
    else:
        df = df.rename(columns={c: c.lower() if isinstance(c, str) else str(c).lower() for c in df.columns})
    return df


def fetch_daily(symbol: str, period: str = "1y") -> pd.DataFrame:
    """获取日线 OHLCV"""
    d = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True, threads=False, group_by="ticker")
    if d is None or (isinstance(d, tuple) and len(d) == 2):
        return pd.DataFrame()
    if isinstance(d, tuple):
        d = d[0] if len(d) else pd.DataFrame()
    return _normalize(d)


def fetch_minute(symbol: str, period: str = "5d") -> pd.DataFrame:
    """获取分钟线（yfinance 约 5–7 天）"""
    d = yf.download(symbol, period=period, interval="1m", progress=False, auto_adjust=True, threads=False, group_by="ticker")
    if d is None or (isinstance(d, tuple) and len(d) == 2):
        return pd.DataFrame()
    if isinstance(d, tuple):
        d = d[0] if len(d) else pd.DataFrame()
    if d.empty:
        return d
    return _normalize(d)


# 每次请求间隔（秒）。调大（如 15～30）可进一步降低限流概率
FETCH_INTERVAL_SEC = 15


def align_and_analyze_daily(gold: pd.DataFrame, silver: pd.DataFrame) -> dict:
    """日线：收益相关、高低点相关、领先滞后"""
    common = gold.join(silver, how="inner", lsuffix="_g", rsuffix="_s")
    if common.empty or len(common) < 10:
        return {}
    # 收益
    ret_g = common["close_g"].pct_change().dropna()
    ret_s = common["close_s"].pct_change().dropna()
    ret_g, ret_s = ret_g.align(ret_s, join="inner")
    ret_g = ret_g.dropna()
    ret_s = ret_s.loc[ret_g.index].dropna()
    idx = ret_g.index.intersection(ret_s.index)
    ret_g, ret_s = ret_g.loc[idx], ret_s.loc[idx]
    corr_return = float(ret_g.corr(ret_s)) if len(ret_g) > 1 else 0.0
    # 日高/日低（水平相关）
    high_g, high_s = common["high_g"], common["high_s"]
    low_g, low_s = common["low_g"], common["low_s"]
    corr_high = float(high_g.corr(high_s)) if high_g.std() > 0 and high_s.std() > 0 else 0.0
    corr_low = float(low_g.corr(low_s)) if low_g.std() > 0 and low_s.std() > 0 else 0.0
    # 日高/低 的“同步性”：同一天是否同向（涨跌日一致）
    daily_up_g = (common["close_g"] > common["open_g"]).astype(int)
    daily_up_s = (common["close_s"] > common["open_s"]).astype(int)
    same_direction = (daily_up_g == daily_up_s).mean()
    # 领先滞后 1 日
    ret_s_next = ret_s.shift(-1).dropna()
    ret_g_lead_1d = ret_g.reindex(ret_s_next.index).dropna()
    ret_s_next = ret_s_next.reindex(ret_g_lead_1d.index).dropna()
    ret_g_lead_1d, ret_s_next = ret_g_lead_1d.align(ret_s_next, join="inner")
    gold_lead_1d_corr = float(ret_g_lead_1d.corr(ret_s_next)) if len(ret_g_lead_1d) > 1 else 0.0
    ret_g_next = ret_g.shift(-1).dropna()
    ret_s_lead_1d = ret_s.reindex(ret_g_next.index).dropna()
    ret_g_next = ret_g_next.reindex(ret_s_lead_1d.index).dropna()
    ret_s_lead_1d, ret_g_next = ret_s_lead_1d.align(ret_g_next, join="inner")
    silver_lead_1d_corr = float(ret_s_lead_1d.corr(ret_g_next)) if len(ret_s_lead_1d) > 1 else 0.0
    return {
        "corr_return": float(corr_return),
        "corr_high": float(corr_high),
        "corr_low": float(corr_low),
        "same_direction_pct": float(same_direction),
        "gold_lead_1d_corr": gold_lead_1d_corr,
        "silver_lead_1d_corr": silver_lead_1d_corr,
        "n_days": len(common),
    }


def align_and_analyze_minute(gold: pd.DataFrame, silver: pd.DataFrame) -> dict:
    """分钟线：收益相关、同分钟高低点是否同步"""
    if gold.empty or silver.empty:
        return {}
    # 按时间对齐
    common_idx = gold.index.intersection(silver.index)
    if len(common_idx) < 30:
        return {}
    gold_a = gold.loc[common_idx].sort_index()
    silver_a = silver.loc[common_idx].sort_index()
    ret_g = gold_a["close"].pct_change().dropna()
    ret_s = silver_a["close"].pct_change().dropna()
    common_ret_idx = ret_g.index.intersection(ret_s.index)
    ret_g, ret_s = ret_g.loc[common_ret_idx], ret_s.loc[common_ret_idx]
    corr_return_1m = ret_g.corr(ret_s)
    # 同分钟：高点是否同向（当根K线收涨则算“高点相对前一根”更复杂，这里用收益相关已能说明同步）
    # 额外：同一分钟内的 high 与 high 的相关系数（水平）
    high_g = gold_a.loc[common_ret_idx, "high"] if "high" in gold_a.columns else gold_a.loc[common_ret_idx, "close"]
    high_s = silver_a.loc[common_ret_idx, "high"] if "high" in silver_a.columns else silver_a.loc[common_ret_idx, "close"]
    corr_high_1m = high_g.corr(high_s)
    return {
        "corr_return_1m": corr_return_1m,
        "corr_high_1m": corr_high_1m,
        "n_bars": len(common_ret_idx),
    }


def main():
    import time
    interval = FETCH_INTERVAL_SEC
    print("正在慢速获取黄金(GC=F)、白银(SI=F)数据（每次请求间隔 {} 秒，避免限流）…".format(interval))
    time.sleep(2)  # 首次请求前稍等，避免连续运行脚本时立刻撞限流
    # 逐个拉取，中间等待，降低限流概率
    gold_d = fetch_daily("GC=F", period="1y")
    time.sleep(interval)
    print("  黄金日线 OK，等待 {} 秒后拉取白银…".format(interval))
    silver_d = fetch_daily("SI=F", period="1y")
    time.sleep(interval)
    print("  白银日线 OK，等待 {} 秒后拉取黄金分钟…".format(interval))
    gold_m = fetch_minute("GC=F", period="5d")
    time.sleep(interval)
    print("  黄金分钟 OK，等待 {} 秒后拉取白银分钟…".format(interval))
    silver_m = fetch_minute("SI=F", period="5d")

    if gold_d.empty or silver_d.empty:
        print("日线数据获取失败（可能被限流）。使用近期合成数据演示分析流程…")
        # 演示模式：生成高度相关的合成日线，便于验证分析逻辑
        n = 252
        np.random.seed(42)
        dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B")
        base = 2600 + np.cumsum(np.random.randn(n) * 3)
        gold_d = pd.DataFrame({"open": base, "high": base + np.abs(np.random.randn(n) * 5), "low": base - np.abs(np.random.randn(n) * 5), "close": base + np.random.randn(n) * 4}, index=dates)
        silver_d = gold_d.copy()
        silver_d["open"] = silver_d["open"] * 0.04 + np.random.randn(n) * 0.5
        silver_d["high"] = silver_d["high"] * 0.04 + np.random.randn(n) * 0.3
        silver_d["low"] = silver_d["low"] * 0.04 + np.random.randn(n) * 0.3
        silver_d["close"] = silver_d["close"] * 0.04 + np.random.randn(n) * 0.3
        gold_d.index.name = "Date"
        silver_d.index.name = "Date"
        gold_m = pd.DataFrame()
        silver_m = pd.DataFrame()
        print("(提示: 稍后重试或设置代理后可获取真实 GC=F / SI=F 数据)\n")

    print("\n========== 日线同步性（约 1 年）==========")
    daily_res = align_and_analyze_daily(gold_d, silver_d)
    if daily_res:
        print(f"  样本天数: {daily_res['n_days']}")
        print(f"  日收益相关系数:     {daily_res['corr_return']:.4f}")
        print(f"  日最高价相关系数:   {daily_res['corr_high']:.4f}")
        print(f"  日最低价相关系数:   {daily_res['corr_low']:.4f}")
        print(f"  同日涨跌一致比例:   {daily_res['same_direction_pct']:.2%}")
        print(f"  黄金领先 1 日(与次日白银收益相关): {daily_res['gold_lead_1d_corr']:.4f}")
        print(f"  白银领先 1 日(与次日黄金收益相关): {daily_res['silver_lead_1d_corr']:.4f}")

    if not gold_m.empty and not silver_m.empty:
        print("\n========== 分钟线同步性（约 5 天）==========")
        minute_res = align_and_analyze_minute(gold_m, silver_m)
        if minute_res:
            print(f"  对齐分钟数: {minute_res['n_bars']}")
            print(f"  分钟收益相关系数:   {minute_res['corr_return_1m']:.4f}")
            print(f"  分钟最高价相关系数: {minute_res['corr_high_1m']:.4f}")
    else:
        print("\n(分钟数据不可用或不足，仅做日线分析)")

    # 简要结论
    print("\n========== 结论 ==========")
    if daily_res:
        c = daily_res["corr_return"]
        if c > 0.7:
            print("  黄金与白银日收益高度正相关，走势高度同步。")
        elif c > 0.4:
            print("  黄金与白银日收益中度正相关，多数时段同向。")
        else:
            print("  黄金与白银日收益相关性较弱。")
        if daily_res.get("gold_lead_1d_corr", 0) > daily_res.get("silver_lead_1d_corr", 0):
            print("  领先滞后上，黄金对次日白银的预测力略强（黄金略领先）。")
        else:
            print("  领先滞后上，白银对次日黄金的预测力略强（白银略领先）。")

    # 保存日线对齐数据供后续使用
    common = gold_d.join(silver_d, how="inner", lsuffix="_g", rsuffix="_s")
    if not common.empty:
        out_path = OUT_DIR / "gold_silver_daily_aligned.csv"
        common.to_csv(out_path, encoding="utf-8-sig")
        print(f"\n已保存对齐日线: {out_path}")
    return daily_res, gold_m.empty or silver_m.empty and {} or align_and_analyze_minute(gold_m, silver_m)


if __name__ == "__main__":
    main()
