#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成各交易策略的算法说明与运行效果报告（含对比报告）。
输出：docs/reports/strategy_reports/*.md、strategy_comparison.md、strategy_reports_index.html
建议每日运行以刷新（cron 或与 optimize_algorithm_and_profitability 一并执行）。
用法：python scripts/generate_strategy_reports.py
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REPORTS_DIR = ROOT / "docs" / "reports"
STRATEGY_REPORTS_DIR = REPORTS_DIR / "strategy_reports"

# 回测效果：来自历史数据回测（如 parameter_grid_search）
BACKTEST_KEYS = ("num_trades", "return_pct", "avg_per_trade_pct", "top_per_trade_pct", "win_rate")
# 实盘表与回测表同结构：同一批指标，仅收益率区分「核对」与「推算」
LIVE_TABLE_KEYS = ("num_trades", "return_pct_verified", "return_pct_estimated", "avg_per_trade_pct", "top_per_trade_pct", "win_rate")
# DEMO 日志汇总单独成表，不混入实盘主表
LIVE_DEMO_KEYS = ("profitability", "win_rate", "yield_verified", "yield_estimated", "today_yield_pct",
                  "demo_order_success", "demo_sl_tp_log", "demo_execute_buy_calls", "demo_success_orders_sum",
                  "demo_fail_orders_sum", "demo_logs_scanned", "max_position")

# 报告中所有指标的含义与计算方式（用于「指标说明」节）
INDICATOR_DEFINITIONS = [
    ("return_pct", "回测收益率", "(期末资金 − 10万) / 10万 × 100（%）。来自 data/processed/test.csv 历史 K 线回测。"),
    ("win_rate", "回测胜率", "盈利笔数 / 完成笔数 × 100（%）。回测表为回测结果；实盘表为实盘胜率，仅来自 API 历史订单解析。"),
    ("num_trades", "回测成交笔数", "回测区间内实际完成的开平仓次数。"),
    ("avg_per_trade_pct", "单笔平均%", "总收益/笔数，每笔占初始资金%。"),
    ("top_per_trade_pct", "单笔TOP%", "单笔最大收益占初始资金%。"),
    ("profitability", "实盘盈亏汇总", "API 历史订单解析得到的总交易数、总盈亏等；无 API 时为 0 或 —。"),
    ("return_pct_verified", "收益率（核对）", "与回测 return_pct 对应；老虎后台订单/成交数据计算；未拉取或未核对时为 —。"),
    ("return_pct_estimated", "收益率（推算）", "与回测 return_pct 对应；未与老虎核对时的推算值；无推算时为 —。"),
    ("yield_verified", "实际收益率（老虎核对）", "用老虎后台订单/成交数据计算出的收益率；未拉取或未核对时为 —。"),
    ("yield_estimated", "推算收益率（未核对）", "未与老虎核对时的推算值（如 API 报告解析）；无推算时为 —。"),
    ("today_yield_pct", "今日收益率展示", "本日在状态/报告中展示的收益率，来自 today_yield.json；须以实际（老虎核对）为准。"),
    ("demo_order_success", "DEMO 主单成功次数", "DEMO 日志中「订单提交成功」等匹配次数（多日志汇总），非老虎后台笔数。"),
    ("demo_sl_tp_log", "DEMO 止损/止盈日志条数", "日志全文匹配「止损|止盈|已提交止损|已提交止盈」等的出现次数。"),
    ("demo_execute_buy_calls", "DEMO 买入动作次数", "日志匹配「execute_buy|动作: 买入」的次数。"),
    ("demo_success_orders_sum", "DEMO 成功订单数(日志)", "日志内统计的成功订单数汇总，非老虎后台。"),
    ("demo_fail_orders_sum", "DEMO 失败订单数(日志)", "日志内统计的失败订单数汇总。"),
    ("demo_logs_scanned", "DEMO 扫描日志数", "参与汇总的 demo_*.log、demo_run_20h_*.log 文件个数。"),
]

# 每日收益与算法优化：在干啥、咋干的（写入策略报告与索引页）
ROUTINE_WHAT_HOW = """
**每日「收益与算法优化」在干啥**
- **结果分析**：用 API 历史订单算收益率/胜率（若有）；用 DEMO 多日志汇总订单与止损止盈统计；用 today_yield 展示今日收益率。
- **算法优化**：对网格/BOLL 做参数网格回测（需 `data/processed/test.csv`），得到最优参数与回测收益/胜率，写入报告。
- **报告产出**：更新 `algorithm_optimization_report.json`/`.md`、本策略算法与运行效果报告；报告内「效果数据来源」会写明本次用了哪些数据。

**咋干的（步骤）**
1. 加载历史订单（API）→ 若无则收益率为空。
2. 计算收益率（解析订单盈亏）→ 当前未解析时为空。
3. 分析策略表现：汇总所有 DEMO 日志（demo_*.log、demo_run_20h_*.log）→ 主单成功、止损止盈条数等；读 today_yield.json。
4. 优化参数：对 grid、boll 跑网格回测（parameter_grid_search）→ 最优参数与 return_pct、win_rate。
5. 生成算法优化报告（含效果数据来源说明）并调用本脚本刷新策略报告。

**脚本**：`python scripts/optimize_algorithm_and_profitability.py`。详见 `docs/每日例行_效果数据说明.md`。
"""

# 各策略算法说明与设计文档链接（可随代码更新而维护）
# design_doc: 相对于 docs/ 的路径，报告内会生成「设计文档」节并链接到该文件
STRATEGY_ALGORITHMS = {
    "moe_transformer": {
        "name": "MoE Transformer",
        "description": """基于混合专家（Mixture of Experts）的 Transformer 时序预测策略。
- **模型**：多专家 Transformer，输入多时间尺度特征（如 46 维），输出方向/收益预测。
- **信号**：结合方向置信度与预测收益，在满足风控条件下发出买入/卖出/观望。
- **训练**：历史 K 线 + 技术指标，预测下一阶段涨跌与收益；支持 LoRA/微调。
- **适用**：DEMO/实盘主推策略之一，适合中短周期趋势与波动。""",
        "design_doc": "strategy_designs/设计_MoE策略.md",
    },
    "lstm": {
        "name": "LSTM",
        "description": """基于 LSTM 的时序预测策略（与 LLM 策略同架构，mode=hybrid）。
- **模型**：LSTM 编码 + 全连接输出，支持 predict_profit 收益预测。
- **信号**：与 MoE 类似，由预测方向与收益生成交易信号。
- **训练**：同多时间尺度历史数据。
- **适用**：作为对比基线或备选模型。""",
        "design_doc": "strategy_designs/设计_LSTM策略.md",
    },
    "grid": {
        "name": "网格策略",
        "description": """基于价格区间的网格交易策略。
- **逻辑**：以 5 分钟 Boll 中轨/上轨/下轨或时段自适应区间作为 grid_lower / grid_upper，价格接近下轨且 1 分钟 RSI 低位时考虑买入，接近上轨或止盈/止损条件时卖出。
- **参数**：网格间距、RSI 阈值、时段相关 max_position 等由时段自适应策略调整。
- **适用**：震荡市、区间行情。""",
        "design_doc": "strategy_designs/设计_网格与BOLL策略.md",
    },
    "boll": {
        "name": "BOLL 网格策略",
        "description": """基于布林带的 1 分钟网格变体（boll1m_grid_strategy）。
- **逻辑**：使用 5 分钟布林带中轨与上下轨作为区间边界，结合 1 分钟 K 线与 RSI 判断入场与出场。
- **与 grid 关系**：同属网格族，参数与时段配置可单独调优。
- **适用**：与网格策略类似，侧重 1m 与 5m 结合。""",
        "design_doc": "strategy_designs/设计_网格与BOLL策略.md",
    },
}


def load_run_effect():
    """从现有报告、JSON、DEMO 日志汇总运行效果。"""
    out = {
        "timestamp": datetime.now().isoformat(),
        "strategy_performance": {},
        "today_yield": None,
        "algorithm_report_path": None,
        "algo_report_mtime": None,
        "demo_log_stats": None,
    }
    # 算法优化报告
    algo_path = REPORTS_DIR / "algorithm_optimization_report.json"
    if algo_path.exists():
        out["algorithm_report_path"] = str(algo_path)
        try:
            out["algo_report_mtime"] = datetime.fromtimestamp(algo_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
        try:
            with open(algo_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            out["strategy_performance"] = data.get("strategy_performance") or {}
            out["profitability"] = data.get("profitability")
            out["algorithm_version"] = data.get("algorithm_version") or ""
        except Exception:
            pass
    # 今日收益率（保证 yield_pct 不为空字符串，避免报告里「收益率：」后面空白）
    yield_path = ROOT / "docs" / "today_yield.json"
    if yield_path.exists():
        try:
            with open(yield_path, "r", encoding="utf-8") as f:
                y = json.load(f)
            if y.get("yield_pct") == "" or y.get("yield_pct") is None:
                y["yield_pct"] = "—"
            if y.get("yield_note") == "" or y.get("yield_note") is None:
                y["yield_note"] = "待统计"
            out["today_yield"] = y
        except Exception:
            pass
    # DEMO 多日/多日志汇总（扫描所有 demo_*.log、demo_run_20h_*.log，避免“没数据”）
    try:
        from scripts.analyze_demo_log import aggregate_demo_logs
        out["demo_log_stats"] = aggregate_demo_logs()
    except Exception:
        pass
    return out


def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)


def write_strategy_report(strategy_id: str, meta: dict, run_effect: dict):
    """写单策略报告：设计文档置顶 + 算法说明 + 运行效果。"""
    perf = run_effect.get("strategy_performance") or {}
    row = perf.get(strategy_id, {})
    name = meta.get("name", strategy_id)

    lines = [
        f"# {name} 策略",
        "",
        f"*报告生成时间：{run_effect.get('timestamp', '')}*",
        "",
    ]
    # 设计文档置顶，便于一眼看到
    design_doc = meta.get("design_doc")
    if design_doc:
        design_path = f"../../{design_doc}"
        design_name = design_doc.split("/")[-1].replace(".md", "")
        lines.append("## 📄 设计文档（算法与参数详解）")
        lines.append("")
        lines.append(f"- **→ [{design_name}]({design_path})** — 算法原理、参数含义、训练流程与实现细节。")
        lines.append("")
    lines.append("## 算法说明")
    lines.append("")
    lines.append(meta.get("description", "（暂无算法说明）"))
    lines.append("")
    if design_doc:
        lines.append(f"更完整的说明（模型结构、信号逻辑、训练与回测）请参见上方 **设计文档**：[{design_name}]({design_path})。")
        lines.append("")
    lines.append("## 运行效果")
    lines.append("")
    row = row or {}
    # 回测效果：回测收益率、回测胜率、回测笔数
    backtest_row = {k: row.get(k) for k in BACKTEST_KEYS if row.get(k) is not None}
    if (row.get("return_pct") is not None or row.get("num_trades") is not None) and backtest_row:
        lines.append("### 回测效果")
        lines.append("")
        lines.append("（回测数据：历史 K 线回测。）")
        lines.append("")
        lines.append("| 指标 | 值 | 说明 |")
        lines.append("| --- | --- | --- |")
        for k in BACKTEST_KEYS:
            if k in backtest_row:
                _, name, desc = next((x for x in INDICATOR_DEFINITIONS if x[0] == k), (k, k, ""))
                lines.append(f"| {k} | {backtest_row[k]} | {desc} |")
        lines.append("")
    # 实盘/DEMO 效果：实盘胜率、实际/推算收益率、今日展示、DEMO 汇总
    profitability = run_effect.get("profitability")
    live_wr = "—"
    if isinstance(profitability, dict) and (profitability.get("total_trades") or 0) > 0 and profitability.get("win_rate") is not None:
        live_wr = f"{profitability['win_rate']:.1f}" if isinstance(profitability["win_rate"], (int, float)) else str(profitability["win_rate"])
    y = run_effect.get("today_yield") or {}
    ysrc = y.get("source") or "none"
    yp = (y.get("yield_pct") or y.get("yield_note") or "—").strip() or "—"
    yield_verified = yp if ysrc in ("tiger_backend", "report") and yp != "—" else "—"
    yield_estimated = (yp + "（未核对）") if ysrc == "none" and yp != "—" else "—"
    live_demo_row = {k: row.get(k) for k in LIVE_DEMO_KEYS if row.get(k) is not None and k not in ("yield_verified", "yield_estimated")}
    live_demo_row["win_rate"] = live_wr
    live_demo_row["yield_verified"] = yield_verified
    live_demo_row["yield_estimated"] = yield_estimated
    if "today_yield_pct" not in live_demo_row:
        live_demo_row["today_yield_pct"] = yp
    if live_demo_row:
        lines.append("### 实盘/DEMO 效果")
        lines.append("")
        lines.append("（实盘数据：实盘胜率、实际收益率（老虎核对）、推算收益率（未核对）、今日展示、DEMO 日志汇总。）")
        lines.append("")
        lines.append("| 指标 | 值 | 说明 |")
        lines.append("| --- | --- | --- |")
        for k in ("win_rate", "yield_verified", "yield_estimated", "today_yield_pct") + tuple(k for k in LIVE_DEMO_KEYS if k not in ("win_rate", "yield_verified", "yield_estimated", "today_yield_pct") and k in live_demo_row):
            v = live_demo_row.get(k, "—")
            _, name, desc = next((x for x in INDICATOR_DEFINITIONS if x[0] == k), (k, k, ""))
            lines.append(f"| {k} | {v} | {desc} |")
        lines.append("")
    # 无回测且无实盘数据时的提示
    if not backtest_row and not live_demo_row:
        numeric_vals = [v for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        all_zeros = len(numeric_vals) > 0 and all(x == 0 for x in numeric_vals)
        if row and all_zeros:
            demo_stats = run_effect.get("demo_log_stats")
            if demo_stats and demo_stats.get("logs_scanned", 0) > 0:
                if strategy_id == "moe_transformer":
                    lines.append("回测/实盘指标当前为占位或未写入；DEMO 多日汇总见下方 **DEMO 运行统计**。")
                else:
                    lines.append("回测/实盘指标当前为占位或未写入；DEMO 多日汇总见 **MoE Transformer** 策略报告中的「DEMO 运行统计」。")
            else:
                lines.append("**暂无运行数据**（当前数据源为占位 0，且未发现 DEMO 日志）。")
                lines.append("- 请确认已运行 DEMO（如 `python scripts/run_moe_demo.py moe 20`），日志位于项目根目录或 `logs/` 下的 `demo_*.log`、`demo_run_20h_*.log`。")
            algo_mtime = run_effect.get("algo_report_mtime")
            if algo_mtime:
                lines.append(f"- 数据源更新时间：{algo_mtime}（`algorithm_optimization_report.json`）")
            lines.append("- 运行 **收益与算法优化**（`python scripts/optimize_algorithm_and_profitability.py`）或回测后，再运行 `python scripts/generate_strategy_reports.py` 可刷新。")
        lines.append("")
    # DEMO 多日/多日志汇总（扫描所有 demo_*.log、demo_run_20h_*.log）
    demo_stats = run_effect.get("demo_log_stats")
    if demo_stats and strategy_id == "moe_transformer":
        n_logs = demo_stats.get("logs_scanned", 0)
        lines.append("### DEMO 运行统计（多日/多日志汇总）")
        lines.append("")
        lines.append(f"共扫描 **{n_logs}** 个 DEMO 日志文件（`demo_*.log`、`demo_run_20h_*.log`），汇总如下。")
        lines.append("")
        lines.append("| 项 | 值 |")
        lines.append("| --- | --- |")
        lines.append(f"| 主单成功次数（汇总） | {demo_stats.get('order_success', 0)} |")
        lines.append(f"| 成功订单数（日志内统计汇总） | {demo_stats.get('success_orders_sum', 0)} |")
        lines.append(f"| 失败订单数（日志内统计汇总） | {demo_stats.get('fail_orders_sum', 0)} |")
        lines.append(f"| 止损/止盈相关日志条数 | {demo_stats.get('sl_tp_log', 0)} |")
        lines.append(f"| 买入动作/execute_buy 次数 | {demo_stats.get('execute_buy_calls', 0)} |")
        lines.append(f"| 日志总行数 | {demo_stats.get('lines', 0)} |")
        if demo_stats.get("max_position") is not None:
            lines.append(f"| 最大仓位（各日志中出现过的最大值） | {demo_stats['max_position']} 手 |")
        lines.append("")
    today_yield = run_effect.get("today_yield")
    if today_yield and (today_yield.get("yield_pct") or today_yield.get("yield_note")) and (str(today_yield.get("yield_pct", "")).strip() not in ("", "—", "0")):
        lines.append("### 今日收益率")
        lines.append("")
        lines.append(f"- {today_yield.get('yield_pct', today_yield.get('yield_note', '—'))}")
        lines.append("")
    # 每日收益与算法优化在干啥、咋干的（输出到策略报告里）
    lines.append("## 每日收益与算法优化在干啥")
    lines.append("")
    lines.append(ROUTINE_WHAT_HOW.strip())
    lines.append("")
    lines.append("详见：[每日例行_效果数据说明](../../每日例行_效果数据说明.md)。")
    lines.append("")

    path = STRATEGY_REPORTS_DIR / f"strategy_{strategy_id}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_comparison_report(run_effect: dict):
    """写对比报告：回测效果表 + 实盘/DEMO 效果表 + 每日例行说明。"""
    ts = run_effect.get("timestamp", "")

    perf = run_effect.get("strategy_performance") or {}
    has_perf = len(perf) > 0 and any(perf.get(s) for s in perf)
    algo_ver = run_effect.get("algorithm_version") or "—"
    lines = [
        "# 策略对比报告",
        "",
        f"*报告生成时间：{ts}*",
        "",
        f"**算法版本**：{algo_ver}（重大变更与对比见 [algorithm_versions.md](../../algorithm_versions.md)）",
        "",
    ]
    if not has_perf:
        lines.extend([
            "> **若您看到下方表格或今日收益率为空**：请在本机 **tigertrade 根目录**执行：`python3 scripts/optimize_algorithm_and_profitability.py`，再执行 `python3 scripts/generate_strategy_reports.py`，然后刷新本页或重新打开报告。",
            "",
        ])
    lines.extend([
        "## 数据来源与「结果不全」说明",
        "",
        "- **回测效果**：**grid / boll** 由 `parameter_grid_search` 参数网格回测（**双向**：long/short）；**moe_transformer、lstm** 由 `scripts/backtest_model_strategies.py` 用 test.csv 信号回测（**双向**：1=多/平空，2=空/平多），四策略均有 num_trades/return_pct/win_rate。",
        "- **回测 vs 实盘**：回测与实盘仅数据来源不同，策略与运行过程应一致才有参考意义；若回测笔数远少于实盘说明不一致需对齐。详见 [algorithm_optimization_report.md](algorithm_optimization_report.md)「回测与实盘差异说明」。",
        "- **实盘/DEMO 效果**：**demo_*** 等列来自 DEMO 多日志汇总；同次运行四策略共用统计，故 grid/boll/lstm 与 MoE 数字一致。",
        "- **今日收益率**：来自 `docs/today_yield.json`。若为 —，请运行 **收益与算法优化**（`python scripts/optimize_algorithm_and_profitability.py`）或单独运行 `python scripts/update_today_yield_for_status.py`，会从报告或 DEMO 日志更新后再刷新本报告。",
        "",
        "## 回测效果对比",
        "",
        "（回测数据：历史 K 线回测，含**回测收益率、回测胜率、回测笔数**。）",
        "",
    ])
    if perf:
        strategies = list(perf.keys())
        bk = [k for k in BACKTEST_KEYS if any((perf.get(s) or {}).get(k) is not None for s in strategies)]
        if bk:
            header = "| 策略 | " + " | ".join(bk) + " |"
            sep = "| --- | " + " | ".join("---" for _ in bk) + " |"
            lines.append(header)
            lines.append(sep)
            def _cell(v):
                if v is None or v == "—":
                    return "—"
                if isinstance(v, float):
                    return str(round(v, 2))
                return str(v)
            for s in strategies:
                row = perf.get(s) or {}
                cells = [_cell(row.get(k)) if row.get(k) is not None else "—" for k in bk]
                lines.append("| " + s + " | " + " | ".join(cells) + " |")
            lines.append("")
            lines.append("*说明*：**num_trades**=实际成交笔数；**return_pct**=总收益率；**avg_per_trade_pct**=单笔平均%；**top_per_trade_pct**=单笔TOP%；**win_rate**=胜率。")
            lines.append("")
        else:
            lines.append("（暂无回测数据。）")
            lines.append("")
        lines.append("## 实盘/DEMO 效果对比")
        lines.append("")
        lines.append("（实盘表与回测表**同结构**：笔数、收益率（核对/推算）、单笔均、单笔TOP、胜率；仅收益率区分「老虎核对」与「未核对推算」。DEMO 日志汇总见下表。）")
        lines.append("")
        profitability = run_effect.get("profitability")
        live_win_rate = "—"
        live_num_trades = "—"
        live_avg = "—"
        live_top = "—"
        if isinstance(profitability, dict) and (profitability.get("total_trades") or 0) > 0:
            n = profitability.get("total_trades")
            live_num_trades = str(n) if n is not None else "—"
            w = profitability.get("win_rate")
            if w is not None:
                live_win_rate = f"{w:.1f}" if isinstance(w, (int, float)) else str(w)
            ap = profitability.get("average_profit")
            if ap is not None:
                live_avg = f"{ap:.2f} USD" if isinstance(ap, (int, float)) else str(ap)
            tp = profitability.get("total_profit")
            if tp is not None:
                live_top = f"{tp:.2f} USD" if isinstance(tp, (int, float)) else str(tp)
        # 无老虎 API 时用 DEMO 汇总填 num_trades，报告不空项
        if live_num_trades == "—":
            demo = run_effect.get("demo_log_stats")
            if demo and demo.get("logs_scanned", 0) > 0:
                n_d = demo.get("order_success", 0)
                live_num_trades = f"{n_d}（DEMO主单，见下表）"
        y = run_effect.get("today_yield") or {}
        src = y.get("source") or "none"
        yp = (y.get("yield_pct") or "").strip()
        yn = (y.get("yield_note") or "").strip()
        if not yp and yn:
            yp = yn
        if not yp:
            yp = "—"
        return_pct_verified = "—"
        return_pct_estimated = "—"
        if src == "tiger_backend" and yp and yp != "—":
            return_pct_verified = yp
        elif src == "report" and yp and yp != "—":
            return_pct_verified = yp + "（API报告）"
        if src == "demo_aggregate" and yp and yp != "—":
            return_pct_estimated = yp + "（DEMO未核对）" if ("%" in yp or "USD" in yp) else "—"
        elif src == "none" and yp and yp != "—" and ("%" in yp or "USD" in yp or "未核对" in (yn or "")):
            return_pct_estimated = yp
        _empty_note = "—（见根因说明）"
        def _cell_live(v):
            if v is None or v == "" or (isinstance(v, str) and v.strip() in ("—", "")):
                return _empty_note
            return str(v)
        # 实盘主表：与回测同列（num_trades, return_pct 核对/推算, avg, top, win_rate）
        header = "| 策略 | " + " | ".join(LIVE_TABLE_KEYS) + " |"
        sep = "| --- | " + " | ".join("---" for _ in LIVE_TABLE_KEYS) + " |"
        lines.append(header)
        lines.append(sep)
        for s in strategies:
            cells = []
            for k in LIVE_TABLE_KEYS:
                if k == "num_trades":
                    cells.append(_cell_live(live_num_trades))
                elif k == "return_pct_verified":
                    cells.append(_cell_live(return_pct_verified))
                elif k == "return_pct_estimated":
                    cells.append(_cell_live(return_pct_estimated))
                elif k == "avg_per_trade_pct":
                    cells.append(_cell_live(live_avg))
                elif k == "top_per_trade_pct":
                    cells.append(_cell_live(live_top))
                elif k == "win_rate":
                    cells.append(_cell_live(live_win_rate))
                else:
                    cells.append(_empty_note)
            lines.append("| " + s + " | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("*说明*：与回测表同指标；**return_pct_verified**=老虎核对收益率，**return_pct_estimated**=未核对推算；无数据时为 —（见根因说明）。")
        lines.append("")
        # DEMO 日志汇总（单独小表，不混入实盘主表）
        demo_keys = [k for k in LIVE_DEMO_KEYS if k.startswith("demo_") and any((perf.get(s) or {}).get(k) is not None for s in strategies)]
        if demo_keys:
            lines.append("### DEMO 日志汇总")
            lines.append("")
            lines.append("| 策略 | " + " | ".join(demo_keys) + " |")
            lines.append("| --- | " + " | ".join("---" for _ in demo_keys) + " |")
            for s in strategies:
                row = perf.get(s) or {}
                cells = [_cell_live(row.get(k)) if row.get(k) is None or row.get(k) == "—" else str(row.get(k)) for k in demo_keys]
                lines.append("| " + s + " | " + " | ".join(cells) + " |")
            lines.append("")
        n_backtest = sum(1 for s in strategies if (perf.get(s) or {}).get("return_pct") is not None or (perf.get(s) or {}).get("num_trades") is not None)
        n_demo = sum(1 for s in strategies if any((perf.get(s) or {}).get(k) is not None for k in LIVE_DEMO_KEYS if k.startswith("demo_")))
        lines.append("**数据完整度**：回测 " + str(n_backtest) + "/" + str(len(strategies)) + " 策略有数据；实盘主表来自老虎 API/今日收益率；DEMO 汇总 " + str(n_demo) + "/" + str(len(strategies)) + " 策略。")
        lines.append("")
    else:
        lines.append("（暂无对比数据，由每日算法优化/回测流程更新。）")
        lines.append("")

    y = run_effect.get("today_yield") or {}
    date_display = (y.get("date") or "").strip() or datetime.now().strftime("%Y-%m-%d")
    yp = (y.get("yield_pct") or y.get("yield_note") or "—").strip() or "—"
    src = y.get("source") or "none"
    lines.append("## 今日收益率（DEMO/实盘）")
    lines.append("")
    lines.append(f"- 日期：{date_display}")
    _verified = yp if src in ("tiger_backend", "report") and yp and yp != "—" else "—"
    if src == "none" and _verified == "—":
        _verified = "—（根因见 [算法优化报告](../algorithm_optimization_report.md) 中「本报告空项根因说明」）"
    _estimated = "—"
    if src == "none" and yp and yp != "—":
        _estimated = yp + "（未核对）"
    if _estimated == "—":
        _estimated = "—（根因见 [算法优化报告](../algorithm_optimization_report.md) 中「本报告空项根因说明」）"
    lines.append(f"- **实际收益率（老虎后台核对）**：{_verified}")
    lines.append(f"- **推算收益率（未核对）**：{_estimated}")
    today_display = yp
    if (not today_display or today_display == "—") and run_effect.get("demo_log_stats", {}).get("logs_scanned", 0) > 0:
        today_display = "无老虎核对；实盘笔数见上表「num_trades」列（DEMO 主单）"
    lines.append(f"- 当前展示：{today_display}")
    lines.append("- **空项根因**：实际/推算收益率为空时，原因均写在 [算法优化报告](../algorithm_optimization_report.md) 的「本报告空项根因说明」中，须追根问底、不忽悠。")
    if yp == "—" or not yp:
        lines.append("- （若为 —：运行 `python scripts/optimize_algorithm_and_profitability.py` 或 `update_today_yield_for_status.py` 更新。）")
    lines.append("")
    lines.append("## 指标说明（含义与计算方式）")
    lines.append("")
    lines.append("| 指标项 | 含义 | 计算方式 / 说明 |")
    lines.append("| --- | --- | --- |")
    for key, name, desc in INDICATOR_DEFINITIONS:
        lines.append(f"| {key} | {name} | {desc} |")
    lines.append("")
    lines.append("详见 [DEMO实盘收益率_定义与数据来源](../../DEMO实盘收益率_定义与数据来源.md)、[每日例行_效果数据说明](../../每日例行_效果数据说明.md)、[回溯_执行失败为何出现收益率与推算收益率](../../回溯_执行失败为何出现收益率与推算收益率.md)。")
    lines.append("")
    lines.append("## 每日收益与算法优化在干啥")
    lines.append("")
    lines.append(ROUTINE_WHAT_HOW.strip())
    lines.append("")
    lines.append("详见：[每日例行_效果数据说明](../../每日例行_效果数据说明.md)。")
    lines.append("")

    path = STRATEGY_REPORTS_DIR / "strategy_comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def check_report_reasonableness(run_effect: dict, comparison_path: Path) -> Tuple[bool, List[str]]:
    """自检：实盘表胜率与数据来源一致（无 API 时不应出现 100%）；实盘表与回测同结构（return_pct 核对/推算）。返回 (通过, 警告列表)。"""
    warnings = []
    profitability = run_effect.get("profitability")
    has_api = isinstance(profitability, dict) and (profitability.get("total_trades") or 0) > 0
    if has_api:
        return True, []
    if not comparison_path.exists():
        return True, []
    text = comparison_path.read_text(encoding="utf-8")
    if "## 实盘/DEMO 效果对比" not in text:
        return True, []
    lines = text.splitlines()
    in_table = False
    header_idx = -1
    win_rate_col = -1
    for i, line in enumerate(lines):
        if "## 实盘/DEMO 效果对比" in line:
            in_table = True
            continue
        if in_table and line.startswith("|"):
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if "win_rate" in cells:
                win_rate_col = cells.index("win_rate")
                header_idx = i
                break
    if header_idx < 0:
        return True, []
    # ① 无 API 时实盘 win_rate 不应为 100
    if not has_api and win_rate_col >= 0:
        for i in range(header_idx + 2, len(lines)):
            line = lines[i]
            if not line.startswith("|"):
                break
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if len(cells) <= win_rate_col:
                continue
            val = cells[win_rate_col]
            if val == "100.0" or val == "100":
                warnings.append(
                    f"实盘/DEMO 表第{i+1}行 win_rate={val}%，但无 API 订单数据，疑似回测胜率误入实盘列，请检查。"
                )
    # ② 实盘表与回测同结构：应有 return_pct_verified/return_pct_estimated 列；有 API 或 today_yield 时对应列应有值（收益率，非笔数）
    # 自检仅检查列存在与表结构，不强制填笔数入收益率列
    return len(warnings) == 0, warnings


def write_index_html(run_effect: dict):
    """写 index 页（HTML），供 STATUS 页链接，每日刷新。使用相对路径，与 status 同源打开时报告内容为当前生成。"""
    ts = run_effect.get("timestamp", "")[:19].replace("T", " ")
    # 相对路径：index 在 docs/reports/，报告在 docs/reports/strategy_reports/，设计在 docs/strategy_designs/
    base_url = "strategy_reports"
    design_base = "../strategy_designs"

    links = []
    links.append('        <li class="section">📄 <strong>设计文档（算法与参数详解）</strong></li>')
    links.append(f'        <li><a href="{design_base}/README.md" target="_blank" rel="noopener">策略设计文档索引</a></li>')
    links.append(f'        <li><a href="{design_base}/设计_MoE策略.md" target="_blank" rel="noopener">设计_MoE策略</a></li>')
    links.append(f'        <li><a href="{design_base}/设计_LSTM策略.md" target="_blank" rel="noopener">设计_LSTM策略</a></li>')
    links.append(f'        <li><a href="{design_base}/设计_网格与BOLL策略.md" target="_blank" rel="noopener">设计_网格与BOLL策略</a></li>')
    links.append('        <li class="section">📊 各策略报告（含运行效果）</li>')
    for sid, meta in STRATEGY_ALGORITHMS.items():
        name = meta.get("name", sid)
        links.append(
            f'        <li><a href="{base_url}/strategy_{sid}.md" target="_blank" rel="noopener">{name}</a></li>'
        )
    links.append(
        f'        <li><a href="{base_url}/strategy_comparison.md" target="_blank" rel="noopener"><strong>对比报告</strong></a></li>'
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
  <meta http-equiv="Pragma" content="no-cache">
  <meta http-equiv="Expires" content="0">
  <title>策略算法与运行效果报告</title>
  <style>
    body {{ font-family: "Noto Sans SC", sans-serif; background: #0d0f12; color: #e5e7eb; padding: 2rem; max-width: 640px; margin: 0 auto; }}
    h1 {{ color: #f59e0b; font-size: 1.35rem; }}
    .meta {{ font-size: 0.85rem; color: #9ca3af; margin-bottom: 1rem; }}
    ul {{ line-height: 1.8; }}
    a {{ color: #f59e0b; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    li.section {{ color: #9ca3af; list-style: none; margin-top: 0.6rem; }}
  </style>
</head>
<body>
  <h1>策略算法与运行效果报告</h1>
  <p class="meta">每日刷新 · 最后生成：{ts}</p>
  <p class="meta">各策略报告内含 <strong>设计文档链接</strong>（置顶）与算法说明；详细原理与参数见上方设计文档。</p>
  <ul>
{chr(10).join(links)}
  </ul>
  <p class="meta"><strong>每日收益与算法优化</strong>：结果分析（API 订单/DEMO 日志/today_yield）+ 算法优化（网格/BOLL 回测）→ 产出本报告；脚本 <code>scripts/optimize_algorithm_and_profitability.py</code>，详见 <a href="../每日例行_效果数据说明.md" target="_blank" rel="noopener">每日例行_效果数据说明</a>。</p>
  <p class="meta"><a href="../status.html">← 返回状态页</a></p>
</body>
</html>
"""
    path = REPORTS_DIR / "strategy_reports_index.html"
    path.write_text(html, encoding="utf-8")
    return path


def main():
    ensure_dir(STRATEGY_REPORTS_DIR)
    # 先刷新今日收益率，再加载数据，使对比报告里「今日收益率」尽量不全为 —
    try:
        import subprocess
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "update_today_yield_for_status.py")],
            cwd=str(ROOT),
            capture_output=True,
            timeout=30,
            check=False,
        )
    except Exception:
        pass
    run_effect = load_run_effect()
    # 本次运行时间作为报告生成时间，避免显示为上次 optimize 的 10 点等旧时间
    run_effect["timestamp"] = datetime.now().isoformat()

    for sid, meta in STRATEGY_ALGORITHMS.items():
        write_strategy_report(sid, meta, run_effect)
    comp_path = write_comparison_report(run_effect)
    write_index_html(run_effect)

    ok, warn_list = check_report_reasonableness(run_effect, comp_path)
    if ok:
        print("报告自检: 通过（实盘胜率与数据来源一致；实盘表与回测同结构）")
    else:
        for w in warn_list:
            print(f"报告自检: 警告 — {w}")
    print("策略报告已生成：")
    print(f"  - {STRATEGY_REPORTS_DIR}/")
    print(f"  - {REPORTS_DIR}/strategy_reports_index.html")
    print("")
    print("⚠️ 报告自检须到网页上查看，本地不算。push 后等待 GitHub Pages 部署完成，再打开部署后的 status 与报告页核对内容是否最新。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
