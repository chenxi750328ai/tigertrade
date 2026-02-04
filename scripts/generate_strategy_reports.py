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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REPORTS_DIR = ROOT / "docs" / "reports"
STRATEGY_REPORTS_DIR = REPORTS_DIR / "strategy_reports"

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
        except Exception:
            pass
    # 今日收益率
    yield_path = ROOT / "docs" / "today_yield.json"
    if yield_path.exists():
        try:
            with open(yield_path, "r", encoding="utf-8") as f:
                out["today_yield"] = json.load(f)
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
    # 判断是否为“全占位 0”（未写入真实回测/DEMO 数据）
    numeric_vals = [v for k, v in (row or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    all_zeros = len(numeric_vals) > 0 and all(x == 0 for x in numeric_vals)
    if row and not all_zeros:
        lines.append("| 指标 | 值 |")
        lines.append("| --- | --- |")
        for k, v in row.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                lines.append(f"| {k} | {v} |")
            else:
                lines.append(f"| {k} | {v} |")
        lines.append("")
    else:
        demo_stats = run_effect.get("demo_log_stats")
        if demo_stats and demo_stats.get("logs_scanned", 0) > 0:
            if strategy_id == "moe_transformer":
                lines.append("回测/优化指标（profitability、win_rate 等）当前为占位或未写入；DEMO 多日汇总见上方 **DEMO 运行统计**。")
            else:
                lines.append("回测/优化指标当前为占位或未写入；DEMO 多日汇总见 **MoE Transformer** 策略报告中的「DEMO 运行统计」。")
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

    path = STRATEGY_REPORTS_DIR / f"strategy_{strategy_id}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_comparison_report(run_effect: dict):
    """写对比报告：多策略指标对比表。"""
    perf = run_effect.get("strategy_performance") or {}
    ts = run_effect.get("timestamp", "")

    lines = [
        "# 策略对比报告",
        "",
        f"*报告生成时间：{ts}*",
        "",
        "## 各策略运行效果对比",
        "",
    ]
    if perf:
        strategies = list(perf.keys())
        if strategies:
            keys = list(perf[strategies[0]].keys()) if perf[strategies[0]] else []
            header = "| 策略 | " + " | ".join(keys) + " |"
            sep = "| --- | " + " | ".join("---" for _ in keys) + " |"
            lines.append(header)
            lines.append(sep)
            for s in strategies:
                row = perf[s] or {}
                cells = [str(row.get(k, "—")) for k in keys]
                lines.append("| " + s + " | " + " | ".join(cells) + " |")
            lines.append("")
    else:
        lines.append("（暂无对比数据，由每日算法优化/回测流程更新。）")
        lines.append("")

    if run_effect.get("today_yield"):
        y = run_effect["today_yield"]
        lines.append("## 今日收益率（DEMO/回测）")
        lines.append("")
        lines.append(f"- 日期：{y.get('date', '—')}")
        lines.append(f"- 收益率：{y.get('yield_pct', y.get('yield_note', '—'))}")
        lines.append("")

    path = STRATEGY_REPORTS_DIR / "strategy_comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_index_html(run_effect: dict):
    """写 index 页（HTML），供 STATUS 页链接，每日刷新。"""
    ts = run_effect.get("timestamp", "")[:19].replace("T", " ")
    base_url = "https://github.com/chenxi750328ai/tigertrade/blob/main/docs/reports/strategy_reports"
    design_base = "https://github.com/chenxi750328ai/tigertrade/blob/main/docs/strategy_designs"

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
  <p class="meta"><a href="../status.html">← 返回状态页</a></p>
</body>
</html>
"""
    path = REPORTS_DIR / "strategy_reports_index.html"
    path.write_text(html, encoding="utf-8")
    return path


def main():
    ensure_dir(STRATEGY_REPORTS_DIR)
    run_effect = load_run_effect()

    for sid, meta in STRATEGY_ALGORITHMS.items():
        write_strategy_report(sid, meta, run_effect)
    write_comparison_report(run_effect)
    write_index_html(run_effect)

    print("策略报告已生成：")
    print(f"  - {STRATEGY_REPORTS_DIR}/")
    print(f"  - {REPORTS_DIR}/strategy_reports_index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
