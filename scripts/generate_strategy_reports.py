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
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "docs" / "reports"
STRATEGY_REPORTS_DIR = REPORTS_DIR / "strategy_reports"

# 各策略算法说明（可随代码更新而维护）
STRATEGY_ALGORITHMS = {
    "moe_transformer": {
        "name": "MoE Transformer",
        "description": """基于混合专家（Mixture of Experts）的 Transformer 时序预测策略。
- **模型**：多专家 Transformer，输入多时间尺度特征（如 46 维），输出方向/收益预测。
- **信号**：结合方向置信度与预测收益，在满足风控条件下发出买入/卖出/观望。
- **训练**：历史 K 线 + 技术指标，预测下一阶段涨跌与收益；支持 LoRA/微调。
- **适用**：DEMO/实盘主推策略之一，适合中短周期趋势与波动。""",
    },
    "lstm": {
        "name": "LSTM",
        "description": """基于 LSTM 的时序预测策略（与 LLM 策略同架构，mode=hybrid）。
- **模型**：LSTM 编码 + 全连接输出，支持 predict_profit 收益预测。
- **信号**：与 MoE 类似，由预测方向与收益生成交易信号。
- **训练**：同多时间尺度历史数据。
- **适用**：作为对比基线或备选模型。""",
    },
    "grid": {
        "name": "网格策略",
        "description": """基于价格区间的网格交易策略。
- **逻辑**：以 5 分钟 Boll 中轨/上轨/下轨或时段自适应区间作为 grid_lower / grid_upper，价格接近下轨且 1 分钟 RSI 低位时考虑买入，接近上轨或止盈/止损条件时卖出。
- **参数**：网格间距、RSI 阈值、时段相关 max_position 等由时段自适应策略调整。
- **适用**：震荡市、区间行情。""",
    },
    "boll": {
        "name": "BOLL 网格策略",
        "description": """基于布林带的 1 分钟网格变体（boll1m_grid_strategy）。
- **逻辑**：使用 5 分钟布林带中轨与上下轨作为区间边界，结合 1 分钟 K 线与 RSI 判断入场与出场。
- **与 grid 关系**：同属网格族，参数与时段配置可单独调优。
- **适用**：与网格策略类似，侧重 1m 与 5m 结合。""",
    },
}


def load_run_effect():
    """从现有报告与 JSON 汇总运行效果。"""
    out = {
        "timestamp": datetime.now().isoformat(),
        "strategy_performance": {},
        "today_yield": None,
        "algorithm_report_path": None,
    }
    # 算法优化报告
    algo_path = REPORTS_DIR / "algorithm_optimization_report.json"
    if algo_path.exists():
        out["algorithm_report_path"] = str(algo_path)
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
    return out


def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)


def write_strategy_report(strategy_id: str, meta: dict, run_effect: dict):
    """写单策略报告：算法说明 + 运行效果。"""
    perf = run_effect.get("strategy_performance") or {}
    row = perf.get(strategy_id, {})
    name = meta.get("name", strategy_id)

    lines = [
        f"# {name} 策略",
        "",
        f"*报告生成时间：{run_effect.get('timestamp', '')}*",
        "",
        "## 算法说明",
        "",
        meta.get("description", "（暂无算法说明）"),
        "",
        "## 运行效果",
        "",
    ]
    if row:
        lines.append("| 指标 | 值 |")
        lines.append("| --- | --- |")
        for k, v in row.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                lines.append(f"| {k} | {v} |")
            else:
                lines.append(f"| {k} | {v} |")
        lines.append("")
    else:
        lines.append("（暂无运行数据，由每日算法优化/回测流程更新。）")
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

    links = []
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
  </style>
</head>
<body>
  <h1>策略算法与运行效果报告</h1>
  <p class="meta">每日刷新 · 最后生成：{ts}</p>
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
