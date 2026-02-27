# QA 定时分析监控说明

**目的**：定时分析项目进度与质量，对照项目目标持续发现问题，输出可跟进的监控报告。

**责任人**：陈正孤（QA）设计；项目 leader（陈正霞）可安排 cron 或 CI 执行。

---

## 1. 做什么

- **输入**：`docs/reports/algorithm_optimization_report.json`、`stability_stats.json`（若有）、`src/` 下源码、`docs/STATUS.md` 等。
- **逻辑**：
  - 将报告中的**实盘胜率、回测收益率**与项目目标（胜率>60%、第1周回测>15%）对比，未达标则记为问题。
  - 检查报告是否产出**夏普、回撤、盈亏比**，未产出或全为 0 则记为问题。
  - 若存在 **20h 稳定性** 结果，检查错误率是否 < 1%，超标则记为问题。
  - 扫描 **src/** 下 **TODO/FIXME/XXX**，数量过多时提示登记到计划或 Issue。
  - 检查 **STATUS.md** 是否超过 14 天未更新，是则提示更新进度。
- **输出**：`docs/reports/qa_monitor_latest.md`（每次覆盖）；可选 `--append-log` 时追加一行到 `logs/qa_monitor.log`。

---

## 2. 如何运行

### 手动

```bash
cd /home/cx/tigertrade
python3 scripts/qa_progress_quality_monitor.py
# 带追加日志
python3 scripts/qa_progress_quality_monitor.py --append-log
```

### 定时（crontab）

建议**每日**运行一次（与每日例行错开，如例行 8:00 北京则 QA 监控 9:00 北京）：

```bash
# 每日 9:00 北京时间（1:00 UTC）
0 1 * * * cd /home/cx/tigertrade && bash scripts/cron_qa_monitor.sh >> logs/qa_monitor.log 2>&1
```

或使用项目 cron 同一用户下的条目，避免重复：

```bash
0 1 * * * cd /home/cx/tigertrade && python3 scripts/qa_progress_quality_monitor.py --append-log >> logs/qa_monitor.log 2>&1
```

### 与 CI 结合（可选）

若希望 CI 每日生成监控报告并上传 artifact，可在 `.github/workflows/daily_ci_cd.yml` 的 `optimize_algorithm` job 之后增加一步：

```yaml
- name: QA 进度与质量监控
  run: |
    python scripts/qa_progress_quality_monitor.py
- name: 上传 QA 监控报告
  uses: actions/upload-artifact@v4
  with:
    name: qa-monitor
    path: docs/reports/qa_monitor_latest.md
```

---

## 3. 输出说明

- **qa_monitor_latest.md**：含「数据来源」「项目目标」「发现的问题」三部分；问题按高/中/低分级，并给建议。
- **退出码**：存在**高**优先级问题时返回 1，否则返回 0；便于 cron 或 CI 在严重问题时告警（如 `|| echo "QA 监控发现高优问题"`）。

---

## 4. 相关文档

- 质量门禁与「有输出≠OK」：[QA_从项目目标把控质量_测试报告与运行结果](reports/QA_从项目目标把控质量_测试报告与运行结果.md)
- 项目目标与自测：[项目质量与自测目标](项目质量与自测目标.md)
- 例行与报告： [每日例行_效果数据说明](每日例行_效果数据说明.md)
