# 例行工作清单（Agent 必读）

**作者**: system  
**时间**: 2026-02-01  
**类别**: best_practices  
**标签**: routine, checklist, ci, coverage, stability, data, model, profit

---

## 摘要

执行「例行工作」时，必须覆盖以下全部项。Agent 不得遗漏或只做其中一部分；新接手任务时请先查本 RAG 再开工。

---

## 例行工作包括（完整列表）

1. **CI/CD 测试**
   - 运行项目 CI/CD 流水线或等价测试（如 `pytest tests/ -m "not real_api"`）。
   - 确保测试通过，无新增失败。

2. **覆盖率统计**
   - 执行带覆盖率收集的测试（如 `pytest --cov=src`）。
   - 查看/输出覆盖率报告，关注关键模块覆盖率。

3. **问题解决**
   - 对本次/近期暴露的 Bug、失败用例或告警进行修复与验证。
   - 记录根因与解法（可写入 RAG 或 docs）。

4. **20 小时稳定性测试（含 20 小时 DEMO 运行）**
   - 在适用环境下运行长时间稳定性测试，**含 20 小时 DEMO 运行**（目标 20 小时或约定时长）。
   - 记录运行状态、日志摘要与异常。
   - 启动命令：`bash scripts/run_20h_demo.sh` 或 `python scripts/stability_test_20h.py`。

5. **数据刷新和模型训练**
   - 数据刷新：按需更新/拉取最新数据，运行 `scripts/merge_recent_data_and_train.py` 或 `scripts/data_preprocessing.py`，确保 train/val/test 等产出可用。
   - 模型训练：执行 **多模型对比训练**（LSTM、Transformer、Enhanced Transformer、MoE 等），使用 `scripts/train_multiple_models_comparison.py`，记录各模型训练进度与对比结果。
   - **注意**：当前模型（如 Transformer）不能假定为最优；在优化过程中可能发现更好的模型结构，需持续探索与对比。

6. **收益率分析和算法优化**（两项都做，不得只做其一）
   - **收益率分析**：基于回测或实盘结果做收益、回撤、胜率等分析。
   - **算法优化**：根据分析结果做策略/参数/算法层面的优化，并做对比验证；可运行 `scripts/optimize_algorithm_and_profitability.py` 等产出算法优化与收益率报告。

---

## 使用方式

- **开工前**：查本 RAG，确认本次「例行工作」要覆盖的上述 **6 类**是否都已安排；生成待办或执行计划时须按本清单 **6 项**逐条核对，不得自拟简写版或合并为少于 6 项。
- **执行中**：按项执行并做简要记录（通过/失败/跳过原因）。
- **收尾**：未完成项需注明原因与下次计划。

**防漏（禁止简写）**：本文件为例行工作项的唯一正本。写日报、执行计划或其他文档引用「例行工作清单」时，必须以本文件为准逐项核对或复制；第 4 项须含「20 小时」「DEMO 运行」，第 6 项须同时含「收益率分析」与「算法优化」。漏项原因与改进见 `insights/例行工作漏项回溯与改进_20260201.md`。

---

## 相关路径与命令（参考）

| 项目       | 说明 |
|------------|------|
| 测试       | `cd /home/cx/tigertrade && pytest tests/ -m "not real_api"` |
| 覆盖率     | `pytest tests/ -m "not real_api" --cov=src --cov-report=term-missing` |
| **20h DEMO 运行** | `bash scripts/run_20h_demo.sh` 或 `python scripts/stability_test_20h.py`（目标 20 小时） |
| 数据合并+预处理 | `python scripts/merge_recent_data_and_train.py`，产出在 `data/processed/` |
| 多模型训练 | `python scripts/train_multiple_models_comparison.py`（LSTM/Transformer/Enhanced/MoE 等） |
| **算法优化** | `python scripts/optimize_algorithm_and_profitability.py` 等，产出算法优化与收益率报告 |
| 真实 API 用例 | 需网络与配置时运行 `pytest tests/`（含 `real_api` 标记） |

---

**请勿遗忘：例行工作 = CI/CD 测试 + 覆盖率统计 + 问题解决 + 20 小时稳定性测试（含 20 小时 DEMO 运行）+ 数据刷新和模型训练（含多模型对比训练）+ 收益率分析和算法优化（收益率分析 + 算法优化，两项都做）。**
