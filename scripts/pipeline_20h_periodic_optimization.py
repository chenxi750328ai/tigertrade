#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20h 与训练/算法优化并行、流水式运行：
- 与 run_20h_demo.sh / stability_test_20h 并行使用（DEMO 单独启动）。
- 本脚本按间隔定时执行「数据合并 → 训练 → 算法优化」；优化完成且回测 OK 时可记录「待重启 DEMO」或「停盘时刷新」。
- 间隔默认 4 小时，可通过环境变量 PIPELINE_OPTIMIZATION_INTERVAL_HOURS 覆盖。
- 用法：在 20h DEMO 已启动的前提下，本机执行 python scripts/pipeline_20h_periodic_optimization.py；
  或由 cron 每 4h 调用一次「数据+训练+优化」实现同等效果（无需本脚本常驻）。
"""

import os
import sys
import time
import logging
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 间隔（小时），默认 4
INTERVAL_HOURS = float(os.environ.get("PIPELINE_OPTIMIZATION_INTERVAL_HOURS", "4"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_data_and_train():
    """数据合并 + 预处理 + 训练。"""
    import subprocess
    for script in ("merge_recent_data_and_train.py", "data_preprocessing.py"):
        path = os.path.join(ROOT, "scripts", script)
        if not os.path.isfile(path):
            continue
        try:
            r = subprocess.run(
                [sys.executable, path],
                cwd=ROOT,
                timeout=3600,
                capture_output=True,
                text=True,
            )
            if r.returncode == 0:
                logger.info("数据/预处理完成: %s", script)
                break
        except subprocess.TimeoutExpired:
            logger.warning("%s 超时", script)
        except Exception as e:
            logger.debug("%s: %s", script, e)
    # 训练
    train_script = os.path.join(ROOT, "scripts", "train_multiple_models_comparison.py")
    if os.path.isfile(train_script):
        try:
            subprocess.run(
                [sys.executable, train_script],
                cwd=ROOT,
                timeout=7200,
            )
        except subprocess.TimeoutExpired:
            logger.warning("训练超时")
        except Exception as e:
            logger.warning("训练未执行: %s", e)


def run_optimization():
    """算法优化与报告；返回是否自检通过（回测有数据、无 exit(1)）。"""
    import subprocess
    opt_script = os.path.join(ROOT, "scripts", "optimize_algorithm_and_profitability.py")
    if not os.path.isfile(opt_script):
        return False
    try:
        r = subprocess.run(
            [sys.executable, opt_script],
            cwd=ROOT,
            timeout=3600,
        )
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        logger.warning("算法优化超时")
        return False
    except Exception as e:
        logger.warning("算法优化未执行: %s", e)
        return False


def one_cycle():
    """单轮：数据+训练 → 算法优化。优化通过则记录「可重启 DEMO 或停盘时刷新」."""
    logger.info("======== 开始一轮：数据+训练+算法优化 ========")
    run_data_and_train()
    ok = run_optimization()
    if ok:
        logger.info("本轮优化完成且自检通过；可临时重启 DEMO 或于每日停盘时间刷新参数/重启 DEMO。")
    else:
        logger.info("本轮优化自检未通过，见报告空项根因与建议；下一轮继续。")
    return ok


def main_loop(max_hours=20, interval_hours=None):
    """循环执行，每 interval_hours 跑一轮，总时长不超过 max_hours。"""
    interval_hours = interval_hours or INTERVAL_HOURS
    interval_sec = interval_hours * 3600
    end_time = time.time() + max_hours * 3600
    cycle = 0
    while time.time() < end_time:
        cycle += 1
        one_cycle()
        if time.time() >= end_time:
            break
        logger.info("下一轮 %.1f 小时后执行（间隔 %.1f h）", interval_hours, interval_hours)
        time.sleep(min(interval_sec, end_time - time.time()))
    logger.info("流水线结束（共 %d 轮）", cycle)


if __name__ == "__main__":
    # 单次运行：只跑一轮（供 cron 每 4h 调用）
    if os.environ.get("PIPELINE_ONE_SHOT"):
        one_cycle()
        sys.exit(0)
    # 常驻：每 INTERVAL_HOURS 跑一轮，总时长 20h
    main_loop(max_hours=20)
