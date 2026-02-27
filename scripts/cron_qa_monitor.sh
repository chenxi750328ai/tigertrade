#!/bin/bash
# QA 定时分析：供 crontab 每日或定期调用，生成进度与质量监控报告并追加日志
# 用法: bash scripts/cron_qa_monitor.sh
# 建议 crontab（每日 9:00 北京时间 = 1:00 UTC）:
#   0 1 * * * cd /home/cx/tigertrade && bash scripts/cron_qa_monitor.sh >> logs/qa_monitor.log 2>&1
# 或与例行错开（如每日 2:00 UTC）:
#   0 2 * * * cd /home/cx/tigertrade && bash scripts/cron_qa_monitor.sh >> logs/qa_monitor.log 2>&1

cd /home/cx/tigertrade
mkdir -p logs
exec python3 scripts/qa_progress_quality_monitor.py --append-log
