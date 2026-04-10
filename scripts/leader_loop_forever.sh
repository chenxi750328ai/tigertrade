#!/usr/bin/env bash
# 后台循环：按间隔反复执行 leader_routine_tick.sh（数据 + 跟进信号 + 可选 Webhook）
# 用法:
#   LEADER_LOOP_INTERVAL_SEC=900 nohup bash scripts/leader_loop_forever.sh >> logs/leader_loop_forever.log 2>&1 &
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export TIGERTRADE_ROOT="$REPO_ROOT"
mkdir -p logs

INTERVAL_SEC="${LEADER_LOOP_INTERVAL_SEC:-900}"
if [ "${INTERVAL_SEC}" -lt 300 ] 2>/dev/null; then
  echo "⚠️ LEADER_LOOP_INTERVAL_SEC=${INTERVAL_SEC} 过小，已抬到 300"
  INTERVAL_SEC=300
fi

echo "leader_loop_forever: REPO_ROOT=${REPO_ROOT} INTERVAL_SEC=${INTERVAL_SEC} 启动 $(date -Iseconds)"

while true; do
  echo "---- $(date -Iseconds) tick ----"
  bash "$REPO_ROOT/scripts/leader_routine_tick.sh" || echo "leader_routine_tick exit=$?"
  sleep "${INTERVAL_SEC}"
done
