#!/usr/bin/env bash
# Leader 例行：本机进程里跑的「一轮」工作 —— 可只跑轻量，也可串门禁/测试/优化/你自己的 hook。
#
# 固定顺序：
#   1) ROUTINE_PULSE_REFRESH=1 → write_routine_pulse（今日收益、QA、export_order_log、routine_pulse）
#   2) leader_routine_notify.py → 跟进信号 + 可选 Webhook/NOTIFY_CMD
#   3) 按 LEADER_ROUTINE_PROFILE 追加：DFX 门禁 / pytest / 轻量 optimize
#   4) 若设置了 LEADER_ROUTINE_POST_HOOK → 再 bash 执行你的脚本（合并数据、训练、pipeline 等）
#
# 环境变量（仓库根 .env 可自动加载）：
#   LEADER_ROUTINE_PROFILE=minimal|standard|heavy   默认 minimal；standard 加 DFX 门禁；heavy 再加 pytest + 轻量 optimize
#   LEADER_ROUTINE_WEBHOOK_URL / LEADER_ROUTINE_NOTIFY_CMD  见 leader_routine_notify.py
#   LEADER_ROUTINE_POST_HOOK=/path/to/your.sh        每轮结束后额外执行（工作目录=仓库根）
#   LEADER_ROUTINE_RUN_OPTIMIZE_LIGHT=1              即使 profile 非 heavy 也跑轻量 optimize（耗 CPU，慎用间隔）
#
# crontab 示例:
#   */15 * * * * /path/to/tigertrade/scripts/leader_routine_tick.sh >> /path/to/tigertrade/logs/leader_routine_tick.log 2>&1
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export TIGERTRADE_ROOT="$REPO_ROOT"
mkdir -p logs run docs/reports

# 与 tiger1 一致：加载 .env，便于只配一次 Webhook
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$REPO_ROOT/.env"
  set +a
fi

export ROUTINE_PULSE_REFRESH=1
python3 scripts/write_routine_pulse.py
python3 scripts/leader_routine_notify.py

PROFILE="${LEADER_ROUTINE_PROFILE:-minimal}"
echo "[leader_routine_tick] PROFILE=${PROFILE} $(date -Iseconds)"

_run_gate() {
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 scripts/verify_design_completeness_and_dfx.py || true
}

_run_pytest() {
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -m "not real_api" -q --tb=no || true
}

_run_opt_light() {
  ROUTINE_SKIP_SLOW_BACKTEST=1 ROUTINE_SKIP_PRO1_BACKTEST=1 ROUTINE_SELF_CHECK_SOFT=1 \
    python3 scripts/optimize_algorithm_and_profitability.py || true
}

case "$PROFILE" in
  standard|heavy)
    _run_gate
    ;;
esac
case "$PROFILE" in
  heavy)
    _run_pytest
    _run_opt_light
    ;;
esac

if [ "${LEADER_ROUTINE_RUN_OPTIMIZE_LIGHT:-0}" = "1" ] && [ "$PROFILE" != "heavy" ]; then
  _run_opt_light
fi

HOOK="${LEADER_ROUTINE_POST_HOOK:-}"
if [ -n "$HOOK" ] && [ -f "$HOOK" ]; then
  echo "[leader_routine_tick] POST_HOOK=$HOOK"
  bash "$HOOK" || echo "[leader_routine_tick] POST_HOOK exit=$?"
fi
