#!/usr/bin/env bash
# 夜间一次性：全量 pytest（not real_api）+ DFX 门禁。供 crontab 调用，避免每 15 分钟跑全量。
# 例: 0 21 * * * /path/to/tigertrade/scripts/leader_routine_nightly_heavy.sh >> /path/to/tigertrade/logs/nightly_heavy.log 2>&1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export TIGERTRADE_ROOT="$REPO_ROOT"
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi

echo "=== $(date -Iseconds) leader_routine_nightly_heavy ==="
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 scripts/verify_design_completeness_and_dfx.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -m "not real_api" -q --tb=line
echo "=== done ==="
