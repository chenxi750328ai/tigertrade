#!/usr/bin/env bash
# 推送重试直到成功（陈老大例行：不得让陈老零重复要求）
# 用法: 在 tigertrade 或 agentfuture 仓库根目录执行 ./scripts/push_until_success.sh
set -e
cd "$(git rev-parse --show-toplevel)"
attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "=== push attempt $attempt ==="
  if git push origin main 2>&1; then
    echo "✅ Push 成功"
    exit 0
  fi
  echo "⏳ 2 秒后重试..."
  sleep 2
done
