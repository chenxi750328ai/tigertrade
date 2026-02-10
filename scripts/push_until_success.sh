#!/usr/bin/env bash
# 推送重试直到成功（陈正霞例行：不得让陈正落重复要求）
# 用法: 在 tigertrade 或 agentfuture 仓库根目录执行 ./scripts/push_until_success.sh
set -e
cd "$(git rev-parse --show-toplevel)"
attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "=== push attempt $attempt ==="
  if git push origin main 2>&1; then
    echo "✅ Push origin 成功"
    if git remote get-url gitee &>/dev/null; then
      echo "=== 同步推送到 Gitee ==="
      git push gitee main 2>&1 && echo "✅ Push gitee 成功" || echo "⚠️ Push gitee 失败（可稍后单独执行 git push gitee main）"
    fi
    exit 0
  fi
  echo "⏳ 2 秒后重试..."
  sleep 2
done
