#!/usr/bin/env bash
# 推送重试直到成功（陈正霞例行：不得让陈正落重复要求）
# 用法: 在 tigertrade 或 agentfuture 仓库根目录执行 bash scripts/push_until_success.sh
# 需已配置 HTTPS 凭据/凭证助手，或改用 SSH remote；无凭据时本脚本会反复失败。
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export GIT_TERMINAL_PROMPT=0
MAX="${PUSH_MAX_ATTEMPTS:-30}"
attempt=0
ok_origin=0
ok_gitee=0
while [ "$attempt" -lt "$MAX" ]; do
  attempt=$((attempt + 1))
  if [ "$ok_origin" -eq 0 ]; then
    echo "=== origin main 第 $attempt 次 ==="
    if git push origin main; then
      echo "✅ Push origin 成功"
      ok_origin=1
    else
      echo "⚠️ origin 失败，8s 后重试..."
      sleep 8
      continue
    fi
  fi
  if git remote get-url gitee &>/dev/null; then
    if [ "$ok_gitee" -eq 0 ]; then
      echo "=== gitee main 第 $attempt 次 ==="
      if git push gitee main; then
        echo "✅ Push gitee 成功"
        ok_gitee=1
      else
        echo "⚠️ gitee 失败，8s 后重试 origin 已 OK 仅重试 gitee..."
        sleep 8
        continue
      fi
    fi
  else
    ok_gitee=1
  fi
  if [ "$ok_origin" -eq 1 ] && [ "$ok_gitee" -eq 1 ]; then
    exit 0
  fi
done
echo "❌ 已达最大重试次数 $MAX；请在本机配置 git 凭据或 SSH 后重跑本脚本。"
exit 1
