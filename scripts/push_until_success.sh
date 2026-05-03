#!/usr/bin/env bash
# 推 origin（GitHub）+ gitee 直到成功。优先 PAT 顺序：
#   TIGERTRADE_GITHUB_PAT、CHENXI750328AI_GITHUB_PAT（tigertrade/.env → agentfuture/.env）
# 勿与江湖 chenxi750328 的 GITHUB_TOKEN 混用。
#
# 用法: cd tigertrade && bash scripts/push_until_success.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export GIT_TERMINAL_PROMPT=0
MAX="${PUSH_MAX_ATTEMPTS:-30}"

_load_env_files() {
  set -a
  [ -f "$REPO_ROOT/.env" ] && . "$REPO_ROOT/.env"
  [ -f "/home/cx/agentfuture/.env" ] && . "/home/cx/agentfuture/.env"
  set +a
}

_load_env_files

PAT="${TIGERTRADE_GITHUB_PAT:-${CHENXI750328AI_GITHUB_PAT:-}}"
GITHUB_PUSH_URL=""
if [ -n "$PAT" ]; then
  GITHUB_PUSH_URL="https://x-access-token:${PAT}@github.com/chenxi750328ai/tigertrade.git"
fi

GITEE_URL_RAW="$(git remote get-url gitee 2>/dev/null || true)"
GITEE_PUSH_URL=""
if [ -n "${GITEE_PRIVATE_TOKEN:-}" ] && [ -n "$GITEE_URL_RAW" ]; then
  GITEE_PUSH_URL="${GITEE_URL_RAW/https:\/\//https:\/\/oauth2:${GITEE_PRIVATE_TOKEN}@}"
fi

attempt=0
ok_origin=0
ok_gitee=0

while [ "$attempt" -lt "$MAX" ]; do
  attempt=$((attempt + 1))

  if [ "$ok_origin" -eq 0 ]; then
    echo "=== origin main 第 $attempt 次 ==="
    if [ -n "$GITHUB_PUSH_URL" ]; then
      if git push "$GITHUB_PUSH_URL" main; then
        echo "✅ Push origin（chenxi750328ai/tigertrade）成功"
        ok_origin=1
      else
        echo "⚠️ origin 失败，8s 后重试..."
        sleep 8
        continue
      fi
    else
      if git push origin main; then
        echo "✅ Push origin 成功"
        ok_origin=1
      else
        echo "⚠️ 未设置 TIGERTRADE_GITHUB_PAT / CHENXI750328AI_GITHUB_PAT 且 git push origin 失败。"
        sleep 8
        continue
      fi
    fi
  fi

  if [ -n "$GITEE_URL_RAW" ]; then
    if [ "$ok_gitee" -eq 0 ]; then
      echo "=== gitee main 第 $attempt 次 ==="
      if [ -n "$GITEE_PUSH_URL" ]; then
        if git push "$GITEE_PUSH_URL" main; then
          echo "✅ Push gitee 成功"
          ok_gitee=1
        else
          echo "⚠️ gitee 失败，8s 后重试..."
          sleep 8
          continue
        fi
      else
        if git push gitee main; then
          echo "✅ Push gitee 成功"
          ok_gitee=1
        else
          echo "⚠️ gitee 失败（可在 .env 设置 GITEE_PRIVATE_TOKEN 后重试）"
          sleep 8
          continue
        fi
      fi
    fi
  else
    ok_gitee=1
  fi

  if [ "$ok_origin" -eq 1 ] && [ "$ok_gitee" -eq 1 ]; then
    exit 0
  fi
done

echo "❌ 已达最大重试次数 $MAX"
exit 1
