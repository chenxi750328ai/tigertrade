#!/bin/bash
# 在 GitHub 上创建 tigertrade 仓库并推送当前分支（需 .env 中有 GITHUB_USER/GITHUB_PAT 或 GITHUB_USERNAME/GITHUB_TOKEN）
set -e
cd /home/cx/tigertrade

# 从 /home/cx/.env 或 /home/cx/agentfuture/.env 读取凭证（不打印 PAT）
if [ -f /home/cx/.env ]; then
  source /home/cx/.env 2>/dev/null || true
fi
if [ -f /home/cx/agentfuture/.env ]; then
  source /home/cx/agentfuture/.env 2>/dev/null || true
fi
USER="${GITHUB_USER:-$GITHUB_USERNAME}"
PAT="${GITHUB_PAT:-$GITHUB_TOKEN}"
REPO_NAME="tigertrade"

if [ -z "$USER" ] || [ -z "$PAT" ]; then
  echo "❌ 请在 /home/cx/.env 中设置 GITHUB_USER 和 GITHUB_PAT（或 GITHUB_USERNAME/GITHUB_TOKEN）"
  exit 1
fi

echo "📦 使用 GitHub API 创建仓库: $USER/$REPO_NAME"
RESP=$(curl -s -w "\n%{http_code}" -X POST -H "Authorization: token $PAT" -H "Accept: application/vnd.github.v3+json" \
  "https://api.github.com/user/repos" -d "{\"name\":\"$REPO_NAME\",\"private\":false,\"description\":\"TigerTrade AI-driven futures trading\"}")

HTTP_CODE=$(echo "$RESP" | tail -1)
BODY=$(echo "$RESP" | sed '$d')

if [ "$HTTP_CODE" = "201" ]; then
  echo "✅ 仓库已创建"
elif [ "$HTTP_CODE" = "422" ]; then
  echo "ℹ️ 仓库已存在，继续推送"
else
  echo "❌ 创建失败 HTTP $HTTP_CODE: $BODY"
  exit 1
fi

# 推送（使用带 PAT 的 URL，仅本次）
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "📤 推送分支: $BRANCH"
if ! git push -u origin "$BRANCH" 2>/dev/null; then
  git push "https://${USER}:${PAT}@github.com/${USER}/${REPO_NAME}.git" "$BRANCH"
  git branch --set-upstream-to=origin/"$BRANCH" "$BRANCH" 2>/dev/null || true
fi
echo "✅ 推送完成: https://github.com/${USER}/${REPO_NAME}"
