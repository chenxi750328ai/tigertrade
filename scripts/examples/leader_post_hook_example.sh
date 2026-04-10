#!/usr/bin/env bash
# 示例：供 LEADER_ROUTINE_POST_HOOK 指向；每轮 leader_routine_tick 末尾执行，工作目录=仓库根。
# 复制到自定义路径并改逻辑；勿把密钥写进仓库。
set -euo pipefail
echo "[leader_post_hook_example] OK $(date -Iseconds) TIGERTRADE_ROOT=${TIGERTRADE_ROOT:-}"
