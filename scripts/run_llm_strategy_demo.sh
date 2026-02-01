#!/bin/bash
# 运行LLM多时间尺度策略（DEMO账户）

cd /home/cx/tigertrade

echo "🚀 启动LLM多时间尺度策略（DEMO账户）..."
echo "=" | head -c 70
echo ""

# 运行策略（使用llm模式）
python -m src.tiger1 llm
