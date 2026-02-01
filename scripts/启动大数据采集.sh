#!/bin/bash

# 大数据采集启动脚本
# 用途：采集大量真实市场数据用于模型训练

set -e  # 遇到错误立即退出

echo "========================================================================"
echo "大数据采集启动"
echo "========================================================================"
echo ""

# 从RAG检索约束（如果RAG服务运行）
echo "📚 检索数据采集规则..."
curl -s -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"数据采集验证规则 API初始化","top_k":3}' 2>/dev/null | \
  python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if data.get('total', 0) > 0:
        print('\\n从RAG检索到的规则:')
        for item in data['results']:
            print(f\"  • {item['content'][:80]}...\")
except:
    pass
" || echo "  (RAG服务未运行，跳过)"

echo ""
echo "========================================================================"
echo "配置参数"
echo "========================================================================"

# 配置参数
SYMBOL=${1:-"SIL2603"}  # 期货代码，默认SIL2603
DAYS=${2:-90}            # 采集天数，默认90天
USE_REAL_API="true"      # 强制使用真实API

echo "期货代码: $SYMBOL"
echo "采集天数: $DAYS"
echo "使用真实API: $USE_REAL_API"
echo ""

# 检查API配置
CONFIG_PATH="/home/cx/openapicfg_dem"
if [ ! -d "$CONFIG_PATH" ]; then
    echo "❌ API配置目录不存在: $CONFIG_PATH"
    exit 1
fi
echo "✅ API配置目录: $CONFIG_PATH"
echo ""

# 创建输出目录
OUTPUT_DIR="/home/cx/tigertrade/data"
mkdir -p "$OUTPUT_DIR"
echo "✅ 输出目录: $OUTPUT_DIR"
echo ""

echo "========================================================================"
echo "开始数据采集"
echo "========================================================================"
echo ""

# 执行数据采集
cd /home/cx/tigertrade

python3 src/collect_large_dataset.py \
  --symbol "$SYMBOL" \
  --days "$DAYS" \
  --real-api \
  --output "$OUTPUT_DIR" \
  2>&1 | tee "logs/collect_$(date +%Y%m%d_%H%M%S).log"

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 数据采集完成"
    echo "========================================================================"
    echo ""
    
    # 验证数据
    echo "📊 数据验证:"
    ls -lh "$OUTPUT_DIR"/*.pkl 2>/dev/null || echo "  未找到PKL文件"
    
    # 检查是否有Mock数据痕迹
    echo ""
    echo "🔍 检查Mock数据痕迹:"
    grep -i "mock\|模拟" "logs/collect_$(date +%Y%m%d)_"*.log | tail -5 || echo "  ✅ 未发现Mock数据"
    
else
    echo "❌ 数据采集失败 (退出码: $EXIT_CODE)"
    echo "========================================================================"
    echo ""
    echo "请检查日志文件"
fi

