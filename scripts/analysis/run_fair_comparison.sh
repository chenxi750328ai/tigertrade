#!/bin/bash
# 运行公平对比测试

cd /home/cx/tigertrade

# 查找训练数据文件
DATA_FILE=$(find /home/cx/trading_data -name "training_data_from_klines*.csv" -type f | head -1)

if [ -z "$DATA_FILE" ]; then
    echo "⚠️ 未找到训练数据文件，将使用默认路径"
    DATA_FILE=""
fi

echo "📊 开始公平模型对比测试"
echo "数据文件: ${DATA_FILE:-自动查找}"

# 运行对比测试
python scripts/analysis/fair_model_comparison.py \
    --data-file "$DATA_FILE" \
    --seq-lengths 10 50 100 \
    --epochs 50

echo "✅ 对比测试完成"
