#!/bin/bash
# 便捷运行脚本 - 模型训练
# 自动处理输出缓冲问题

# 设置无缓冲输出
export PYTHONUNBUFFERED=1

# 切换到脚本目录
cd "$(dirname "$0")/src"

# 运行脚本，传递所有参数
echo "=================================="
echo "🎓 启动模型训练"
echo "=================================="
echo ""

python -u train_with_detailed_logging.py "$@"

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "=================================="
    echo "✅ 训练完成！"
    echo "=================================="
else
    echo "=================================="
    echo "❌ 训练失败 (退出码: $exit_code)"
    echo "=================================="
fi

exit $exit_code
