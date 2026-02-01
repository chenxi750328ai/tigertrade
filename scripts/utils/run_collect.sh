#!/bin/bash
# 便捷运行脚本 - 数据采集
# 自动处理输出缓冲问题

# 设置无缓冲输出
export PYTHONUNBUFFERED=1

# 切换到脚本目录
cd "$(dirname "$0")/src"

# 运行脚本，传递所有参数
echo "=================================="
echo "🚀 启动数据采集"
echo "=================================="
echo ""

python -u collect_large_dataset.py "$@"

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "=================================="
    echo "✅ 数据采集完成！"
    echo "=================================="
else
    echo "=================================="
    echo "❌ 数据采集失败 (退出码: $exit_code)"
    echo "=================================="
fi

exit $exit_code
