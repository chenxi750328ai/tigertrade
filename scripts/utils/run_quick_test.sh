#!/bin/bash
# 快速测试脚本 - 验证输出是否正常显示

export PYTHONUNBUFFERED=1

echo "=================================="
echo "🧪 输出显示测试"
echo "=================================="
echo ""

echo "测试1: 配置显示测试"
echo "----------------------------------"
cd "$(dirname "$0")/src"
python -u config.py | head -30
echo ""

echo "测试2: 实时输出测试"
echo "----------------------------------"
python -u -c "
import time
print('开始测试实时输出...')
for i in range(5):
    print(f'  [{i+1}/5] 消息显示测试... ', end='', flush=True)
    time.sleep(0.5)
    print('✓')
print('实时输出测试完成！')
"
echo ""

echo "测试3: 快速数据采集测试（1天数据）"
echo "----------------------------------"
python -u collect_large_dataset.py --days 1 --max-records 1000
echo ""

echo "=================================="
echo "✅ 所有测试完成！"
echo "=================================="
echo ""
echo "如果您能看到上面所有的输出，说明配置正确！"
echo ""
