#!/bin/bash
# 运行所有测试并启动20小时DEMO

set -e

echo "=========================================="
echo "🧪 步骤1: 运行所有测试"
echo "=========================================="

cd /home/cx/tigertrade

# 运行核心测试
echo "运行执行器模块测试..."
python -m coverage run --source=src/executor tests/test_executor_modules.py tests/test_executor_100_coverage.py
python -m coverage report --include="src/executor/*" --show-missing

echo ""
echo "运行订单执行测试..."
python tests/test_order_execution_real.py

echo ""
echo "运行集成测试..."
python tests/test_run_moe_demo_integration.py

echo ""
echo "✅ 所有测试通过！"
echo ""

echo "=========================================="
echo "🚀 步骤2: 启动20小时DEMO运行"
echo "=========================================="

# 启动DEMO
python scripts/run_moe_demo.py

echo ""
echo "✅ DEMO运行完成！"
