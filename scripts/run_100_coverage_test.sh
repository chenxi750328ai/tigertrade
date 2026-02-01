#!/bin/bash
# 运行100%覆盖率测试

cd /home/cx/tigertrade

echo "=========================================="
echo "运行100%覆盖率测试"
echo "=========================================="
echo ""

# 清理之前的覆盖率数据
rm -rf .coverage htmlcov/

# 运行所有测试并收集覆盖率
echo "📊 运行测试并收集覆盖率数据..."
python -m coverage run --source=. --include="tiger1.py" run_test_clean.py

# 生成报告
echo ""
echo "=========================================="
echo "代码覆盖率报告"
echo "=========================================="
python -m coverage report --include="tiger1.py" --show-missing

# 生成HTML报告
echo ""
echo "生成HTML覆盖率报告..."
python -m coverage html --include="tiger1.py" -d htmlcov

echo ""
echo "✅ 测试完成！"
echo "📊 HTML覆盖率报告: htmlcov/index.html"
echo ""

# 显示覆盖率统计
python -m coverage report --include="tiger1.py" | tail -3
