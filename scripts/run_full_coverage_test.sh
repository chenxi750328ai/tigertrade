#!/bin/bash
# 运行完整的覆盖率测试并生成报告

echo "🚀 开始运行tiger1.py完整覆盖率测试..."

# 切换到tigertrade目录
cd /home/cx/tigertrade

# 清理之前的覆盖率数据
rm -rf .coverage htmlcov/

# 运行测试并收集覆盖率数据
echo "📊 运行测试并收集覆盖率数据..."
python -m coverage run --source=. --include="tiger1.py" test_tiger1_full_coverage.py

# 生成文本报告
echo ""
echo "📈 代码覆盖率报告:"
python -m coverage report --include="tiger1.py" --show-missing

# 生成HTML报告
echo ""
echo "📄 生成HTML覆盖率报告..."
python -m coverage html --include="tiger1.py" -d htmlcov

echo ""
echo "✅ 测试完成！"
echo "📊 HTML覆盖率报告已生成到: htmlcov/index.html"
echo ""
echo "覆盖率统计:"
python -m coverage report --include="tiger1.py" | tail -3
