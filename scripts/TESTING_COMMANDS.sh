#!/bin/bash
# tiger1.py 测试常用命令脚本

echo "========================================="
echo "tiger1.py 测试工具箱"
echo "========================================="
echo ""

# 1. 运行所有测试
echo "1. 运行所有测试"
echo "   python ./run_test_clean.py"
echo ""

# 2. 查看覆盖率
echo "2. 查看覆盖率摘要"
echo "   coverage report tiger1.py"
echo ""

# 3. 查看详细未覆盖行
echo "3. 查看未覆盖的代码行"
echo "   coverage report tiger1.py --show-missing | less"
echo ""

# 4. 生成HTML报告
echo "4. 生成/更新HTML报告"
echo "   coverage html && echo '已生成: htmlcov/index.html'"
echo ""

# 5. 运行特定测试
echo "5. 运行特定阶段测试"
echo "   python -m unittest test_tiger1_phase3_coverage"
echo ""

# 6. 查看测试日志
echo "6. 查看最近测试日志"
echo "   tail -100 clean_test_output.log"
echo ""

# 7. 清理并重新测试
echo "7. 清理并重新运行测试"
echo "   rm -f .coverage coverage.json && python ./run_test_clean.py"
echo ""

# 8. 统计测试数量
echo "8. 统计所有测试用例数量"
echo "   grep -r 'def test_' test_*.py | wc -l"
echo ""

echo "========================================="
echo "快速执行命令："
echo "========================================="
echo ""
echo "查看当前覆盖率:"
echo "  cd /home/cx/tigertrade && coverage report tiger1.py"
echo ""
echo "打开HTML报告:"
echo "  cd /home/cx/tigertrade && firefox htmlcov/index.html"
echo ""

