#!/bin/bash
# 使用pytest和coverage运行完整测试（混合方案）
# 先运行测试收集覆盖率数据，然后生成报告

cd /home/cx/tigertrade

echo "=========================================="
echo "🧪 运行完整测试（pytest + coverage）"
echo "=========================================="
echo

# 清理之前的覆盖率数据
rm -rf .coverage htmlcov/

# 避免 ROS/launch_testing_ros 干扰：无 PYTHONPATH 时 pytest 可正常收集用例
# 若需保留 PYTHONPATH，可改用 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
echo "📊 运行测试并收集覆盖率数据..."
echo "⚠️  若环境有 ROS，已通过 unset PYTHONPATH 避免 pytest 插件冲突"
echo

# 第一步：使用 coverage.py 运行 pytest（无 ROS 环境）
unset PYTHONPATH
python -m coverage run --source=src -m pytest tests/ -v --tb=short -x

# 检查退出码
EXIT_CODE=$?

# 第二步：生成覆盖率报告
echo
echo "=========================================="
echo "📈 生成覆盖率报告"
echo "=========================================="
python -m coverage report --show-missing | tail -30

echo
echo "生成HTML覆盖率报告..."
python -m coverage html

echo
echo "✅ 测试完成！"
echo "📊 HTML覆盖率报告已生成到: htmlcov/index.html"
echo
echo "覆盖率统计:"
python -m coverage report --include="src/*" | tail -5

# 返回pytest的退出码
exit $EXIT_CODE
