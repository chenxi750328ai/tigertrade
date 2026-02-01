#!/bin/bash
# 使用pytest和coverage运行完整测试（禁用ROS插件）

cd /home/cx/tigertrade

echo "=========================================="
echo "🧪 运行完整测试（pytest + coverage）"
echo "=========================================="
echo

# 清理之前的覆盖率数据
rm -rf .coverage htmlcov/

# 运行pytest并收集覆盖率（使用coverage.py包装pytest）
echo "📊 运行测试并收集覆盖率数据..."
echo "⚠️  注意：由于ROS插件冲突，使用coverage.py包装pytest来收集覆盖率"
echo

# 使用coverage.py运行pytest（这样可以避免ROS插件问题）
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m coverage run --source=src -m pytest tests/ -v --tb=short -x

# 检查退出码
EXIT_CODE=$?

# 生成覆盖率报告
echo
echo "生成覆盖率报告..."
python -m coverage report --show-missing | tail -30
python -m coverage html

echo
echo "✅ 测试完成！"
echo "📊 HTML覆盖率报告已生成到: htmlcov/index.html"
echo
echo "覆盖率统计:"
python -m coverage report --include="src/*" | tail -5

# 返回pytest的退出码
exit $EXIT_CODE
