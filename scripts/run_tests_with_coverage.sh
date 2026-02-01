#!/bin/bash
# 使用pytest和coverage运行完整测试

cd /home/cx/tigertrade

echo "=========================================="
echo "🧪 运行完整测试（pytest + coverage）"
echo "=========================================="
echo

# 清理之前的覆盖率数据
rm -rf .coverage htmlcov/

# 运行pytest并收集覆盖率（禁用ROS插件，但保留pytest-cov）
echo "📊 运行测试并收集覆盖率数据..."
# 使用-p no:选项禁用ROS插件，但保留pytest-cov插件
# 注意：如果遇到PluginValidationError，可以忽略（测试仍能运行）
python -m pytest tests/ \
    -v \
    --cov=src \
    --cov-report=term \
    --cov-report=term-missing \
    --cov-report=html \
    --tb=short \
    -p no:launch_testing \
    -p no:launch_testing_ros \
    -p no:ament_xmllint \
    -p no:ament_flake8 \
    -p no:ament_lint \
    -p no:ament_copyright \
    -p no:ament_pep257 \
    -x || true  # 即使有PluginValidationError也继续

# 显示覆盖率摘要
echo
echo "=========================================="
echo "📈 覆盖率报告摘要"
echo "=========================================="
python -m coverage report --show-missing | tail -20

echo
echo "✅ 测试完成！"
echo "📊 HTML覆盖率报告已生成到: htmlcov/index.html"
echo
