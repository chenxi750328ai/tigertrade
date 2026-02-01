#!/bin/bash
# 运行20小时DEMO交易策略

set -e

echo "=========================================="
echo "启动20小时DEMO交易策略"
echo "=========================================="

cd /home/cx/tigertrade

# 1. 先运行核心测试确保没问题
echo -e "\n[1/3] 运行核心测试..."
python -m unittest tests.test_feature_risk_management tests.test_feature_order_execution.TestFeatureOrderExecutionWithMock -q || {
    echo "❌ 核心测试失败，请先修复测试错误"
    exit 1
}
echo "✅ 核心测试通过"

# 2. 检查API配置
echo -e "\n[2/3] 检查DEMO账户配置..."
if [ ! -f "./openapicfg_dem/tiger_openapi_config.properties" ]; then
    echo "❌ DEMO账户配置文件不存在"
    exit 1
fi
echo "✅ 配置文件存在"

# 3. 启动20小时运行
echo -e "\n[3/3] 启动20小时交易策略..."
echo "=========================================="
echo "开始时间: $(date)"
echo "预计结束时间: $(date -d '+20 hours')"
echo "=========================================="

# 后台运行，输出到日志文件
nohup python scripts/run_moe_demo.py > logs/demo_20h_$(date +%Y%m%d_%H%M%S).log 2>&1 &
DEMO_PID=$!

echo "✅ DEMO策略已启动（PID: $DEMO_PID）"
echo "📝 日志文件: logs/demo_20h_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "监控命令:"
echo "  tail -f logs/demo_20h_*.log"
echo "  或"
echo "  ps aux | grep run_moe_demo"
echo ""
echo "停止命令:"
echo "  kill $DEMO_PID"
