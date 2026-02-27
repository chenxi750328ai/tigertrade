#!/bin/bash
# 运行20小时DEMO交易策略

set -e

echo "=========================================="
echo "启动20小时DEMO交易策略"
echo "=========================================="

cd /home/cx/tigertrade

# 确保 DEMO 可真实下单（sandbox 模式不检查，但 production 路径会用到）
export ALLOW_REAL_TRADING=1

# 1. 先运行核心测试确保没问题
echo -e "\n[1/3] 运行核心测试..."
python -m unittest tests.test_feature_risk_management tests.test_feature_order_execution.TestFeatureOrderExecutionWithMock -q || {
    echo "❌ 核心测试失败，请先修复测试错误"
    exit 1
}
echo "✅ 核心测试通过"

# 2. 检查API配置与 account
echo -e "\n[2/3] 检查DEMO账户配置..."
if [ ! -f "./openapicfg_dem/tiger_openapi_config.properties" ]; then
    echo "❌ DEMO账户配置文件不存在"
    exit 1
fi
echo "✅ 配置文件存在"
python3 -c "
from tigeropen.tiger_open_config import TigerOpenClientConfig
cfg = TigerOpenClientConfig(props_path='./openapicfg_dem')
acc = getattr(cfg, 'account', None)
if not acc:
    print('❌ openapicfg_dem 中 account 未配置，下单会报 1010')
    exit(1)
print('✅ account 已配置')
" || { echo "❌ account 校验失败"; exit 1; }

# 3. 启动20小时运行
echo -e "\n[3/3] 启动20小时交易策略..."

# 防重复：若已有 DEMO 在跑则跳过，避免多实例（多实例会共享账户但各自 current_position 独立→超限如52手）
EXISTING=$(pgrep -f "run_moe_demo.py" | head -1)
[ -z "$EXISTING" ] && EXISTING=$(pgrep -f "tiger1.py.*moe" | head -1)
if [ -n "$EXISTING" ]; then
  echo "⚠️ DEMO 已在运行（PID=$EXISTING），跳过本次启动。若要重启请先: pkill -f run_moe_demo; pkill -f 'tiger1.py.*moe'"
  exit 0
fi

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
echo ""
echo "异常订单检查（跑完后或定期执行）:"
echo "  bash scripts/run_anomaly_order_check.sh"
echo "  或由 cron 每 30 分钟执行（见 cron_routine_monitor.sh）"
echo ""
echo "异常订单检查（跑完后或定期执行）:"
echo "  bash scripts/run_anomaly_order_check.sh"
echo "  或由 cron 每 30 分钟执行，或查看 logs/routine_monitor.log（cron_routine_monitor 每半点会跑一次）"
