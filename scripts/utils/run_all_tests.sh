#!/bin/bash

# Tiger Trade 完整测试套件
# 自动运行所有测试并生成报告

echo "=================================="
echo "🚀 Tiger Trade 测试套件"
echo "=================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 切换到src目录
cd /home/cx/tigertrade/src

# 1. 运行策略测试
echo -e "${YELLOW}步骤 1/3: 运行策略测试...${NC}"
echo "测试次数: ${1:-10}"
echo ""
python test_strategies.py ${1:-10}

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 策略测试完成${NC}"
else
    echo -e "${RED}❌ 策略测试失败${NC}"
    exit 1
fi

echo ""
echo "=================================="
echo ""

# 2. 获取历史数据
echo -e "${YELLOW}步骤 2/3: 获取历史数据...${NC}"
echo "获取天数: ${2:-7}"
echo ""
python fetch_more_data.py ${2:-7}

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 数据获取完成${NC}"
else
    echo -e "${RED}❌ 数据获取失败${NC}"
    # 数据获取失败不中断流程
fi

echo ""
echo "=================================="
echo ""

# 3. 生成报告
echo -e "${YELLOW}步骤 3/3: 生成测试报告...${NC}"
echo ""
python generate_report.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 报告生成完成${NC}"
else
    echo -e "${RED}❌ 报告生成失败${NC}"
    exit 1
fi

echo ""
echo "=================================="
echo -e "${GREEN}✅ 所有测试完成！${NC}"
echo "=================================="
echo ""
echo "📁 结果文件位置:"
echo "  - 策略测试结果: /home/cx/trading_data/strategy_tests/"
echo "  - 历史数据: /home/cx/trading_data/historical/"
echo "  - 分析报告: /home/cx/trading_data/reports/"
echo "  - 总结文档: /home/cx/tigertrade/TESTING_SUMMARY.md"
echo ""
echo "💡 提示:"
echo "  - 运行测试: ./run_all_tests.sh [测试次数] [获取天数]"
echo "  - 示例: ./run_all_tests.sh 20 14  # 每个策略测试20次，获取14天数据"
echo ""
