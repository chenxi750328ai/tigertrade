#!/bin/bash
# CI测试脚本 - 使用pytest-cov进行完整的测试和覆盖率报告

set -e  # 遇到错误立即退出

echo "=========================================="
echo "TigerTrade CI测试和覆盖率报告"
echo "=========================================="

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 运行Feature测试（业务需求验证）
echo -e "\n${YELLOW}[1/4] 运行Feature级测试...${NC}"
FEATURE_RESULT=0
python -m pytest tests/test_feature_*.py -v --tb=short || FEATURE_RESULT=$?

if [ $FEATURE_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ Feature测试通过${NC}"
else
    echo -e "${RED}❌ Feature测试失败${NC}"
fi

# 2. 运行代码级测试（技术逻辑验证）
echo -e "\n${YELLOW}[2/4] 运行代码级测试...${NC}"
UNIT_RESULT=0
python -m pytest tests/ -v --tb=short --ignore=tests/test_feature_*.py || UNIT_RESULT=$?

if [ $UNIT_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ 代码级测试通过${NC}"
else
    echo -e "${RED}❌ 代码级测试失败${NC}"
fi

# 3. 生成覆盖率报告
echo -e "\n${YELLOW}[3/4] 生成覆盖率报告...${NC}"
python -m pytest tests/ -v --cov=src \
    --cov-report=term-missing \
    --cov-report=html:htmlcov \
    --cov-report=xml:coverage.xml \
    --cov-fail-under=20 \
    --tb=short || true  # 不因覆盖率失败而退出

echo -e "${GREEN}✅ 覆盖率报告已生成${NC}"
echo "   - HTML报告: htmlcov/index.html"
echo "   - XML报告: coverage.xml"

# 4. 显示覆盖率摘要
echo -e "\n${YELLOW}[4/4] 覆盖率摘要${NC}"
python -m coverage report -m --include="src/executor/*,src/api_adapter.py" | head -20

# 5. 汇总结果
echo -e "\n=========================================="
echo "测试结果汇总"
echo "=========================================="

if [ $FEATURE_RESULT -eq 0 ] && [ $UNIT_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ 所有测试通过${NC}"
    exit 0
else
    echo -e "${RED}❌ 部分测试失败${NC}"
    echo "  Feature测试: $([ $FEATURE_RESULT -eq 0 ] && echo '通过' || echo '失败')"
    echo "  代码级测试: $([ $UNIT_RESULT -eq 0 ] && echo '通过' || echo '失败')"
    exit 1
fi
