#!/bin/bash
# 统一的覆盖率测试脚本 - 支持unittest和pytest

set -e

echo "=========================================="
echo "TigerTrade 覆盖率测试（coverage.py + pytest-cov）"
echo "=========================================="

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 1. 使用coverage.py运行unittest测试（兼容现有测试）
echo -e "\n${YELLOW}[1/3] 运行unittest测试（coverage.py）...${NC}"
python -m coverage run -m unittest discover -s tests -p "test_*.py" -t . || true

# 2. 使用pytest-cov运行pytest测试（如果有）
echo -e "\n${YELLOW}[2/3] 运行pytest测试（pytest-cov）...${NC}"
python -m pytest tests/ --cov=src --cov-append --cov-report=term-missing -q || true

# 3. 生成统一报告
echo -e "\n${YELLOW}[3/3] 生成覆盖率报告...${NC}"

# HTML报告
python -m coverage html --include="src/*" -d htmlcov
echo -e "${GREEN}✅ HTML报告: htmlcov/index.html${NC}"

# XML报告（CI使用）
python -m coverage xml --include="src/*" -o coverage.xml
echo -e "${GREEN}✅ XML报告: coverage.xml${NC}"

# 终端报告
echo -e "\n${YELLOW}覆盖率摘要（关键模块）:${NC}"
python -m coverage report -m --include="src/executor/*,src/api_adapter.py"

echo -e "\n${YELLOW}总体覆盖率:${NC}"
python -m coverage report --include="src/*" | tail -3
