#!/usr/bin/env python3
"""
记录项目错误到RAG系统
"""
import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000/api/v1"

def check_service():
    """检查RAG服务是否运行"""
    try:
        r = requests.get("http://localhost:8000/health", timeout=2)
        if r.status_code == 200:
            print(f"✅ RAG服务运行中")
            return True
    except Exception as e:
        print(f"❌ RAG服务未运行: {e}")
        return False

def record_error():
    """记录CI测试问题错误"""
    
    error_doc = {
        "id": f"error-ci-test-issue-{datetime.now().strftime('%Y%m%d')}",
        "content": """
项目错误记录：CI测试问题

错误时间：2026年1月28日
错误类型：严重性：高 - CI/CD流程问题

问题描述：
1. CI测试失败被掩盖：CI配置中使用`|| echo "⚠️ 跳过"`掩盖测试失败，即使pytest返回非零退出码（测试失败），`|| echo`也会让命令返回0（成功），GitHub Actions会认为测试通过了。

2. 测试报告是假的：测试报告硬编码为✅，无论测试是否通过，没有真正检查测试结果，没有使用pytest的JUnit XML报告。

3. 测试用例与架构不匹配：test_run_moe_demo_integration.py检查的是旧代码模式，检查run_moe_demo.py中是否有place_tiger_order调用，但run_moe_demo.py已经重构，使用OrderExecutor，不再直接调用place_tiger_order。

根本原因：
1. CI配置设计错误：使用`|| echo`掩盖失败，而不是真正处理错误
2. 测试用例过时：测试用例没有随架构重构更新
3. 测试报告虚假：硬编码结果，没有真正检查测试状态

影响：
1. 代码质量问题：测试失败被掩盖，导致代码质量问题无法及时发现
2. 架构重构后测试失效：测试用例没有更新，无法验证新架构
3. CI流程失效：CI流程无法真正保证代码质量

修复方案：
1. 修复CI配置：移除`|| echo`，让测试失败真正失败，使用pytest的JUnit XML报告
2. 修复测试报告：使用真实的测试结果生成报告，解析JUnit XML报告
3. 更新测试用例：更新test_run_moe_demo_integration.py，检查新架构（OrderExecutor、TradingExecutor），移除对旧代码模式的检查

教训：
1. 不要掩盖测试失败：使用`|| echo`会掩盖问题，应该让失败真正失败
2. 测试用例需要随架构更新：架构重构后，测试用例必须同步更新
3. 测试报告必须真实：不能硬编码结果，必须真正检查测试状态

责任人：AI助手
状态：已发现，部分修复（添加了真实测试，修复了代码问题），待修复（CI配置和测试用例更新）
        """,
        "metadata": {
            "type": "lesson_learned",
            "tags": ["ci", "testing", "error", "lesson", "high-priority", "tigertrade"],
            "title": "CI测试问题：测试失败被掩盖",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "project": "tigertrade",
            "priority": "high",
            "severity": "high"
        }
    }
    
    try:
        r = requests.post(f"{BASE_URL}/documents", 
                        json=error_doc, 
                        timeout=10)
        if r.status_code in [200, 201]:
            print(f"✅ 错误记录已添加到RAG系统")
            print(f"   文档ID: {error_doc['id']}")
            return True
        else:
            print(f"❌ 添加失败: HTTP {r.status_code}")
            print(f"   响应: {r.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ 添加失败: {e}")
        # 如果RAG服务未运行，保存到文件
        error_file = f"/home/cx/tigertrade/docs/rag_pending_errors_{datetime.now().strftime('%Y%m%d')}.json"
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_doc, f, ensure_ascii=False, indent=2)
        print(f"   已保存到文件: {error_file}")
        print(f"   待RAG服务启动后，可以使用以下命令添加：")
        print(f"   curl -X POST http://localhost:8000/api/v1/documents -H 'Content-Type: application/json' -d @{error_file}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("记录项目错误到RAG系统")
    print("=" * 70)
    
    if check_service():
        record_error()
    else:
        print("\n⚠️  RAG服务未运行，保存到文件待后续添加")
        record_error()  # 会保存到文件
