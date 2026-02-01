#!/usr/bin/env python3
"""
将API配置问题添加到RAG系统
这是一个关键问题，需要记录在知识库中供所有agents参考
"""

import os
import requests
import json
import time
from datetime import datetime


def check_rag_service(rag_url):
    """
    检查RAG服务是否运行
    """
    try:
        health_response = requests.get(f"{rag_url}/../health", timeout=5)
        return health_response.status_code == 200
    except requests.exceptions.RequestException:
        # 尝试稍等后再次检查
        time.sleep(2)
        try:
            health_response = requests.get(f"{rag_url}/../health", timeout=5)
            return health_response.status_code == 200
        except requests.exceptions.RequestException:
            return False


def add_api_configuration_issue():
    """
    添加API配置问题到RAG系统
    """
    rag_url = "http://localhost:8000/api/v1"
    
    # 检查RAG服务是否运行
    if not check_rag_service(rag_url):
        print("❌ RAG服务未运行，请先启动服务")
        print("   启动命令: cd /home/cx/rag_system && python app/main.py")
        return False
    
    # API配置问题描述
    issue_content = """
紧急发现：Tiger API配置问题

问题根源：
所有之前的"真实数据"实际上都是Mock数据。根本原因是配置文件中的凭证都是占位符：
- tiger_id=demoid
- tiger_account=democount
- private_key_path=./demoprivatekey

影响范围：
1. 之前所有的数据采集：全部使用Mock数据
2. 之前的模型训练：全部基于Mock数据
3. 高准确率问题：Mock数据导致特征简单、模式明显

异常现象解释：
- 准确率98-99%：Mock数据过于规律，价格线性递增
- 特征全是0或常量：Mock数据生成算法简单，没有真实的市场波动
- API显示"初始化成功"但用Mock数据：凭证无效，程序静默回退到Mock数据

解决方案：
1. 获取真实Tiger API凭证（推荐）
2. 使用其他数据源
3. 改进Mock数据（临时方案）

关键经验教训：
- 配置验证至关重要，不仅检查文件存在，还要检查内容有效性
- 数据源必须明确验证，不能仅看日志，必须检查实际数据特征
- 问题追踪要深入，API初始化"成功"不代表可用
    """
    
    # 创建文档对象
    document = {
        "id": f"issue-tiger-api-config-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "content": issue_content.strip(),
        "metadata": {
            "type": "lesson_learned",
            "title": "Tiger API配置问题 - 关键发现",
            "tags": ["api", "configuration", "tiger-trade", "mock-data", "critical-issue"],
            "date": datetime.now().strftime("%Y-%m-%d"),
            "author": "proper_agent_v2",
            "project": "TigerTrade",
            "severity": "critical"
        }
    }
    
    # 发送到RAG系统
    try:
        response = requests.post(
            f"{rag_url}/documents",
            json=document,
            timeout=10
        )
        
        if response.status_code == 201:
            print(f"✅ API配置问题已成功添加到RAG系统")
            print(f"   文档ID: {document['id']}")
            return True
        else:
            print(f"❌ 添加文档失败: {response.status_code}")
            print(f"   响应: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 添加文档时发生错误: {str(e)}")
        return False


def add_solution_lesson():
    """
    添加解决方案经验到RAG系统
    """
    rag_url = "http://localhost:8000/api/v1"
    
    # 检查服务
    if not check_rag_service(rag_url):
        print("❌ RAG服务未运行")
        return False
    
    # 解决方案经验
    solution_content = """
API配置验证检查清单

验证Tiger API配置是否有效的完整检查清单：

1. 检查配置文件是否存在真实凭证
   - cat /home/cx/openapicfg_dem/tiger_openapi_config.properties
   - 确认tiger_id、tiger_account、private_key_path是真实值而非占位符

2. 检查关键字段是否包含占位符
   - grep -E "demo|placeholder|fake" /home/cx/openapicfg_dem/*.properties
   - 如果有匹配项则配置无效

3. 检查private key文件
   - ls -la /home/cx/openapicfg_dem/*.pem
   - 确认文件存在且不是示例文件

4. 实际测试API连接
   ```python
   from tigeropen.tiger_open_config import get_client_config
   from tigeropen.quote.quote_client import QuoteClient
   
   config = get_client_config('/home/cx/openapicfg_dem/')
   client = QuoteClient(config)
   
   # 实际调用API验证
   try:
       quote = client.get_market_quote(symbols=['SIL2503.US'])
       if quote:
           print("✅ API连接正常")
   except Exception as e:
       print(f"❌ API连接失败: {e}")
   ```

5. 验证获取的数据是否为真实数据
   - 检查时间戳合理性（不应是1970年或未来时间）
   - 检查价格波动性（不应是常量或线性变化）
   - 检查成交量数据（不应是0或常量）
    """
    
    # 创建文档对象
    document = {
        "id": f"lesson-api-verification-checklist-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "content": solution_content.strip(),
        "metadata": {
            "type": "guide",
            "title": "API配置验证检查清单",
            "tags": ["api", "verification", "checklist", "tiger-trade", "configuration"],
            "date": datetime.now().strftime("%Y-%m-%d"),
            "author": "proper_agent_v2",
            "project": "TigerTrade"
        }
    }
    
    # 发送到RAG系统
    try:
        response = requests.post(
            f"{rag_url}/documents",
            json=document,
            timeout=10
        )
        
        if response.status_code == 201:
            print(f"✅ API验证检查清单已成功添加到RAG系统")
            print(f"   文档ID: {document['id']}")
            return True
        else:
            print(f"❌ 添加文档失败: {response.status_code}")
            print(f"   响应: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 添加文档时发生错误: {str(e)}")
        return False


def main():
    """主函数"""
    print("📚 将API配置问题和解决方案添加到RAG系统")
    print("="*70)
    print("记录关键问题和解决方案，供所有agents参考")
    print("="*70)
    
    # 1. 添加API配置问题
    print("\n1️⃣ 添加API配置问题到RAG系统...")
    issue_added = add_api_configuration_issue()
    
    # 2. 添加解决方案经验
    print("\n2️⃣ 添加API验证检查清单到RAG系统...")
    solution_added = add_solution_lesson()
    
    print("\n" + "="*70)
    if issue_added and solution_added:
        print("✅ 所有关键信息已成功添加到RAG系统")
        print("   其他agents现在可以检索到这些问题和解决方案")
        print("   避免重复犯同样的错误")
    else:
        print("⚠️ 部分信息未能添加到RAG系统")
        print("   请检查RAG服务是否正常运行")
    print("="*70)


if __name__ == "__main__":
    main()