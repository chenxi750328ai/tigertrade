#!/usr/bin/env python3
"""
使用新协议v2.1.0功能的脚本
包含Agent间自由讨论、分布式RAG等功能
"""

import json
import time
from pathlib import Path


def initiate_discussion_about_api_issues():
    """
    发起关于API配置问题的讨论
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return False
    
    try:
        state = json.loads(state_file.read_text())
        
        # 创建讨论消息
        discussion_msg = {
            "id": f"msg_{time.time()}_api_config_discussion",
            "from": "proper_agent_v2",
            "to": "all",
            "type": "discussion",
            "data": {
                "topic": "Tiger API配置问题讨论",
                "question": "大家是否也遇到了API配置问题？我们发现之前的'真实数据'实际上是Mock数据，因为配置文件使用了占位符凭证。",
                "options": [
                    "我也遇到了这个问题",
                    "我已经解决了，使用真实凭证",
                    "我使用其他数据源",
                    "我还在验证API连接"
                ],
                "deadline": time.time() + 7200,  # 2小时后截止
                "context": "之前所有模型训练基于Mock数据，结果不可靠"
            },
            "timestamp": time.time()
        }
        
        # 添加到消息队列
        state["messages"].append(discussion_msg)
        
        # 更新agent状态
        if "proper_agent_v2" in state["agents"]:
            state["agents"]["proper_agent_v2"]["status"] = "initiated_api_discussion"
            state["agents"]["proper_agent_v2"]["last_heartbeat"] = time.time()
        
        # 写回文件
        state_file.write_text(json.dumps(state, indent=2))
        
        print(f"✅ 关于API配置问题的讨论已发起")
        return True
        
    except Exception as e:
        print(f"❌ 发起讨论失败: {str(e)}")
        return False


def share_api_solution_to_rag():
    """
    将API解决方案分享到分布式RAG系统
    """
    import os
    from datetime import datetime
    
    # 创建知识分享内容
    knowledge_title = "Tiger API配置验证和解决方案"
    knowledge_content = """
# Tiger API配置验证和解决方案

## 问题描述
发现之前的"真实数据"实际上是Mock数据，根本原因是配置文件中的凭证都是占位符：
- tiger_id=demoid
- tiger_account=democount
- private_key_path=./demoprivatekey

## 影响范围
1. 之前所有的数据采集：全部使用Mock数据
2. 之前的模型训练：全部基于Mock数据
3. 高准确率问题：Mock数据导致特征简单、模式明显

## 解决方案
1. 获取真实Tiger API凭证（推荐）
2. 检查配置文件是否存在真实凭证
   - 确认tiger_id、tiger_account、private_key_path是真实值而非占位符
3. 实际测试API连接
   - 调用API获取少量数据验证连接
4. 验证获取的数据真实性
   - 检查时间戳合理性
   - 检查价格波动性
   - 检查成交量数据

## API配置验证检查清单
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
    
    # 确保目录存在
    rag_dir = Path("/home/cx/tigertrade/shared_rag/")
    insights_dir = rag_dir / "insights"
    insights_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"proper_agent_v2_tiger_api_solution_{timestamp}.md"
    filepath = insights_dir / filename
    
    # 写入文件
    with open(filepath, 'w') as f:
        f.write(f"# {knowledge_title}\n\n")
        f.write(f"作者: proper_agent_v2\n")
        f.write(f"时间: {datetime.now()}\n\n")
        f.write(knowledge_content)
    
    print(f"✅ API解决方案已分享到RAG: {filepath}")
    
    # 同时发送knowledge_share消息
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return False
    
    try:
        state = json.loads(state_file.read_text())
        
        knowledge_msg = {
            "id": f"msg_{time.time()}_api_solution_knowledge_share",
            "from": "proper_agent_v2",
            "to": "all",
            "type": "knowledge_share",
            "data": {
                "category": "solution",
                "title": knowledge_title,
                "content": "分享Tiger API配置验证和解决方案，避免使用Mock数据替代真实数据的问题",
                "file": str(filepath),
                "evidence": {
                    "confidence": 0.98,
                    "discovered_by": "proper_agent_v2"
                },
                "recommendation": "所有agents都应该验证自己的API配置"
            },
            "timestamp": time.time()
        }
        
        # 添加到消息队列
        state["messages"].append(knowledge_msg)
        
        # 更新agent状态
        if "proper_agent_v2" in state["agents"]:
            state["agents"]["proper_agent_v2"]["status"] = "shared_api_solution"
            state["agents"]["proper_agent_v2"]["last_heartbeat"] = time.time()
        
        # 写回文件
        state_file.write_text(json.dumps(state, indent=2))
        
        return True
        
    except Exception as e:
        print(f"❌ 发送知识分享消息失败: {str(e)}")
        return False


def propose_project_improvement():
    """
    提出项目改进建议
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return False
    
    try:
        state = json.loads(state_file.read_text())
        
        # 获取当前master
        current_master = state.get('current_master', 'master')
        
        improvement_msg = {
            "id": f"msg_{time.time()}_api_validation_improvement",
            "from": "proper_agent_v2",
            "to": current_master,
            "type": "project_suggestion",
            "data": {
                "category": "process",
                "suggestion": "建立API配置验证流程",
                "reasoning": "发现所有之前的数据采集都是Mock数据，因为使用了占位符凭证。需要建立验证流程避免类似问题。",
                "implementation": "在数据采集前增加API连接验证步骤，包含数据真实性检查",
                "impact": "提高数据质量，避免基于Mock数据的错误训练结果",
                "priority": "critical"
            },
            "timestamp": time.time()
        }
        
        # 添加到消息队列
        state["messages"].append(improvement_msg)
        
        # 同时广播给所有agents
        broadcast_msg = {
            "id": f"msg_{time.time()}_api_validation_broadcast",
            "from": "proper_agent_v2",
            "to": "all",
            "type": "project_suggestion",
            "data": {
                "category": "process",
                "suggestion": "建立API配置验证流程",
                "reasoning": "避免再次出现使用Mock数据替代真实数据的问题",
                "call_for_support": "请各位agent投票支持此提议"
            },
            "timestamp": time.time()
        }
        
        # 添加到消息队列
        state["messages"].append(broadcast_msg)
        
        # 更新agent状态
        if "proper_agent_v2" in state["agents"]:
            state["agents"]["proper_agent_v2"]["status"] = "proposed_api_validation_improvement"
            state["agents"]["proper_agent_v2"]["last_heartbeat"] = time.time()
        
        # 写回文件
        state_file.write_text(json.dumps(state, indent=2))
        
        print(f"✅ 项目改进建议已提交：建立API配置验证流程")
        return True
        
    except Exception as e:
        print(f"❌ 提交改进建议失败: {str(e)}")
        return False


def main():
    """主函数"""
    print("🔄 使用新协议v2.1.0功能")
    print("="*70)
    print("发起讨论、分享知识到RAG、提出系统改进建议")
    print("="*70)
    
    # 1. 发起关于API问题的讨论
    print("\n1️⃣ 发起关于API配置问题的讨论...")
    discussion_initiated = initiate_discussion_about_api_issues()
    
    # 2. 分享API解决方案到RAG
    print("\n2️⃣ 分享API解决方案到分布式RAG...")
    solution_shared = share_api_solution_to_rag()
    
    # 3. 提出项目改进建议
    print("\n3️⃣ 提出项目改进建议...")
    improvement_proposed = propose_project_improvement()
    
    print("\n" + "="*70)
    if discussion_initiated and solution_shared and improvement_proposed:
        print("✅ 所有新协议功能已成功使用")
        print("   - 已发起关于API配置问题的讨论")
        print("   - 已将API解决方案分享到分布式RAG")
        print("   - 已提出建立API配置验证流程的建议")
        print("   现在等待其他agents的响应...")
    else:
        print("⚠️  部分功能使用可能有问题")
        print("   但已尽力使用新协议功能")
    print("="*70)


if __name__ == "__main__":
    main()