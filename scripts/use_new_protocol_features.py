#!/usr/bin/env python3
"""
使用新协议v2.1.0功能的脚本
包含Agent间自由讨论、分布式RAG等功能
"""

import json
import time
from pathlib import Path


def initiate_discussion():
    """
    发起一个讨论，使用新协议的讨论功能
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return False
    
    try:
        state = json.loads(state_file.read_text())
        
        # 创建讨论消息
        discussion_msg = {
            "id": f"msg_{time.time()}_protocol_discussion",
            "from": "proper_agent_v2",
            "to": "all",
            "type": "discussion",
            "data": {
                "topic": "API配置问题解决方案",
                "question": "大家都遇到了API配置问题，使用占位符凭证导致获取Mock数据，你们是如何验证API连接有效性的？",
                "options": [
                    "直接调用API获取少量数据验证",
                    "使用ping/health检查",
                    "检查返回数据特征（时间戳、波动性等）",
                    "其他方法"
                ],
                "deadline": time.time() + 3600,  # 1小时后截止
                "context": "发现所有之前的数据采集都是Mock数据的问题"
            },
            "timestamp": time.time()
        }
        
        # 添加到消息队列
        state["messages"].append(discussion_msg)
        
        # 更新agent状态
        if "proper_agent_v2" in state["agents"]:
            state["agents"]["proper_agent_v2"]["status"] = "initiated_discussion"
            state["agents"]["proper_agent_v2"]["last_heartbeat"] = time.time()
        
        # 写回文件
        state_file.write_text(json.dumps(state, indent=2))
        
        print(f"✅ 讨论已发起：API配置问题解决方案")
        return True
        
    except Exception as e:
        print(f"❌ 发起讨论失败: {str(e)}")
        return False


def share_knowledge_to_rag():
    """
    分享知识到分布式RAG系统
    """
    import os
    from datetime import datetime
    
    # 创建知识分享内容
    knowledge_title = "API配置验证最佳实践"
    knowledge_content = """
# API配置验证最佳实践

## 问题
之前所有数据采集使用Mock数据，因为API配置使用占位符凭证。

## 解决方案
1. 检查配置文件是否存在真实凭证
   - 确认tiger_id、tiger_account、private_key_path是真实值而非占位符

2. 检查关键字段是否包含占位符
   - 避免使用demo、placeholder、fake等关键词

3. 实际测试API连接
   - 调用API获取少量数据验证连接

4. 验证获取的数据真实性
   - 检查时间戳合理性
   - 检查价格波动性
   - 检查成交量数据
    """
    
    # 确保目录存在
    rag_dir = Path("/home/cx/tigertrade/shared_rag/")
    insights_dir = rag_dir / "insights"
    insights_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"proper_agent_v2_api_verification_best_practices_{timestamp}.md"
    filepath = insights_dir / filename
    
    # 写入文件
    with open(filepath, 'w') as f:
        f.write(f"# {knowledge_title}\n\n")
        f.write(f"作者: proper_agent_v2\n")
        f.write(f"时间: {datetime.now()}\n\n")
        f.write(knowledge_content)
    
    print(f"✅ 知识已分享到RAG: {filepath}")
    
    # 同时发送knowledge_share消息
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return False
    
    try:
        state = json.loads(state_file.read_text())
        
        knowledge_msg = {
            "id": f"msg_{time.time()}_knowledge_share",
            "from": "proper_agent_v2",
            "to": "all",
            "type": "knowledge_share",
            "data": {
                "category": "insight",
                "title": knowledge_title,
                "content": knowledge_content[:200] + "...",  # 只显示部分内容
                "file": str(filepath),
                "evidence": {
                    "confidence": 0.95,
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
            state["agents"]["proper_agent_v2"]["status"] = "shared_knowledge"
            state["agents"]["proper_agent_v2"]["last_heartbeat"] = time.time()
        
        # 写回文件
        state_file.write_text(json.dumps(state, indent=2))
        
        return True
        
    except Exception as e:
        print(f"❌ 发送知识分享消息失败: {str(e)}")
        return False


def propose_system_improvement():
    """
    提出系统改进建议
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
            "id": f"msg_{time.time()}_system_improvement_proposal",
            "from": "proper_agent_v2",
            "to": current_master,
            "type": "project_suggestion",
            "data": {
                "category": "process",
                "suggestion": "建立API配置验证流程",
                "reasoning": "发现所有之前的数据采集都是Mock数据，因为使用了占位符凭证。需要建立验证流程避免类似问题。",
                "implementation": "在数据采集前增加API连接验证步骤",
                "impact": "提高数据质量，避免基于Mock数据的错误训练结果",
                "priority": "critical"
            },
            "timestamp": time.time()
        }
        
        # 添加到消息队列
        state["messages"].append(improvement_msg)
        
        # 同时广播给所有agents
        broadcast_msg = {
            "id": f"msg_{time.time()}_system_improvement_broadcast",
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
            state["agents"]["proper_agent_v2"]["status"] = "proposed_improvement"
            state["agents"]["proper_agent_v2"]["last_heartbeat"] = time.time()
        
        # 写回文件
        state_file.write_text(json.dumps(state, indent=2))
        
        print(f"✅ 系统改进建议已提交：建立API配置验证流程")
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
    
    # 1. 发起讨论
    print("\n1️⃣ 发起关于API配置问题的讨论...")
    discussion_initiated = initiate_discussion()
    
    # 2. 分享知识到RAG
    print("\n2️⃣ 分享API配置验证最佳实践到RAG...")
    knowledge_shared = share_knowledge_to_rag()
    
    # 3. 提出系统改进建议
    print("\n3️⃣ 提出系统改进建议...")
    improvement_proposed = propose_system_improvement()
    
    print("\n" + "="*70)
    if discussion_initiated and knowledge_shared and improvement_proposed:
        print("✅ 所有新协议功能已成功使用")
        print("   - 已发起关于API配置问题的讨论")
        print("   - 已将最佳实践分享到分布式RAG")
        print("   - 已提出系统改进建议")
        print("   现在等待其他agents的响应...")
    else:
        print("⚠️  部分功能使用可能有问题")
        print("   但已尽力使用新协议功能")
    print("="*70)


if __name__ == "__main__":
    main()