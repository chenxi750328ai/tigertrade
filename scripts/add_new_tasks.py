#!/usr/bin/env python3
"""
向Master添加新的TigerTrade任务
"""

import sys
sys.path.insert(0, '/home/cx/agentfuture')

from src.coordinator.master_agent import TaskQueue
import json

def add_tigertrade_tasks():
    """添加TigerTrade核心任务"""
    
    print("="*70)
    print("📋 添加TigerTrade新任务")
    print("="*70)
    
    # 创建任务队列实例
    task_queue = TaskQueue()
    
    # 定义新任务
    new_tasks = [
        {
            "type": "transformer_training",
            "description": "🤖 训练Transformer模型预测未来收益",
            "details": {
                "model_type": "Transformer",
                "input_data": "/home/cx/tigertrade/data/processed/train.csv",
                "target": "target_return_5",
                "features": 57,
                "epochs": 100,
                "batch_size": 64,
                "output": "/home/cx/tigertrade/models/transformer_v1.pth"
            },
            "priority": "high",
            "estimated_time": "2小时",
            "dependencies": []
        },
        {
            "type": "strategy_backtest",
            "description": "📊 策略回测：评估盈利能力",
            "details": {
                "model": "/home/cx/tigertrade/models/transformer_v1.pth",
                "test_data": "/home/cx/tigertrade/data/processed/test.csv",
                "metrics": [
                    "总收益率",
                    "月度收益率 (目标: 20%)",
                    "夏普比率 (目标: >2.0)",
                    "最大回撤 (限制: <10%)",
                    "胜率",
                    "盈亏比"
                ],
                "output": "/home/cx/tigertrade/backtest_results/"
            },
            "priority": "high",
            "estimated_time": "1小时",
            "dependencies": ["transformer_training"]
        },
        {
            "type": "risk_management",
            "description": "🛡️ 实现风险管理系统",
            "details": {
                "components": [
                    "止损机制 (固定/移动)",
                    "仓位管理 (Kelly公式/固定比例)",
                    "风险监控 (实时预警)",
                    "最大回撤控制"
                ],
                "risk_limits": {
                    "max_drawdown": "10%",
                    "max_position": "30%",
                    "stop_loss": "2%"
                },
                "output": "/home/cx/tigertrade/src/risk/"
            },
            "priority": "medium",
            "estimated_time": "2小时",
            "dependencies": []
        },
        {
            "type": "feature_discovery",
            "description": "💡 发现自定义特征指标",
            "details": {
                "goal": "找到比传统指标更有效的特征",
                "methods": [
                    "相关性分析",
                    "特征重要性排序",
                    "非线性特征组合",
                    "时间序列模式挖掘"
                ],
                "output_format": "可解释的指标（类似RSI/MACD）",
                "output": "/home/cx/tigertrade/analysis/custom_features.md"
            },
            "priority": "medium",
            "estimated_time": "3小时",
            "dependencies": []
        },
        {
            "type": "model_optimization",
            "description": "🔄 模型超参数优化和集成",
            "details": {
                "tasks": [
                    "网格搜索/贝叶斯优化超参数",
                    "训练LSTM/GRU作为对比",
                    "模型融合（集成学习）",
                    "交叉验证"
                ],
                "output": "/home/cx/tigertrade/models/ensemble/"
            },
            "priority": "low",
            "estimated_time": "3小时",
            "dependencies": ["transformer_training"]
        }
    ]
    
    # 添加任务
    task_queue.add_tasks(new_tasks, created_by="tigertrade_master")
    
    print(f"\n✅ 成功添加 {len(new_tasks)} 个新任务！")
    print("\n📋 任务列表：")
    for i, task in enumerate(new_tasks, 1):
        priority_emoji = "🔴" if task['priority'] == 'high' else "🟡" if task['priority'] == 'medium' else "🟢"
        print(f"  {i}. {priority_emoji} {task['description']}")
        print(f"     优先级: {task['priority']}, 预计: {task['estimated_time']}")
    
    # 显示队列状态
    status = task_queue.get_status()
    print(f"\n📊 任务队列状态：")
    print(f"   待办: {status['pending']} 个")
    print(f"   进行中: {status['in_progress']} 个")
    print(f"   已完成: {status['completed']} 个")
    
    print("\n" + "="*70)
    print("🚀 新任务已就绪！Worker可以开始领取！")
    print("="*70)

if __name__ == "__main__":
    add_tigertrade_tasks()
