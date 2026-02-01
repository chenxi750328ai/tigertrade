#!/usr/bin/env python3
"""
TigerTrade Master Agent - 协调多Agent实现盈利目标
目标：月盈利率 20%
"""

import sys
sys.path.insert(0, '/home/cx/agentfuture')
sys.path.insert(0, '/home/cx/tigertrade')

from src.coordinator.master_agent import MasterAgent
from src.coordinator.coordinator import AgentCoordinator
import time
import json

def main():
    print("=" * 70)
    print("🚀 TigerTrade Master Agent 启动")
    print("=" * 70)
    print("目标：💰 月盈利率 20%")
    print("策略：多Agent并行协作")
    print("=" * 70)
    
    # 创建Master
    master = MasterAgent("tigertrade_master")
    
    # 定义TigerTrade任务
    tasks = [
        {
            "type": "data_preprocessing",
            "description": "数据预处理和特征工程",
            "details": {
                "input": "/home/cx/tigertrade/data/tick_data.csv",
                "output": "/home/cx/tigertrade/data/processed/",
                "steps": [
                    "1. 数据清洗（缺失值、异常值）",
                    "2. 特征工程（技术指标：RSI, MACD, Bollinger）",
                    "3. 时间窗口特征（5/10/30/60分钟）",
                    "4. 数据增强（时间扰动、噪声注入）",
                    "5. 训练/验证/测试集分割（时间序列）"
                ]
            },
            "priority": "high",
            "estimated_time": "2小时",
            "dependencies": []
        },
        {
            "type": "model_training",
            "description": "Transformer模型训练和优化",
            "details": {
                "input": "/home/cx/tigertrade/data/processed/",
                "output": "/home/cx/tigertrade/models/",
                "models": [
                    "Transformer (基线)",
                    "LSTM (对比)",
                    "GRU (对比)",
                    "Transformer + Attention",
                    "Ensemble (集成)"
                ],
                "hyperparameters": {
                    "learning_rate": [0.0001, 0.0005, 0.001],
                    "batch_size": [32, 64, 128],
                    "hidden_dim": [128, 256, 512],
                    "num_layers": [2, 4, 6]
                }
            },
            "priority": "high",
            "estimated_time": "4小时",
            "dependencies": ["data_preprocessing"]
        },
        {
            "type": "strategy_backtest",
            "description": "策略回测和盈利评估",
            "details": {
                "input": "/home/cx/tigertrade/models/",
                "output": "/home/cx/tigertrade/backtest_results/",
                "metrics": [
                    "总收益率",
                    "夏普比率",
                    "最大回撤",
                    "胜率",
                    "盈亏比",
                    "月度收益率（目标：20%）"
                ],
                "strategies": [
                    "趋势跟踪",
                    "均值回归",
                    "动量突破",
                    "套利策略"
                ]
            },
            "priority": "high",
            "estimated_time": "2小时",
            "dependencies": ["model_training"]
        },
        {
            "type": "risk_management",
            "description": "风险管理系统实现",
            "details": {
                "components": [
                    "止损机制（固定止损/移动止损）",
                    "仓位管理（Kelly公式/固定比例）",
                    "风险监控（实时预警）",
                    "资金管理（最大回撤限制）"
                ],
                "risk_limits": {
                    "max_drawdown": "10%",
                    "max_position": "30%",
                    "stop_loss": "2%"
                }
            },
            "priority": "medium",
            "estimated_time": "3小时",
            "dependencies": ["strategy_backtest"]
        },
        {
            "type": "feature_discovery",
            "description": "自定义特征指标发现",
            "details": {
                "goal": "发现比传统指标更有效的特征",
                "methods": [
                    "价格-成交量关系分析",
                    "时间周期模式识别",
                    "波动率特征提取",
                    "微观结构特征"
                ],
                "output": "可解释的自定义指标（类似RSI/ROC）"
            },
            "priority": "medium",
            "estimated_time": "3小时",
            "dependencies": ["data_preprocessing"]
        }
    ]
    
    print(f"\n📋 注册TigerTrade项目并创建 {len(tasks)} 个任务...")
    master.register_project("TigerTrade盈利计划", tasks)
    
    print(f"\n✅ 任务创建完成！")
    print(f"\n📊 任务概览：")
    for i, task in enumerate(tasks, 1):
        print(f"  {i}. {task['description']}")
        print(f"     优先级: {task['priority']}, 预计时间: {task['estimated_time']}")
    
    print(f"\n" + "=" * 70)
    print("🎯 Master Agent运行中...")
    print("=" * 70)
    print("其他AI可以：")
    print("  1. 领取任务：python -c 'from src.coordinator import AgentCoordinator; ...'")
    print("  2. 参考文档：/home/cx/NOTIFY_OTHER_AI.txt")
    print("  3. 提议任务：使用 propose_task()")
    print("=" * 70)
    
    # 运行Master
    duration = 7200  # 2小时
    print(f"\n⏱️  Master将运行 {duration//60} 分钟")
    print(f"📁 状态文件：/tmp/tigertrade_agent_state.json")
    print(f"📋 任务队列：/tmp/tigertrade_task_queue.json")
    print(f"\n按 Ctrl+C 停止\n")
    
    try:
        master.run(duration=duration)
    except KeyboardInterrupt:
        print("\n\n⚠️  Master Agent 已停止")
        print("=" * 70)

if __name__ == "__main__":
    main()
