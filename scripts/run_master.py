#!/usr/bin/env python3
"""Master Agent启动脚本"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.coordinator.master_agent import MasterAgent

def main():
    """运行Master Agent"""
    
    # 创建Master
    master = MasterAgent()
    
    # 注册示例项目
    master.register_project("TigerTrade数据处理", [
        {
            "type": "data_download",
            "description": "下载SIL2603历史数据",
            "symbol": "SIL2603",
            "days": 90
        },
        {
            "type": "data_clean",
            "description": "清洗和标准化数据",
            "file": "raw_data.csv"
        },
        {
            "type": "model_train",
            "description": "训练Transformer模型",
            "model": "transformer",
            "epochs": 50
        },
        {
            "type": "backtest",
            "description": "回测交易策略",
            "strategy": "grid",
            "period": "2025-10-01 to 2026-01-21"
        }
    ])
    
    # 运行Master（60秒）
    print("\n⏰ Master将运行60秒，等待Worker连接...")
    print("💡 请在另一个终端或AI工具启动Worker\n")
    
    master.run(duration=120)  # 运行2分钟

if __name__ == '__main__':
    main()
