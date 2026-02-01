#!/usr/bin/env python3
"""
定时显示AGENT任务状态脚本
用于监控多AGENT系统中的各个AGENT状态
"""

import json
import time
from pathlib import Path
import argparse


def load_agent_state():
    """加载AGENT状态"""
    state_file = Path("/tmp/tigertrade_agent_state.json")
    if state_file.exists():
        return json.loads(state_file.read_text())
    else:
        print("❌ 状态文件不存在，请确保RAG系统和多AGENT系统已启动")
        return None


def display_agent_status(refresh_interval=5, count=10):
    """
    显示AGENT状态
    
    Args:
        refresh_interval: 刷新间隔（秒）
        count: 显示次数
    """
    print(f"🔄 开始监控AGENT状态，刷新间隔: {refresh_interval}s，总次数: {count}")
    print("="*80)
    
    for i in range(count):
        state = load_agent_state()
        if state is None:
            time.sleep(refresh_interval)
            continue
            
        print(f"\n📊 AGENT状态快照 #{i+1} - {time.strftime('%H:%M:%S')}")
        print("-" * 80)
        
        agents = state.get("agents", {})
        if not agents:
            print("   暂无AGENT注册")
        else:
            print(f"   {'AGENT ID':<20} {'状态':<12} {'任务':<20} {'进度':<8} {'最后心跳'}")
            print("   " + "-" * 75)
            
            for agent_id, info in agents.items():
                status = info.get("status", "unknown")
                task = (info.get("task") or "N/A")[:20]  # 限制长度，处理None情况
                progress = f"{info.get('progress', 0)*100:.1f}%"
                
                # 转换时间戳为人可读格式
                last_heartbeat = info.get("last_heartbeat", 0)
                if last_heartbeat:
                    heartbeat_str = time.strftime('%H:%M:%S', time.localtime(last_heartbeat))
                else:
                    heartbeat_str = "N/A"
                    
                print(f"   {agent_id:<20} {status:<12} {task:<20} {progress:<8} {heartbeat_str}")
        
        # 显示消息统计
        messages = state.get("messages", [])
        print(f"\n   📨 消息总数: {len(messages)}")
        
        # 显示最近的消息
        if messages:
            print(f"   📝 最近3条消息:")
            for msg in messages[-3:]:  # 显示最后3条消息
                print(f"     • {msg['from']} → {msg['to']} ({msg['type']})")
        
        # 显示选举状态
        election_status = state.get("election_status", {})
        if election_status:
            print(f"\n   🗳️  选举状态:")
            print(f"     • 当前MASTER: {election_status.get('current_master', 'N/A')}")
            candidates = election_status.get('candidates', [])
            print(f"     • 候选人: {', '.join(candidates) if candidates else '无'}")
        
        print("="*80)
        
        if i < count - 1:  # 不在最后一次循环后等待
            time.sleep(refresh_interval)


def main():
    parser = argparse.ArgumentParser(description='定时显示AGENT任务状态')
    parser.add_argument('-i', '--interval', type=int, default=5, 
                        help='刷新间隔（秒），默认为5秒')
    parser.add_argument('-c', '--count', type=int, default=10, 
                        help='显示次数，默认为10次')
    parser.add_argument('--continuous', action='store_true',
                        help='持续显示（除非手动中断）')
    
    args = parser.parse_args()
    
    if args.continuous:
        print("🔄 持续监控AGENT状态，按 Ctrl+C 停止...")
        print("="*80)
        
        i = 0
        try:
            while True:
                state = load_agent_state()
                if state is None:
                    time.sleep(args.interval)
                    continue
                    
                print(f"\n📊 AGENT状态快照 #{i+1} - {time.strftime('%H:%M:%S')}")
                print("-" * 80)
                
                agents = state.get("agents", {})
                if not agents:
                    print("   暂无AGENT注册")
                else:
                    print(f"   {'AGENT ID':<20} {'状态':<12} {'任务':<20} {'进度':<8} {'最后心跳'}")
                    print("   " + "-" * 75)
                    
                    for agent_id, info in agents.items():
                        status = info.get("status", "unknown")
                        task = (info.get("task") or "N/A")[:20]  # 限制长度，处理None情况
                        progress = f"{info.get('progress', 0)*100:.1f}%"
                        
                        # 转换时间戳为人可读格式
                        last_heartbeat = info.get("last_heartbeat", 0)
                        if last_heartbeat:
                            heartbeat_str = time.strftime('%H:%M:%S', time.localtime(last_heartbeat))
                        else:
                            heartbeat_str = "N/A"
                            
                        print(f"   {agent_id:<20} {status:<12} {task:<20} {progress:<8} {heartbeat_str}")
                
                # 显示消息统计
                messages = state.get("messages", [])
                print(f"\n   📨 消息总数: {len(messages)}")
                
                # 显示最近的消息
                if messages:
                    print(f"   📝 最近3条消息:")
                    for msg in messages[-3:]:  # 显示最后3条消息
                        print(f"     • {msg['from']} → {msg['to']} ({msg['type']})")
                
                # 显示选举状态
                election_status = state.get("election_status", {})
                if election_status:
                    print(f"\n   🗳️  选举状态:")
                    print(f"     • 当前MASTER: {election_status.get('current_master', 'N/A')}")
                    candidates = election_status.get('candidates', [])
                    print(f"     • 候选人: {', '.join(candidates) if candidates else '无'}")
                
                print("="*80)
                
                i += 1
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n🛑 已停止监控")
    else:
        display_agent_status(args.interval, args.count)


if __name__ == "__main__":
    main()