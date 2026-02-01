#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时监控DEMO运行状态
"""
import os
import time
import subprocess
from datetime import datetime, timedelta
import re

LOG_FILE = '/tmp/moe_demo.log'
MONITOR_INTERVAL = 10  # 每10秒更新一次

def get_process_info():
    """获取进程信息"""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        lines = result.stdout.split('\n')
        for line in lines:
            if 'run_moe_demo' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) >= 11:
                    return {
                        'pid': parts[1],
                        'cpu': parts[2],
                        'mem': parts[3],
                        'time': ' '.join(parts[9:11])
                    }
    except Exception as e:
        return None
    return None

def get_latest_predictions(log_file, count=5):
    """获取最新的预测结果"""
    predictions = []
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if '预测:' in line or '动作:' in line or '置信度:' in line or '预测收益率:' in line:
                    predictions.append(line.strip())
                    if len(predictions) >= count * 4:  # 每个预测有4行
                        break
    except Exception as e:
        return []
    
    # 提取完整的预测信息
    result = []
    current_pred = {}
    for line in reversed(predictions):
        if '预测:' in line:
            # 提取时间戳
            time_match = re.search(r'\[(\d{2}:\d{2}:\d{2})\]', line)
            if time_match:
                current_pred['time'] = time_match.group(1)
        elif '动作:' in line:
            action_match = re.search(r'动作:\s*(\S+)', line)
            if action_match:
                current_pred['action'] = action_match.group(1)
        elif '置信度:' in line:
            conf_match = re.search(r'置信度:\s*([\d.]+)', line)
            if conf_match:
                current_pred['confidence'] = conf_match.group(1)
        elif '预测收益率:' in line:
            profit_match = re.search(r'预测收益率:\s*([\d.]+)%', line)
            if profit_match:
                current_pred['profit'] = profit_match.group(1)
                if current_pred:
                    result.append(current_pred.copy())
                    current_pred = {}
    
    return list(reversed(result[-count:]))

def get_statistics(log_file):
    """获取统计信息"""
    stats = {
        'total_predictions': 0,
        'buy_signals': 0,
        'sell_signals': 0,
        'hold_signals': 0,
        'avg_confidence': 0.0,
        'errors': 0
    }
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
            # 统计预测次数
            stats['total_predictions'] = content.count('预测:')
            
            # 统计动作
            stats['buy_signals'] = content.count('动作: 买入')
            stats['sell_signals'] = content.count('动作: 卖出')
            stats['hold_signals'] = content.count('动作: 不操作')
            
            # 统计错误
            stats['errors'] = content.count('❌')
            
            # 提取所有置信度
            confidences = re.findall(r'置信度:\s*([\d.]+)', content)
            if confidences:
                confidences = [float(c) for c in confidences]
                stats['avg_confidence'] = sum(confidences) / len(confidences) if confidences else 0.0
            
    except Exception as e:
        pass
    
    return stats

def get_api_status(log_file):
    """获取API连接状态"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            if 'Mock模式: False' in content:
                return '✅ 真实API'
            elif 'Mock模式: True' in content:
                return '⚠️ Mock模式'
            else:
                return '❓ 未知'
    except:
        return '❓ 未知'

def get_runtime_info(log_file):
    """获取运行时间信息"""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if '开始时间:' in line:
                    time_match = re.search(r'开始时间:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
                    if time_match:
                        start_time_str = time_match.group(1)
                        start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
                        elapsed = datetime.now() - start_time
                        return {
                            'start_time': start_time_str,
                            'elapsed': elapsed,
                            'elapsed_str': str(elapsed).split('.')[0]
                        }
                elif '结束时间:' in line:
                    time_match = re.search(r'结束时间:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
                    if time_match:
                        end_time_str = time_match.group(1)
                        end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')
                        remaining = end_time - datetime.now()
                        return {
                            'end_time': end_time_str,
                            'remaining': remaining,
                            'remaining_str': str(remaining).split('.')[0] if remaining.total_seconds() > 0 else '已完成'
                        }
    except Exception as e:
        pass
    return {}

def get_strategy_info(log_file):
    """获取策略信息"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            strategy_match = re.search(r'使用策略:\s*(\S+)', content)
            if strategy_match:
                return strategy_match.group(1)
    except:
        pass
    return '未知'

def main():
    """主监控循环"""
    print("="*70)
    print("📊 DEMO运行状态监控")
    print("="*70)
    print(f"日志文件: {LOG_FILE}")
    print(f"更新间隔: {MONITOR_INTERVAL}秒")
    print("按 Ctrl+C 退出监控")
    print("="*70)
    
    while True:
        try:
            # 清屏
            os.system('clear' if os.name != 'nt' else 'cls')
            
            print("="*70)
            print(f"📊 DEMO运行状态监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*70)
            
            # 1. 进程状态
            print("\n【1】进程状态")
            print("-"*70)
            proc_info = get_process_info()
            if proc_info:
                print(f"  PID: {proc_info['pid']}")
                print(f"  CPU: {proc_info['cpu']}%")
                print(f"  内存: {proc_info['mem']}%")
                print(f"  运行时间: {proc_info['time']}")
            else:
                print("  ⚠️ 进程未运行")
            
            # 2. API连接状态
            print("\n【2】API连接状态")
            print("-"*70)
            api_status = get_api_status(LOG_FILE)
            print(f"  {api_status}")
            
            # 3. 策略信息
            print("\n【3】策略信息")
            print("-"*70)
            strategy_name = get_strategy_info(LOG_FILE)
            print(f"  当前策略: {strategy_name}")
            
            # 4. 运行时间
            print("\n【4】运行时间")
            print("-"*70)
            runtime_info = get_runtime_info(LOG_FILE)
            if runtime_info:
                if 'start_time' in runtime_info:
                    print(f"  开始时间: {runtime_info['start_time']}")
                    print(f"  已运行: {runtime_info['elapsed_str']}")
                if 'end_time' in runtime_info:
                    print(f"  结束时间: {runtime_info['end_time']}")
                    print(f"  剩余时间: {runtime_info['remaining_str']}")
            
            # 5. 统计信息
            print("\n【5】统计信息")
            print("-"*70)
            stats = get_statistics(LOG_FILE)
            print(f"  总预测次数: {stats['total_predictions']}")
            print(f"  买入信号: {stats['buy_signals']}")
            print(f"  卖出信号: {stats['sell_signals']}")
            print(f"  持有信号: {stats['hold_signals']}")
            if stats['total_predictions'] > 0:
                print(f"  平均置信度: {stats['avg_confidence']:.3f}")
            print(f"  错误次数: {stats['errors']}")
            
            # 6. 最新预测结果
            print("\n【6】最新预测结果（最近5次）")
            print("-"*70)
            predictions = get_latest_predictions(LOG_FILE, count=5)
            if predictions:
                for i, pred in enumerate(predictions, 1):
                    action_map = {'买入': '🟢', '卖出': '🔴', '不操作': '⚪'}
                    action_icon = action_map.get(pred.get('action', ''), '❓')
                    print(f"  [{i}] {pred.get('time', 'N/A')} {action_icon} {pred.get('action', 'N/A')}")
                    print(f"      置信度: {pred.get('confidence', 'N/A')}")
                    if 'profit' in pred:
                        print(f"      预测收益率: {pred['profit']}%")
            else:
                print("  ⏳ 暂无预测结果")
            
            # 7. 最新日志（最后3行）
            print("\n【7】最新日志")
            print("-"*70)
            try:
                with open(LOG_FILE, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-3:]:
                        print(f"  {line.rstrip()}")
            except Exception as e:
                print(f"  ⚠️ 无法读取日志: {e}")
            
            print("\n" + "="*70)
            print(f"下次更新: {MONITOR_INTERVAL}秒后... (按 Ctrl+C 退出)")
            
            time.sleep(MONITOR_INTERVAL)
        
        except KeyboardInterrupt:
            print("\n\n🛑 监控已停止")
            break
        except Exception as e:
            print(f"\n❌ 监控错误: {e}")
            time.sleep(MONITOR_INTERVAL)

if __name__ == '__main__':
    main()
