#!/usr/bin/env python3
"""
真正的流水线并发演示

对比：
1. 串行模式（依赖链，资源浪费）
2. 流水线模式（队列解耦，高效并发）
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import threading
from queue import Queue, Empty
from src.coordinator import AgentCoordinator


# ==================== 模式1: 串行（依赖链）====================

def demo_serial():
    """串行模式：Agent互相等待"""
    print("\n" + "="*80)
    print("模式1: 串行（依赖链）- 资源浪费")
    print("="*80)
    
    NUM_BATCHES = 10
    
    def agent1():
        coord = AgentCoordinator("serial_agent1")
        for i in range(NUM_BATCHES):
            coord.update_status("working", f"批次{i}", i/NUM_BATCHES)
            print(f"[Serial Agent1] 处理批次 {i}")
            time.sleep(1)  # 模拟处理
            coord.send_message("serial_agent2", "batch_ready", {"batch_id": i})
        coord.cleanup()
    
    def agent2():
        coord = AgentCoordinator("serial_agent2")
        for i in range(NUM_BATCHES):
            # 等待消息
            msg = coord.wait_for_message("batch_ready", timeout=30)
            coord.update_status("working", f"批次{i}", i/NUM_BATCHES)
            print(f"[Serial Agent2] 处理批次 {i} (等待后)")
            time.sleep(2)  # 模拟处理（瓶颈）
            coord.send_message("serial_agent3", "batch_ready", {"batch_id": i})
        coord.cleanup()
    
    def agent3():
        coord = AgentCoordinator("serial_agent3")
        for i in range(NUM_BATCHES):
            msg = coord.wait_for_message("batch_ready", timeout=30)
            coord.update_status("working", f"批次{i}", i/NUM_BATCHES)
            print(f"[Serial Agent3] 处理批次 {i} (等待后)")
            time.sleep(1)  # 模拟处理
        coord.cleanup()
    
    start = time.time()
    
    t1 = threading.Thread(target=agent1)
    t2 = threading.Thread(target=agent2)
    t3 = threading.Thread(target=agent3)
    
    t1.start(); t2.start(); t3.start()
    t1.join(); t2.join(); t3.join()
    
    elapsed = time.time() - start
    
    print(f"\n串行总耗时: {elapsed:.1f}秒")
    print(f"理论耗时: {NUM_BATCHES * (1+2+1)} = {NUM_BATCHES*4}秒")
    print(f"平均CPU利用率: ~33% (只有1个Agent在工作)")
    
    return elapsed


# ==================== 模式2: 流水线并发 ====================

def demo_pipeline():
    """流水线模式：队列解耦，并发执行"""
    print("\n" + "="*80)
    print("模式2: 流水线并发 - 高效利用资源")
    print("="*80)
    
    NUM_BATCHES = 10
    
    # 创建队列
    queue_12 = Queue(maxsize=5)
    queue_23 = Queue(maxsize=5)
    
    def agent1():
        coord = AgentCoordinator("pipeline_agent1")
        for i in range(NUM_BATCHES):
            coord.update_status("working", f"批次{i}", i/NUM_BATCHES)
            print(f"[Pipeline Agent1] 处理批次 {i}")
            time.sleep(1)
            
            # 放入队列（不等待Agent2完成）
            queue_12.put({"batch_id": i, "data": f"data_{i}"})
        
        # 发送结束信号
        queue_12.put(None)
        coord.cleanup()
    
    def agent2():
        coord = AgentCoordinator("pipeline_agent2")
        batch_count = 0
        
        while True:
            # 从队列获取（可能需要等待Agent1）
            batch = queue_12.get()
            
            if batch is None:  # 结束信号
                queue_23.put(None)
                break
            
            coord.update_status("working", f"批次{batch_count}", batch_count/NUM_BATCHES)
            print(f"[Pipeline Agent2] 处理批次 {batch['batch_id']} (瓶颈)")
            time.sleep(2)  # 瓶颈
            
            # 放入下一个队列
            queue_23.put({"batch_id": batch['batch_id'], "result": "processed"})
            batch_count += 1
        
        coord.cleanup()
    
    def agent3():
        coord = AgentCoordinator("pipeline_agent3")
        batch_count = 0
        
        while True:
            batch = queue_23.get()
            
            if batch is None:
                break
            
            coord.update_status("working", f"批次{batch_count}", batch_count/NUM_BATCHES)
            print(f"[Pipeline Agent3] 处理批次 {batch['batch_id']}")
            time.sleep(1)
            batch_count += 1
        
        coord.cleanup()
    
    start = time.time()
    
    t1 = threading.Thread(target=agent1)
    t2 = threading.Thread(target=agent2)
    t3 = threading.Thread(target=agent3)
    
    t1.start(); t2.start(); t3.start()
    t1.join(); t2.join(); t3.join()
    
    elapsed = time.time() - start
    
    print(f"\n流水线总耗时: {elapsed:.1f}秒")
    print(f"理论耗时: 启动(4s) + 稳态({NUM_BATCHES}*2s) = {4 + NUM_BATCHES*2}秒")
    print(f"平均CPU利用率: ~100% (3个Agent同时工作)")
    
    return elapsed


# ==================== 模式3: 完全并行 ====================

def demo_full_parallel():
    """完全并行：多个独立任务同时执行"""
    print("\n" + "="*80)
    print("模式3: 完全并行 - 独立任务")
    print("="*80)
    
    results_queue = Queue()
    
    def train_model(model_name, duration):
        coord = AgentCoordinator(f"model_{model_name}")
        
        print(f"[{model_name}] 开始训练...")
        coord.update_status("working", f"训练{model_name}", 0.0)
        
        for i in range(5):
            time.sleep(duration / 5)
            coord.update_status("working", f"训练{model_name}", (i+1)/5)
            print(f"[{model_name}] 进度 {(i+1)*20}%")
        
        print(f"[{model_name}] 训练完成")
        results_queue.put({
            "model": model_name,
            "accuracy": 0.8 + (hash(model_name) % 10) / 100
        })
        
        coord.cleanup()
    
    start = time.time()
    
    # 4个模型同时训练
    models = [
        ("Transformer", 3),
        ("LSTM", 2.5),
        ("RandomForest", 2),
        ("XGBoost", 2.8)
    ]
    
    threads = []
    for model_name, duration in models:
        t = threading.Thread(target=train_model, args=(model_name, duration))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    elapsed = time.time() - start
    
    # 收集结果
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())
    
    print(f"\n完全并行总耗时: {elapsed:.1f}秒")
    print(f"对比串行: {sum(d for _, d in models):.1f}秒")
    print(f"加速比: {sum(d for _, d in models) / elapsed:.1f}x")
    print(f"\n训练结果:")
    for r in results:
        print(f"  {r['model']}: 准确率 {r['accuracy']:.2%}")
    
    return elapsed


# ==================== 实时监控 ====================

def monitor_pipeline(duration=30):
    """监控流水线状态"""
    coord = AgentCoordinator("monitor")
    
    print(f"\n{'─'*60}")
    print("实时监控 (30秒)")
    print(f"{'─'*60}\n")
    
    start = time.time()
    last_print = 0
    
    while time.time() - start < duration:
        current = time.time() - start
        
        # 每2秒打印一次
        if current - last_print >= 2:
            status = coord.get_all_agents_status()
            
            print(f"⏰ {current:.0f}s")
            for agent_id in sorted(status.keys()):
                if not agent_id.startswith("monitor"):
                    s = status[agent_id]
                    print(f"  [{agent_id:20}] {s['status']:8} | {s.get('task', 'N/A'):15} | {s.get('progress', 0)*100:5.1f}%")
            print()
            
            last_print = current
        
        time.sleep(0.5)
    
    coord.cleanup()


# ==================== 主函数 ====================

def main():
    print("\n" + "="*80)
    print("🚀 流水线并发 vs 串行 - 性能对比")
    print("="*80)
    
    print("\n场景：处理10批数据")
    print("  Agent1: 1秒/批")
    print("  Agent2: 2秒/批 (瓶颈)")
    print("  Agent3: 1秒/批")
    
    # 运行对比
    time_serial = demo_serial()
    time.sleep(2)  # 间隔
    
    time_pipeline = demo_pipeline()
    time.sleep(2)
    
    time_parallel = demo_full_parallel()
    
    # 总结
    print("\n" + "="*80)
    print("📊 性能对比总结")
    print("="*80)
    
    print(f"\n串行模式:")
    print(f"  耗时: {time_serial:.1f}秒")
    print(f"  特点: Agent互相等待，资源浪费")
    print(f"  适用: 单次完整流程")
    
    print(f"\n流水线模式:")
    print(f"  耗时: {time_pipeline:.1f}秒")
    print(f"  加速: {time_serial / time_pipeline:.1f}x")
    print(f"  特点: 队列解耦，持续并发")
    print(f"  适用: 持续数据流 ✅")
    
    print(f"\n完全并行模式:")
    print(f"  耗时: {time_parallel:.1f}秒")
    print(f"  加速: {sum([3, 2.5, 2, 2.8]) / time_parallel:.1f}x")
    print(f"  特点: 独立任务，完全并行")
    print(f"  适用: 多模型训练 ✅✅✅")
    
    print("\n" + "="*80)
    print("💡 关键洞察")
    print("="*80)
    print("\n1. 串行 = 资源浪费")
    print("   → 每次只有1个Agent工作")
    
    print("\n2. 流水线 = 高效并发")
    print("   → 所有Agent同时工作")
    print("   → 吞吐量由瓶颈决定（Agent2: 2秒/批）")
    
    print("\n3. 完全并行 = 最大加速")
    print("   → 无依赖的任务直接并行")
    print("   → 加速比 ≈ Agent数量")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
