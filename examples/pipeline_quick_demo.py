#!/usr/bin/env python3
"""流水线并发快速演示"""
import time, threading
from queue import Queue

def demo_serial():
    print("\n" + "="*70)
    print("🐌 串行模式")
    print("="*70)
    results = []
    
    def agent1():
        for i in range(3):
            print(f"  {time.time()-start:.1f}s | [Agent1] 批次{i}")
            time.sleep(1)
            results.append(("1", i))
    
    def agent2():
        while len([r for r in results if r[0]=="1"]) < 3: time.sleep(0.1)
        for i in range(3):
            print(f"  {time.time()-start:.1f}s | [Agent2] 批次{i} (等待后)")
            time.sleep(2)
            results.append(("2", i))
    
    def agent3():
        while len([r for r in results if r[0]=="2"]) < 3: time.sleep(0.1)
        for i in range(3):
            print(f"  {time.time()-start:.1f}s | [Agent3] 批次{i} (等待后)")
            time.sleep(1)
    
    start = time.time()
    t1 = threading.Thread(target=agent1)
    t2 = threading.Thread(target=agent2)
    t3 = threading.Thread(target=agent3)
    t1.start(); t2.start(); t3.start()
    t1.join(); t2.join(); t3.join()
    elapsed = time.time() - start
    print(f"\n  耗时: {elapsed:.1f}秒 | CPU: ~33% | ❌ 资源浪费")
    return elapsed

def demo_pipeline():
    print("\n" + "="*70)
    print("🚀 流水线模式")
    print("="*70)
    q1, q2 = Queue(), Queue()
    
    def agent1():
        for i in range(3):
            print(f"  {time.time()-start:.1f}s | [Agent1] 批次{i}")
            time.sleep(1)
            q1.put(i)
        q1.put(None)
    
    def agent2():
        while True:
            b = q1.get()
            if b is None: q2.put(None); break
            print(f"  {time.time()-start:.1f}s | [Agent2] 批次{b} (队列)")
            time.sleep(2)
            q2.put(b)
    
    def agent3():
        while True:
            b = q2.get()
            if b is None: break
            print(f"  {time.time()-start:.1f}s | [Agent3] 批次{b} (队列)")
            time.sleep(1)
    
    start = time.time()
    t1 = threading.Thread(target=agent1)
    t2 = threading.Thread(target=agent2)
    t3 = threading.Thread(target=agent3)
    t1.start(); t2.start(); t3.start()
    t1.join(); t2.join(); t3.join()
    elapsed = time.time() - start
    print(f"\n  耗时: {elapsed:.1f}秒 | CPU: ~100% | ✅ 高效并发")
    return elapsed

print("\n" + "="*70)
print("📊 流水线并发 vs 串行对比")
print("="*70)
print("\n场景: 3批数据, Agent1(1s) → Agent2(2s) → Agent3(1s)\n")

t1 = demo_serial()
time.sleep(1)
t2 = demo_pipeline()

print("\n" + "="*70)
print("📊 对比结果")
print("="*70)
print(f"\n串行: {t1:.1f}秒 (理论12秒) ❌")
print(f"流水线: {t2:.1f}秒 (理论8秒) ✅")
print(f"加速: {t1/t2:.1f}x")
print("\n💡 流水线 = 队列解耦 = 真正并发！")
print("="*70 + "\n")
