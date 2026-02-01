# DEMO运行状态查询指南

**日期**: 2026-01-29  
**用途**: 监控DEMO交易策略运行状态

## 一、快速查询命令

### 1.1 查看进程状态
```bash
# 查看DEMO进程是否运行
ps aux | grep -E "run_moe_demo|python.*demo" | grep -v grep

# 查看进程详细信息（PID、运行时长、命令）
ps -p $(pgrep -f "run_moe_demo") -o pid,etime,cmd --no-headers
```

### 1.2 查看最新日志
```bash
# 查看最新的日志文件
ls -lt /home/cx/tigertrade/demo_run_20h_*.log | head -1 | awk '{print $NF}' | xargs tail -50

# 实时监控日志（推荐）
ls -lt /home/cx/tigertrade/demo_run_20h_*.log | head -1 | awk '{print $NF}' | xargs tail -f
```

### 1.3 查看关键信息
```bash
# 查看下单相关日志
ls -lt /home/cx/tigertrade/demo_run_20h_*.log | head -1 | awk '{print $NF}' | xargs grep -E "下单|place_order|Order创建" | tail -20

# 查看错误日志
ls -lt /home/cx/tigertrade/demo_run_20h_*.log | head -1 | awk '{print $NF}' | xargs grep -E "❌|ERROR|错误|失败" | tail -20

# 查看成功日志
ls -lt /home/cx/tigertrade/demo_run_20h_*.log | head -1 | awk '{print $NF}' | xargs grep -E "✅|成功|SUCCESS" | tail -20
```

## 二、详细查询方法

### 2.1 进程状态查询
```bash
cd /home/cx/tigertrade

# 1. 查找DEMO进程
DEMO_PID=$(pgrep -f "run_moe_demo")
echo "进程ID: $DEMO_PID"

# 2. 查看进程运行时长
ps -p $DEMO_PID -o etime,cmd --no-headers

# 3. 查看进程资源使用
ps -p $DEMO_PID -o pid,%cpu,%mem,vsz,rss,etime,cmd
```

### 2.2 日志文件查询
```bash
cd /home/cx/tigertrade

# 1. 找到最新的日志文件
LATEST_LOG=$(ls -t demo_run_20h_*.log 2>/dev/null | head -1)
echo "最新日志: $LATEST_LOG"

# 2. 查看日志文件大小
ls -lh $LATEST_LOG

# 3. 查看最后100行
tail -100 $LATEST_LOG

# 4. 实时监控（Ctrl+C退出）
tail -f $LATEST_LOG
```

### 2.3 关键指标查询

#### 下单统计
```bash
cd /home/cx/tigertrade
LATEST_LOG=$(ls -t demo_run_20h_*.log 2>/dev/null | head -1)

# 下单尝试次数
grep -c "下单调试" $LATEST_LOG

# 下单成功次数
grep -c "下单成功\|Order创建成功" $LATEST_LOG

# 下单失败次数
grep -c "下单失败\|下单异常\|授权失败" $LATEST_LOG
```

#### 错误统计
```bash
cd /home/cx/tigertrade
LATEST_LOG=$(ls -t demo_run_20h_*.log 2>/dev/null | head -1)

# 错误总数
grep -c "❌\|ERROR\|错误" $LATEST_LOG

# 授权错误
grep -c "授权失败\|not authorized" $LATEST_LOG

# 最近10个错误
grep "❌\|ERROR\|错误" $LATEST_LOG | tail -10
```

#### 策略预测统计
```bash
cd /home/cx/tigertrade
LATEST_LOG=$(ls -t demo_run_20h_*.log 2>/dev/null | head -1)

# 预测次数
grep -c "MoE Transformer预测" $LATEST_LOG

# 买入预测次数
grep -c "动作: 买入" $LATEST_LOG

# 卖出预测次数
grep -c "动作: 卖出" $LATEST_LOG

# 不操作预测次数
grep -c "动作: 不操作" $LATEST_LOG
```

## 三、一键查询脚本

### 3.1 创建查询脚本
```bash
cat > /home/cx/tigertrade/scripts/check_demo_status.sh << 'EOF'
#!/bin/bash
# DEMO运行状态查询脚本

cd /home/cx/tigertrade

echo "=========================================="
echo "DEMO运行状态查询"
echo "=========================================="
echo

# 1. 进程状态
echo "📊 进程状态:"
DEMO_PID=$(pgrep -f "run_moe_demo")
if [ -z "$DEMO_PID" ]; then
    echo "❌ DEMO进程未运行"
else
    echo "✅ 进程ID: $DEMO_PID"
    ps -p $DEMO_PID -o etime,cmd --no-headers | awk '{print "   运行时长: " $1}'
fi
echo

# 2. 日志文件
echo "📄 日志文件:"
LATEST_LOG=$(ls -t demo_run_20h_*.log 2>/dev/null | head -1)
if [ -z "$LATEST_LOG" ]; then
    echo "❌ 未找到日志文件"
else
    echo "✅ 最新日志: $LATEST_LOG"
    ls -lh $LATEST_LOG | awk '{print "   文件大小: " $5}'
    echo "   最后更新: $(stat -c %y $LATEST_LOG | cut -d. -f1)"
fi
echo

# 3. 关键指标
if [ ! -z "$LATEST_LOG" ]; then
    echo "📈 关键指标:"
    echo "   下单尝试: $(grep -c "下单调试" $LATEST_LOG 2>/dev/null || echo 0)"
    echo "   下单成功: $(grep -c "Order创建成功" $LATEST_LOG 2>/dev/null || echo 0)"
    echo "   下单失败: $(grep -c "下单失败\|下单异常\|授权失败" $LATEST_LOG 2>/dev/null || echo 0)"
    echo "   预测次数: $(grep -c "MoE Transformer预测" $LATEST_LOG 2>/dev/null || echo 0)"
    echo "   错误总数: $(grep -c "❌\|ERROR\|错误" $LATEST_LOG 2>/dev/null || echo 0)"
fi
echo

# 4. 最近日志（最后10行）
if [ ! -z "$LATEST_LOG" ]; then
    echo "📋 最近日志（最后10行）:"
    tail -10 $LATEST_LOG | sed 's/^/   /'
fi

echo "=========================================="
EOF

chmod +x /home/cx/tigertrade/scripts/check_demo_status.sh
```

### 3.2 使用查询脚本
```bash
# 运行查询脚本
/home/cx/tigertrade/scripts/check_demo_status.sh

# 或使用快捷方式
cd /home/cx/tigertrade && ./scripts/check_demo_status.sh
```

## 四、监控命令示例

### 4.1 实时监控
```bash
# 实时监控日志（推荐）
cd /home/cx/tigertrade
LATEST_LOG=$(ls -t demo_run_20h_*.log 2>/dev/null | head -1)
tail -f $LATEST_LOG
```

### 4.2 定期检查
```bash
# 每30秒检查一次状态
watch -n 30 '/home/cx/tigertrade/scripts/check_demo_status.sh'
```

### 4.3 关键事件监控
```bash
# 只监控下单相关事件
cd /home/cx/tigertrade
LATEST_LOG=$(ls -t demo_run_20h_*.log 2>/dev/null | head -1)
tail -f $LATEST_LOG | grep -E "下单|Order|授权|错误"
```

## 五、常见问题排查

### 5.1 进程不存在
```bash
# 检查进程
ps aux | grep run_moe_demo | grep -v grep

# 如果不存在，检查是否有错误日志
ls -lt /home/cx/tigertrade/*.log | head -5
```

### 5.2 日志文件不更新
```bash
# 检查进程是否还在运行
ps -p $(pgrep -f "run_moe_demo")

# 检查磁盘空间
df -h /home/cx/tigertrade
```

### 5.3 授权错误
```bash
# 查看授权错误详情
cd /home/cx/tigertrade
LATEST_LOG=$(ls -t demo_run_20h_*.log 2>/dev/null | head -1)
grep "授权失败\|not authorized" $LATEST_LOG | tail -5
```

## 六、Python查询脚本

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEMO运行状态查询脚本（Python版本）
"""
import os
import subprocess
import glob
from datetime import datetime

def check_demo_status():
    """检查DEMO运行状态"""
    print("=" * 60)
    print("DEMO运行状态查询")
    print("=" * 60)
    print()
    
    # 1. 检查进程
    print("📊 进程状态:")
    try:
        result = subprocess.run(
            ["pgrep", "-f", "run_moe_demo"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            pid = result.stdout.strip()
            print(f"✅ 进程ID: {pid}")
            
            # 获取运行时长
            result = subprocess.run(
                ["ps", "-p", pid, "-o", "etime", "--no-headers"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"   运行时长: {result.stdout.strip()}")
        else:
            print("❌ DEMO进程未运行")
    except Exception as e:
        print(f"❌ 检查进程失败: {e}")
    print()
    
    # 2. 检查日志文件
    print("📄 日志文件:")
    log_files = sorted(glob.glob("/home/cx/tigertrade/demo_run_20h_*.log"), key=os.path.getmtime, reverse=True)
    if log_files:
        latest_log = log_files[0]
        print(f"✅ 最新日志: {os.path.basename(latest_log)}")
        size = os.path.getsize(latest_log)
        print(f"   文件大小: {size / 1024 / 1024:.2f} MB")
        mtime = datetime.fromtimestamp(os.path.getmtime(latest_log))
        print(f"   最后更新: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 3. 关键指标
        print()
        print("📈 关键指标:")
        with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            print(f"   下单尝试: {content.count('下单调试')}")
            print(f"   下单成功: {content.count('Order创建成功')}")
            print(f"   下单失败: {content.count('下单失败') + content.count('下单异常') + content.count('授权失败')}")
            print(f"   预测次数: {content.count('MoE Transformer预测')}")
            print(f"   错误总数: {content.count('❌') + content.count('ERROR')}")
        
        # 4. 最近日志
        print()
        print("📋 最近日志（最后10行）:")
        with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(f"   {line.rstrip()}")
    else:
        print("❌ 未找到日志文件")
    
    print()
    print("=" * 60)

if __name__ == '__main__':
    check_demo_status()
```

---

**使用建议**:
1. 日常监控：使用 `check_demo_status.sh` 脚本
2. 实时监控：使用 `tail -f` 查看日志
3. 问题排查：查看错误日志和关键指标
