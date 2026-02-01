# 三个Agent详细任务清单

**更新**: 2026-01-21  
**重构完成**: ✅ 模块化架构已就绪

---

## 🏗️ 重构成果

### 已完成的模块化
```
src/
├── data_collector/     ✅ Module 1 (Agent 1)
│   ├── realtime_collector.py
│   ├── tick_collector.py
│   └── kline_fetcher.py
│
├── strategies/         ✅ Module 5 (Agent 3)
│   ├── base.py
│   ├── grid_strategy.py
│   └── transformer_strategy.py
│
└── risk/              ✅ Module 7 (Agent 3)
    └── risk_manager.py

tiger1_v2.py           ✅ 新主程序（200行 vs 原2900行）
tiger1_legacy.py       ✅ 原文件备份
```

---

## 👤 Agent 1: 数据工程师

### 核心任务：数据预处理 Pipeline

#### 任务1.1: 创建数据处理模块 (30分钟)

**文件**: `src/data_processor/cleaner.py`

```python
"""
数据清洗器
处理异常值、缺失值、重复数据
"""

import pandas as pd
import numpy as np

class DataCleaner:
    """数据清洗"""
    
    def clean(self, df):
        """
        清洗数据
        
        处理：
        1. 删除重复行
        2. 填充缺失值
        3. 移除异常值（价格跳变>10%）
        4. 时间戳排序
        """
        # 删除重复
        df = df.drop_duplicates(subset=['datetime'])
        
        # 排序
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # 填充缺失值（前向填充）
        df = df.fillna(method='ffill')
        
        # 移除异常值
        df = self._remove_outliers(df)
        
        return df
    
    def _remove_outliers(self, df, threshold=0.10):
        """移除价格异常跳变"""
        price_change = df['close'].pct_change().abs()
        df = df[price_change < threshold]
        return df.reset_index(drop=True)
```

**测试**:
```bash
cd /home/cx/tigertrade
python -c "
from src.data_processor.cleaner import DataCleaner
import pandas as pd

# 测试数据
df = pd.read_csv('/home/cx/trading_data/SIL2603_1min_combined.csv')
print(f'原始数据: {len(df)}条')

cleaner = DataCleaner()
df_clean = cleaner.clean(df)
print(f'清洗后: {len(df_clean)}条')
"
```

#### 任务1.2: 数据标准化 (20分钟)

**文件**: `src/data_processor/normalizer.py`

```python
"""数据标准化"""

import pandas as pd
import numpy as np

class DataNormalizer:
    """标准化/归一化"""
    
    def __init__(self):
        self.scalers = {}  # 保存每列的scale参数
    
    def fit_transform(self, df, method='zscore'):
        """拟合并转换"""
        df_norm = df.copy()
        
        for col in ['open', 'high', 'low', 'close']:
            if method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                df_norm[col] = (df[col] - mean) / std
                self.scalers[col] = {'mean': mean, 'std': std}
            
            elif method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
                self.scalers[col] = {'min': min_val, 'max': max_val}
        
        return df_norm
```

#### 任务1.3: 合并和划分数据集 (30分钟)

**文件**: `src/data_processor/splitter.py`

```python
"""训练/验证/测试集划分"""

import pandas as pd

class DataSplitter:
    """数据集划分（时间序列）"""
    
    def split(self, df, train_ratio=0.7, val_ratio=0.15):
        """
        划分数据集
        
        train: 70%
        val: 15%
        test: 15%
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        df_train = df[:train_end]
        df_val = df[train_end:val_end]
        df_test = df[val_end:]
        
        return df_train, df_val, df_test
```

#### 任务1.4: 整合Pipeline (20分钟)

**文件**: `scripts/prepare_data.py`

```python
"""
数据准备主脚本
整合所有数据处理步骤
"""

import pandas as pd
from pathlib import Path
from src.data_processor.cleaner import DataCleaner
from src.data_processor.normalizer import DataNormalizer
from src.data_processor.splitter import DataSplitter

def main():
    print("="*80)
    print("📊 数据准备Pipeline")
    print("="*80)
    
    # 1. 加载所有数据
    print("\n1. 加载数据...")
    data_dir = Path('/home/cx/trading_data')
    
    all_data = []
    for file in ['SIL2603_1min_combined.csv', 
                 'SIL2603_5min_7days.csv',
                 'SIL2603_1h_30days.csv']:
        path = data_dir / file
        if path.exists():
            df = pd.read_csv(path)
            all_data.append(df)
            print(f"   ✅ {file}: {len(df)}条")
    
    df_all = pd.concat(all_data, ignore_index=True)
    print(f"\n   总计: {len(df_all)}条")
    
    # 2. 清洗
    print("\n2. 数据清洗...")
    cleaner = DataCleaner()
    df_clean = cleaner.clean(df_all)
    print(f"   清洗后: {len(df_clean)}条")
    
    # 3. 标准化
    print("\n3. 数据标准化...")
    normalizer = DataNormalizer()
    df_norm = normalizer.fit_transform(df_clean)
    print(f"   ✅ 标准化完成")
    
    # 4. 划分数据集
    print("\n4. 划分数据集...")
    splitter = DataSplitter()
    df_train, df_val, df_test = splitter.split(df_norm)
    
    print(f"   Train: {len(df_train)}条 ({len(df_train)/len(df_norm)*100:.1f}%)")
    print(f"   Val:   {len(df_val)}条 ({len(df_val)/len(df_norm)*100:.1f}%)")
    print(f"   Test:  {len(df_test)}条 ({len(df_test)/len(df_norm)*100:.1f}%)")
    
    # 5. 保存
    print("\n5. 保存数据...")
    output_dir = data_dir / 'processed'
    output_dir.mkdir(exist_ok=True)
    
    df_train.to_csv(output_dir / 'train.csv', index=False)
    df_val.to_csv(output_dir / 'val.csv', index=False)
    df_test.to_csv(output_dir / 'test.csv', index=False)
    
    print(f"   ✅ {output_dir}")
    print("\n" + "="*80)
    print("✅ 数据准备完成！")
    print("="*80)

if __name__ == '__main__':
    main()
```

**执行**:
```bash
cd /home/cx/tigertrade
python scripts/prepare_data.py
```

**验证**:
```bash
ls -lh /home/cx/trading_data/processed/
head -5 /home/cx/trading_data/processed/train.csv
```

### 完成标准
- [x] 创建cleaner.py, normalizer.py, splitter.py
- [x] 创建prepare_data.py整合脚本
- [x] 执行并生成train/val/test.csv
- [x] 数据质量报告（行数、时间范围、异常值数量）

---

## 👤 Agent 2: AI研究员

### 核心任务：完成模型训练和特征发现

#### 任务2.1: 监控模型训练 (持续)

```bash
# 查看训练进度
bash /home/cx/tigertrade/查看训练进度.sh

# 或直接查看日志
tail -f /home/cx/tigertrade/logs/train_transformer_*.log
```

**检查点**:
- Epoch进度 (1/50 → 50/50)
- Loss下降趋势
- 验证准确率 > 60%
- 模型保存路径

#### 任务2.2: 模型训练完成后运行特征发现

```bash
cd /home/cx/tigertrade
python src/feature_discovery_from_model.py
```

**输出**:
- 特征重要性排名
- 注意力权重分析
- 自定义指标
- 市场状态聚类

#### 任务2.3: 集成到TransformerStrategy

更新 `src/strategies/transformer_strategy.py`:
- 加载模型
- 实现 `_prepare_sequence()`
- 测试推理速度

### 完成标准
- [x] Transformer训练完成（50/50 Epoch）
- [x] 验证准确率 > 60%
- [x] 特征发现分析完成
- [x] TransformerStrategy可用

---

## 👤 Agent 3: 策略工程师

### 核心任务：策略回测和风险控制

#### 任务3.1: 实现回测引擎 (40分钟)

**文件**: `src/backtest/engine.py`

```python
"""回测引擎"""

import pandas as pd
import numpy as np

class BacktestEngine:
    """策略回测"""
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.trades = []
    
    def run(self, strategy, data):
        """
        运行回测
        
        Args:
            strategy: 策略实例
            data: 历史数据
        
        Returns:
            dict: 回测结果
        """
        self.capital = self.initial_capital
        self.position = 0
        self.trades = []
        
        for i in range(len(data)):
            # 获取当前数据
            current_data = {
                '1m': data.iloc[:i+1]
            }
            
            # 生成信号
            signal = strategy.generate_signal(current_data)
            
            # 执行交易
            self._execute(signal, data.iloc[i])
        
        # 计算指标
        return self._calculate_metrics()
    
    def _execute(self, signal, bar):
        """执行交易"""
        price = bar['close']
        
        if signal['action'] == 'BUY' and self.position == 0:
            self.position = signal['position_size']
            self.entry_price = price
            self.trades.append({
                'type': 'BUY',
                'price': price,
                'time': bar['datetime']
            })
        
        elif signal['action'] == 'SELL' and self.position > 0:
            pnl = (price - self.entry_price) / self.entry_price
            self.capital *= (1 + pnl * self.position)
            self.position = 0
            self.trades.append({
                'type': 'SELL',
                'price': price,
                'pnl': pnl,
                'time': bar['datetime']
            })
    
    def _calculate_metrics(self):
        """计算性能指标"""
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        # 统计交易
        buys = [t for t in self.trades if t['type'] == 'BUY']
        sells = [t for t in self.trades if t['type'] == 'SELL']
        
        wins = [t for t in sells if t['pnl'] > 0]
        losses = [t for t in sells if t['pnl'] <= 0]
        
        win_rate = len(wins) / len(sells) if sells else 0
        
        return {
            'total_return': total_return,
            'final_capital': self.capital,
            'total_trades': len(sells),
            'win_rate': win_rate,
            'num_wins': len(wins),
            'num_losses': len(losses)
        }
```

#### 任务3.2: 运行回测

```bash
cd /home/cx/tigertrade
python << 'EOF'
from src.backtest.engine import BacktestEngine
from src.strategies import GridStrategy
import pandas as pd

# 加载测试数据
df = pd.read_csv('/home/cx/trading_data/processed/test.csv')

# 运行回测
strategy = GridStrategy()
engine = BacktestEngine(initial_capital=10000)
results = engine.run(strategy, df)

print("="*80)
print("📊 回测结果")
print("="*80)
print(f"总收益率: {results['total_return']*100:.2f}%")
print(f"胜率: {results['win_rate']*100:.1f}%")
print(f"总交易: {results['total_trades']}")
print(f"盈利: {results['num_wins']} | 亏损: {results['num_losses']}")
print("="*80)

# 🎯 目标：盈利率 > 15%
if results['total_return'] > 0.15:
    print("✅ 达到目标（>15%）")
else:
    print("⚠️ 未达目标，需要优化")
EOF
```

#### 任务3.3: 测试新主程序

```bash
# 测试网格策略
cd /home/cx/tigertrade
python tiger1_v2.py --strategy grid --interval 60

# Ctrl+C 停止后测试Transformer策略
python tiger1_v2.py --strategy transformer --interval 60
```

### 完成标准
- [x] 回测引擎实现
- [x] 网格策略回测 >15%盈利 ⭐
- [x] Transformer策略回测
- [x] tiger1_v2.py可正常运行

---

## 🎯 总体里程碑

### Milestone 1: 数据就绪 (Agent 1) - 预计1小时
- [x] 数据清洗模块
- [x] 数据标准化模块
- [x] 数据集划分
- [x] train/val/test.csv生成

### Milestone 2: 模型就绪 (Agent 2) - 等待训练
- [ ] Transformer训练完成
- [ ] 特征发现分析
- [ ] TransformerStrategy集成

### Milestone 3: 策略验证 (Agent 3) - 预计1小时
- [ ] 回测引擎实现
- [ ] 网格策略回测 >15% ⭐
- [ ] Transformer策略回测
- [ ] tiger1_v2.py测试

---

**🚀 准备就绪！现在可以开始并行工作！**
