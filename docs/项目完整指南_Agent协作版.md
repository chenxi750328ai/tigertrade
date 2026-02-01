# TigerTrade项目完整指南 - Agent协作版

**更新时间**: 2026-01-21  
**目的**: 让任何Agent读完README和RAG后，立即能够启动工作

---

## 🎯 项目核心信息

### 项目定位
AI驱动的期货交易系统，使用深度学习从秒级Tick数据中自动学习交易策略

### 核心创新点
1. **原生特征学习** - 不使用RSI/BOLL等人为指标，让Transformer自动发现市场规律
2. **秒级数据** - 持续采集Tick数据（每60秒），积累海量训练数据
3. **可解释AI** - 从模型隐藏层提取知识，反向工程出可解释的交易信号

### 当前状态（2026-01-21）
```
✅ 数据采集：39,192条Tick + 43,089条多周期K线
✅ 持续采集：后台运行中（PID: 1353032）
⏳ 模型训练：Transformer训练中（Epoch 1/50）
⏳ 特征发现：等待模型完成
📋 下一步：提取可解释特征 → 回测验证
```

---

## 📚 快速上手（5分钟）

### 1. 阅读README
```bash
cat /home/cx/tigertrade/README.md
```
包含：项目简介、核心特性、快速开始、技术栈、项目状态

### 2. 查询RAG系统
```bash
# 搜索相关知识
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "你想了解的内容", "top_k": 5}'

# 示例查询
curl -X POST http://localhost:8000/api/v1/search \
  -d '{"query": "Tiger API配置", "top_k": 3}'
```

RAG当前包含**33个文档**：
- 7个knowledge（核心知识）
- 7个lesson_learned（经验教训）
- 7个reference（参考文档）
- 6个rule（规则约束）
- 2个todo（待办事项）
- 其他4个

### 3. 检查环境
```bash
# Python环境
python --version  # 应该是3.10+

# 依赖检查
pip list | grep -E "torch|pandas|tigeropen"

# Tiger API连接
python -c "from tigeropen.quote.quote_client import QuoteClient; print('✅ Tiger API OK')"

# RAG服务
curl http://localhost:8000/health
```

### 4. 查看当前进度
```bash
# Tick采集器状态
ps aux | grep tick_data_collector

# 模型训练进度
bash /home/cx/tigertrade/查看训练进度.sh

# 数据统计
ls -lh /home/cx/trading_data/ticks/
```

---

## 🏗️ 项目架构速览

### 目录结构
```
tigertrade/
├── src/                    # 核心源代码
│   ├── tick_data_collector.py      # ⭐ Tick采集器（后台运行）
│   ├── train_raw_features_transformer.py  # ⭐ Transformer训练
│   ├── feature_discovery_from_model.py    # ⭐ 特征发现
│   └── strategy_user_style.py            # 用户风格策略
│
├── scripts/                # 工具脚本
│   ├── backtest_user_style.py
│   └── import_to_rag.py
│
├── models/                 # 训练好的模型
│   └── transformer_raw_features_best.pth
│
├── docs/                   # 文档
│   ├── Tick数据采集完整方案.md
│   ├── 从模型中发现的特征.md
│   └── 真实Tick数据完整分析图.png
│
├── README.md               # ⭐ 项目主文档
├── requirements.txt        # Python依赖
└── .gitignore             # Git忽略规则
```

### 数据流
```
1. Tiger API
   ↓
2. Tick采集器（每60秒）
   ↓
3. 本地存储 (/home/cx/trading_data/ticks/)
   ↓
4. 数据预处理（标准化、特征工程）
   ↓
5. Transformer模型（12维输入 → 3分类输出）
   ↓
6. 特征发现（注意力分析、隐藏层聚类）
   ↓
7. 可解释指标
   ↓
8. 策略回测/实盘
```

---

## 🔧 关键配置

### Tiger API（重要！）
```bash
# 配置文件位置
/home/cx/openapicfg_dem/tiger_openapi_config.properties

# 注意事项
⚠️ 文件必须包含真实凭证（之前有过假配置导致mock数据的教训）
⚠️ private_key必须存在且正确
⚠️ DEMO账户也能获取真实市场数据

# 验证配置
python << 'EOF'
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.quote.quote_client import QuoteClient
config = TigerOpenClientConfig(props_path='./openapicfg_dem')
client = QuoteClient(config)
print("✅ Tiger API配置正确")
