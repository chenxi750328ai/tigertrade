# TigerTrade - AI驱动的期货交易系统

> **面向量化开发者/期货研究者**：基于 Tick 级数据与 Transformer，解决「不用人为指标、让模型自动发现特征」的量化研究与回测验证；目标月盈利率 20%，支持与 agent江湖 多 Agent 协作分工。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Active Development](https://img.shields.io/badge/status-active%20development-green.svg)]()

**状态页（GitHub Pages）**：[docs/status.html](https://chenxi750328ai.github.io/tigertrade/status.html)  
（需在仓库 **Settings → Pages** 里将 Source 选为 **GitHub Actions**，保存后由 workflow 自动部署。）

---

## 📋 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [项目架构](#项目架构)
- [技术栈](#技术栈)
- [项目计划](#项目计划)
- [项目状态](#项目状态)
- [参考资源](#参考资源)
- [团队协作](#团队协作)

**说明**：更多说明类文档见 `docs/readme/`；项目 PL 与发布约定见 `docs/PL.md`。

---

## 🎯 项目简介

> **本项目基于 [AgentFuture](/home/cx/agentfuture/) 多Agent协作框架构建** 🤝

TigerTrade是一个结合传统量化交易和现代深度学习的期货交易系统。通过Tiger Open API获取秒级真实市场数据，使用Transformer模型自动学习市场特征，实现智能化交易决策。

**特点**: 
- 利用AgentFuture的多Agent协作能力，实现数据采集、模型训练、策略回测的并行处理
- 支持投票选举最优策略
- Agent间共享交易洞察和市场发现

### 核心目标

> **💰 第一目标：实现稳定盈利，月盈利率 20%**

> **🛠️ 第二目标：构建高质量AI交易系统**

### 核心理念

> **"真正有效的特征隐藏在价格、成交量和时间轴中，而非人为设计的指标"**

- ✅ 使用原始Tick数据，而非RSI/BOLL/MACD等传统指标
- ✅ 让深度学习模型自动发现市场规律
- ✅ 从模型中提取可解释的交易信号
- ✅ 持续采集数据，提升策略泛化性
- 💰 **以盈利为导向，系统服务于盈利目标**

### 项目目标

> **核心目标：实现稳定盈利，目标月盈利率 20%**

#### 第一目标：盈利指标 💰

**核心KPI**：
- 🎯 **月盈利率目标**: 20%
- 📊 **夏普比率**: > 2.0
- 📉 **最大回撤**: < 10%
- ✅ **胜率**: > 60%
- 💹 **盈亏比**: > 2:1

**阶段性盈利目标**：
- **第1周**: 回测盈利率 > 15%（验证策略可行性）
- **第2-4周**: 小资金实盘测试，月化盈利 > 10%
- **第2-3月**: 扩大资金规模，稳定月盈利 15-20%
- **3-6月**: 达成月盈利 20%，控制回撤 < 10%

#### 第二目标：系统完成度 🛠️

**短期（1-2周）**：
- [x] 获取秒级真实市场Tick数据（已完成：39,192条）
- [x] 实现持续数据采集机制（已完成：每60秒自动采集）
- [x] 训练基于原生特征的Transformer模型（进行中）
- [ ] 从模型中提取可解释特征
- [ ] **回测验证：目标盈利率 > 15%** ⭐

**中期（1-3月）**：
- [ ] 积累多样化市场数据（上涨/下跌/震荡）
- [ ] 实现Market Regime自适应策略
- [ ] **小规模实盘：月化盈利 > 10%** ⭐
- [ ] 建立风险控制体系（止损、仓位管理）

**长期（3-6月）**：
- [ ] **稳定月盈利 20%** ⭐
- [ ] 多合约、多周期策略组合
- [ ] 自动化交易系统
- [ ] 持续学习和模型迭代

**风险提示**: 期货交易具有高风险，20%月盈利率是挑战性目标，需要严格的风险控制和持续优化。

---

## ✨ 核心特性

### 1. 秒级数据采集

```bash
# Tick数据持续采集器（自动运行）
bash scripts/启动Tick采集器.sh

# 特点：
# - 每60秒采集最新Tick
# - 自动按日期分文件、去重、合并
# - 每天积累14万条，1月400万条
```

**数据资产**：
- Tick数据：39,192条（6.38小时，秒级）
- 1分钟K线：2,382条（3天）
- 1小时K线：455条（30天）
- 日K线：60条（90天，+92.5%）
- **总计**：43,089条真实市场数据

### 2. 原生特征Transformer模型

```python
# 训练模型（不使用RSI/BOLL等人为指标）
python src/train_raw_features_transformer.py

# 模型架构：
# - 输入：原始价格、成交量、时间间隔（12维）
# - Transformer Encoder（4层，8头注意力）
# - 序列长度：128个时间点
# - 输出：买入/持有/卖出（3分类）
```

### 3. 特征发现与可解释性

```python
# 从模型中提取知识
python src/feature_discovery_from_model.py

# 分析内容：
# - 注意力权重：模型关注哪些历史时刻
# - 隐藏层表示：识别的市场状态
# - 特征重要性：哪些原始特征最关键
# - 自定义指标：可解释的交易信号
```

### 4. 用户风格策略模拟

基于真实交易记录（70条，15轮完整交易）：
- 分批建仓（网格加仓）
- 灵活止损（非固定比例）
- 目标收益导向

### 适用场景

- 期货量化研究、Tick 级策略开发
- Transformer 在金融时序中的应用与实验
- 白银/期货 API（Tiger Open API）数据采集与回测
- 与 agent江湖 多 Agent 协作：数据采集、模型训练、回测分工

---

## 🚀 快速开始

### 3 步最简上手

```bash
# 1. 克隆 + 安装依赖
git clone https://github.com/chenxi750328ai/tigertrade.git && cd tigertrade
pip install -r requirements.txt

# 2. 配置 Tiger API（或使用示例数据）→ 启动 Tick 采集
# 编辑 openapicfg_dem/tiger_openapi_config.properties 后：
bash scripts/启动Tick采集器.sh

# 3. 训练 + 回测
python src/train_raw_features_transformer.py
# 回测与特征分析见下方「完整步骤」
```

### 前置要求

- Python 3.10+
- CUDA 11.8+（推荐，用于GPU加速）
- Tiger Open API账户

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/tigertrade.git
cd tigertrade
```

### 2. 安装依赖

```bash
# 创建虚拟环境
conda create -n tigertrade python=3.10
conda activate tigertrade

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置Tiger API

```bash
# 编辑配置文件
vim openapicfg_dem/tiger_openapi_config.properties

# 填入你的凭证
tiger_id=你的tiger_id
private_key_path=./你的私钥文件路径
account=你的账户
```

### 4. 启动数据采集

```bash
# 后台启动Tick采集器
bash scripts/启动Tick采集器.sh

# 查看采集日志
tail -f /home/cx/trading_data/ticks/collector.log
```

### 5. 训练模型

```bash
# 准备数据
python scripts/prepare_data.py

# 训练Transformer模型
python src/train_raw_features_transformer.py

# 查看训练进度
bash scripts/查看训练进度.sh
```

### 6. 特征分析

```bash
# 从模型中提取知识
python src/feature_discovery_from_model.py

# 查看报告
cat docs/feature_discovery_report.json
```

### 7. 运行测试

```bash
# 方式一（推荐）：无 ROS 环境时运行，避免 launch_testing_ros 插件干扰
unset PYTHONPATH && python -m pytest tests/ -v

# 方式二：禁用 ROS 插件后运行
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v

# 运行测试并收集覆盖率
unset PYTHONPATH && python -m pytest tests/ -v --cov=src --cov-report=term-missing

# 运行特定测试文件
unset PYTHONPATH && python -m pytest tests/test_account_传递_端到端.py -v

# 查看覆盖率报告
python -m coverage report --show-missing
python -m coverage html  # 生成HTML报告，打开 htmlcov/index.html
```

**注意**：若系统中有 ROS/launch_testing_ros，pytest 可能报错「found no collectors」或「unknown hook pytest_launch_collect_makemodule」。请先执行 `unset PYTHONPATH` 再运行 pytest，或使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`。详见 [docs/pytest使用指南.md](docs/pytest使用指南.md)。

---

## 🏗️ 项目架构

### 目录结构

```
tigertrade/
├── README.md                    # 项目文档
├── requirements.txt             # Python依赖
├── openapicfg_dem/             # Tiger API配置
│   └── tiger_openapi_config.properties
│
├── src/                        # 源代码
│   ├── tick_data_collector.py         # Tick数据采集器
│   ├── train_raw_features_transformer.py  # Transformer训练
│   ├── feature_discovery_from_model.py    # 特征发现
│   ├── strategy_user_style.py         # 用户风格策略
│   └── backtest_engine.py              # 回测引擎
│
├── scripts/                    # 工具脚本
│   ├── backtest_user_style.py      # 策略回测
│   ├── import_to_rag.py            # RAG导入
│   └── prepare_data.py             # 数据准备
│
├── data/                       # 训练数据（git忽略）
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── models/                     # 训练好的模型
│   └── transformer_raw_features_best.pth
│
├── results/                    # 实验结果
│   ├── backtest_reports/
│   ├── training_logs/
│   └── feature_analysis/
│
├── docs/                       # 文档
│   ├── 项目架构.md
│   ├── API参考.md
│   ├── Tick数据采集完整方案.md
│   ├── 从模型中发现的特征.md
│   └── 真实Tick数据完整分析图.png
│
└── tests/                      # 测试
    ├── test_data_collector.py
    ├── test_model.py
    └── test_strategy.py
```

### 数据流

```
Tiger API → Tick采集器 → 本地存储 → 数据预处理 → Transformer模型
    ↓                      ↓              ↓              ↓
实时行情              按日期分文件      特征工程      交易信号
                                          ↓
                                   可解释指标
                                          ↓
                                    策略回测/实盘
```

### 技术架构

```
┌─────────────────────────────────────────────────┐
│                 Web界面/监控                      │
├─────────────────────────────────────────────────┤
│              策略引擎 (Strategy Engine)           │
│  ┌─────────────┬─────────────┬─────────────┐   │
│  │ Transformer │ User Style  │ Ensemble    │   │
│  │   Model     │  Strategy   │   Model     │   │
│  └─────────────┴─────────────┴─────────────┘   │
├─────────────────────────────────────────────────┤
│           特征工程 (Feature Engineering)          │
│  ┌─────────────┬─────────────┬─────────────┐   │
│  │ 原生特征    │ 注意力分析  │ 自定义指标  │   │
│  └─────────────┴─────────────┴─────────────┘   │
├─────────────────────────────────────────────────┤
│            数据层 (Data Layer)                   │
│  ┌─────────────┬─────────────┬─────────────┐   │
│  │ Tick采集器  │ K线数据     │ 历史数据库  │   │
│  └─────────────┴─────────────┴─────────────┘   │
├─────────────────────────────────────────────────┤
│              Tiger Open API                      │
└─────────────────────────────────────────────────┘
```

---

## 🛠️ 技术栈

### 核心框架

- **深度学习**: PyTorch 2.0+, Transformers
- **数据处理**: Pandas, NumPy, TA-Lib
- **可视化**: Matplotlib, Seaborn, Plotly
- **API**: Tiger Open API, FastAPI
- **数据库**: ChromaDB (向量数据库), SQLite (时序数据)

### 开发工具

- **版本控制**: Git, GitHub
- **项目管理**: GitHub Projects, Issues
- **代码质量**: Black (格式化), Pylint (检查)
- **测试**: Pytest, Coverage
- **文档**: Markdown, Sphinx
- **CI/CD**: GitHub Actions

---

## 📅 项目计划

**月度 / 周计划**（任务项、责任者、计划/实际时间、工作步骤、材料链接、风险与进展）：

- **[项目计划（月度·周计划）](docs/项目计划_月度周计划.md)** — 月度目标、周任务表、实际完成时间与备注
- **[项目状态（STATUS）](docs/STATUS.md)** — 当前状态摘要、当前迭代、近期完成项、风险与快速链接

---

## 📊 项目状态

> **STATUS 与计划**：[项目状态总览（STATUS）](docs/STATUS.md) | [项目计划（月度/周计划）](docs/项目计划_月度周计划.md)

### 已完成 ✅

#### 数据采集（2026-01-20/21）
- [x] Tiger API配置和认证
- [x] 实时Tick数据采集（39,192条）
- [x] 多周期K线数据获取
- [x] 持续采集机制（每60秒）
- [x] 数据质量验证（K线图）

#### 模型开发（2026-01-21）
- [x] 数据预处理和特征工程
- [x] Transformer模型架构设计
- [x] 训练/验证集划分（时间序列）
- [x] 模型训练脚本

#### 分析工具（2026-01-21）
- [x] 特征发现框架
- [x] 注意力权重分析
- [x] 隐藏层表示提取
- [x] 特征重要性计算

#### 策略回测
- [x] 用户风格策略实现
- [x] 回测引擎框架
- [x] 性能指标计算

#### 知识管理
- [x] RAG系统部署
- [x] 核心经验教训记录
- [x] API文档整理

### 进行中 🔄

- [ ] Transformer模型训练（Epoch 1/50）
- [ ] 特征发现分析（等待模型训练完成）
- [ ] Tick数据持续采集（后台运行）

### 待完成 📝

#### 短期（本周）
- [ ] 完成Transformer模型训练
- [ ] 运行特征发现分析
- [ ] 提取自定义可解释指标
- [ ] 与传统指标对比验证
- [ ] 完成回测报告

#### 中期（本月）
- [ ] 积累1个月多样化数据
- [ ] 实现Market Regime识别
- [ ] 多模型集成策略
- [ ] 风险控制系统
- [ ] 实盘小规模测试

#### 长期（3-6月）
- [ ] 多合约交易
- [ ] 自动化交易系统
- [ ] 策略持续优化
- [ ] 性能监控和告警

### 已知问题 ⚠️

1. **数据偏差**: 当前90天数据全部上涨(+92.5%)，存在过拟合风险
   - **解决**: 持续采集+数据增强+Market Regime识别

2. **历史Tick数据有限**: Tiger API只保存6-10小时Tick
   - **解决**: 持续采集器已启动，每天积累14万条

3. **模型标签不平衡**: 买入1.7%，持有96.7%，卖出1.7%
   - **解决**: 调整阈值或使用focal loss

---

## 📚 参考资源

### Tiger Open API

- **官方文档**: https://quant.tigerfintech.com/
- **SDK GitHub**: https://github.com/tigerfintech/openapi-python-sdk
- **行情API**: https://quant.tigerfintech.com/docs/market
- **交易API**: https://quant.tigerfintech.com/docs/trade

### 技术文档

- **Transformer原理**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **可解释AI**: [SHAP](https://github.com/slundberg/shap)
- **时序预测**: [Time Series Forecasting with Transformers](https://arxiv.org/abs/2001.08317)
- **量化交易**: [Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)

### 内部文档

- [Tick数据采集完整方案](docs/Tick数据采集完整方案.md)
- [从模型中发现的特征](docs/从模型中发现的特征.md)
- [项目架构详解](docs/项目架构.md)
- [API配置指南](docs/API配置指南.md)

### 数据可视化

- [真实Tick数据完整分析图](docs/真实Tick数据完整分析图.png)
- [白银期货完整K线图](docs/白银期货完整K线图_Tiger_API.png)

---

## 👥 团队协作

### 使用RAG系统快速上手

本项目使用RAG（检索增强生成）系统管理知识：

```bash
# 搜索相关知识
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "如何配置Tiger API", "top_k": 5}'

# 查看所有文档
curl http://localhost:8000/api/v1/documents

# 查看项目统计
curl http://localhost:8000/api/v1/stats
```

**RAG包含的关键信息**：
- ✅ 项目目标和状态
- ✅ 技术架构和方案
- ✅ 经验教训和规则
- ✅ API配置指南
- ✅ 数据资产清单
- ✅ 脚本和工具清单

### Agent协作指南

**新Agent加入流程**：

1. **阅读README** - 理解项目目标和现状
2. **查询RAG** - 搜索相关领域的经验教训
3. **检查TODO** - 了解当前待办任务
4. **运行测试** - 验证环境配置
5. **开始工作** - 选择任务并更新状态

**协作规范**：

```python
# 工作前：从RAG检索相关约束
query_rag("项目约束 文件组织")
query_rag("Tiger API 配置")

# 工作中：记录重要发现
add_to_rag(title="发现的问题", type="lesson_learned", ...)

# 工作后：更新TODO状态
update_todo(id="task-xxx", status="completed")
```

### GitHub项目管理

**Issues标签**：
- `bug` - 代码缺陷
- `feature` - 新功能
- `enhancement` - 功能改进
- `documentation` - 文档更新
- `data` - 数据相关
- `model` - 模型相关
- `strategy` - 策略相关

**分支策略**：
- `main` - 稳定版本
- `develop` - 开发版本
- `feature/*` - 功能分支
- `fix/*` - 修复分支

**Pull Request检查清单**：
- [ ] 代码通过Pylint检查
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
- [ ] 记录到RAG系统
- [ ] 通过所有测试

---

## 📝 变更日志

### 2026-01-21

**重大突破**：
- ✅ 成功获取39,192条秒级Tick数据
- ✅ 实现持续采集机制
- ✅ 训练基于原生特征的Transformer模型
- ✅ 创建特征发现分析框架
- ✅ RAG系统完全修复并导入核心知识

**数据资产**：
- 从0 → 43,089条真实市场数据
- 覆盖多个时间周期（秒级到日级）

**技术进展**：
- 从人为指标 → 原生特征学习
- 从黑盒模型 → 可解释特征提取

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- Tiger证券提供的优质API服务
- PyTorch和Transformers社区
- 所有贡献者和支持者

---

## 📞 联系方式

- **项目主页**: https://github.com/yourusername/tigertrade
- **Issues**: https://github.com/yourusername/tigertrade/issues
- **RAG系统**: http://localhost:8000/docs

---

**⚠️ 风险提示**: 
本项目仅供学习和研究使用，不构成任何投资建议。期货交易具有高风险，可能导致本金损失。请在充分理解风险的情况下谨慎决策。
