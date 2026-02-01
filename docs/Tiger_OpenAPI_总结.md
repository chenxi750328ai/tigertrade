# Tiger Open API 使用总结

**文档来源**: https://quant.itigerup.com/openapi/zh/python/overview/introduction.html  
**总结日期**: 2026-01-21  
**目的**: 为量化交易项目提供API接口指南

---

## 📋 核心要点

### 1. 概述
- **定位**: 为个人开发者和机构客户提供接口服务
- **支持语言**: Java, Python, C++, C#
- **账户类型**: 综合账户、环球账户、**模拟账户（DEMO）**
- **收费**: **免费使用**（交易费用与APP一致，无额外API费用）

### 2. 主要功能
```
✅ 直接管理交易：创建/修改/取消订单，查询订单状态
✅ 查看账户信息：资产信息和持仓信息  
✅ 查询行情变动：股票、期权、期货价格及技术指标
✅ 订阅实时变动：订单、持仓、资产、行情实时推送
```

---

## 🌍 支持的市场和品种

### 期货交易支持情况

| 市场 | 综合账户 | 模拟账户 |
|------|---------|---------|
| 美国期货 | ✅ | ✅ |
| 香港期货 | ✅ | ✅ |
| 新加坡期货 | ✅ | ✅ |

**白银期货（本项目使用）**:
- 交易所: **COMEX**（纽约商品交易所）
- 合约代码: `SIL2503`, `SIL2505`等（年月格式）
- 行情权限: 需CME交易所权限
- 合约乘数: 1000盎司/手

---

## 📊 行情权限说明

### 美股行情
- **Level1标准行情**: 最新价格、逐笔成交、一档买卖
- **Level2高级行情**: 40档深度、量价分布、大单提示

### 期货行情
- **权限**: CME、CBOT等不同交易所需单独权限
- **重要**: **API行情权限独立于APP，需单独购买**

---

## 📈 K线数据获取（本项目核心）

### Python SDK使用示例

```python
from tigeropen.common.consts import Market, BarPeriod
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config

# 配置客户端
client_config = get_client_config(
    private_key_path='/path/to/private_key.pem',
    tiger_id='your_tiger_id',
    account='your_account'
)

quote_client = QuoteClient(client_config)

# 获取期货K线数据
bars = quote_client.get_future_bars(
    symbols=['SIL2503.US'],
    period=BarPeriod.DAY,  # 或 ONE_MINUTE, FIVE_MINUTES, ONE_HOUR
    begin_time=start_time_ms,  # epoch毫秒
    end_time=end_time_ms
)
```

### 支持的周期
```python
BarPeriod.ONE_MINUTE   # 1分钟
BarPeriod.FIVE_MINUTES # 5分钟
BarPeriod.ONE_HOUR     # 1小时
BarPeriod.DAY          # 日K
```

---

## 🔑 配置说明（重要）

### 配置文件结构
```
/home/cx/openapicfg_dem/
└── tiger_openapi_config.properties
```

### 配置内容
```properties
tiger_id=your_tiger_id
tiger_account=your_account
private_key_path=./path/to/private_key.pem
```

### Python代码使用
```python
from tigeropen.tiger_open_config import TigerOpenClientConfig

# 方式1：使用配置文件目录
client_config = TigerOpenClientConfig(props_path='./openapicfg_dem')

# 方式2：直接指定参数
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(
    private_key_path='/path/to/key.pem',
    tiger_id='your_id',
    account='your_account'
)
```

---

## 📦 订单类型

| 订单类型 | 说明 | 适用场景 |
|---------|------|---------|
| 市价单 | 以最新市场价成交 | 快速成交 |
| 限价单 | 指定价格才执行 | 锁定价格范围 |
| 止损单 | 达到止损价后提交市价单 | 保护利润/限定损失 |
| 止损限价单 | 达到止损价后提交限价单 | 避免成交价偏差过大 |
| 跟踪止损单 | 动态调整止损价 | 限制损失/不限收益 |
| 附加订单 | 主订单附带止盈/止损子订单 | 自动风控 |
| TWAP算法单 | 时间加权平均 | 大单分拆 |
| VWAP算法单 | 成交量加权平均 | 跟随市场节奏 |

---

## ⚠️ DEMO账户特性（本项目关键）

### DEMO账户能做什么
```
✅ 获取真实市场行情数据（Level1）
✅ 测试交易策略（模拟下单）
✅ 免费使用
✅ 支持股票、期货交易测试
✅ 支持任意时段下单（含盘前盘后）
```

### DEMO账户限制
```
❌ 无法获取完整历史数据（可能仅限近期）
❌ 行情权限需单独购买
❌ 模拟账户的下单时段可能有限制
```

### 关键发现
**DEMO账户提供的是真实市场行情数据，不是模拟数据！**
- 这意味着通过DEMO账户API获取的价格、成交量等都是真实市场数据
- 只有下单执行是模拟的
- 适合策略开发和回测

---

## 🛠️ 技术实现要点

### 1. 时间戳处理
```python
import time
from datetime import datetime, timedelta

# 转换为epoch毫秒
start_time_ms = int(datetime.now().timestamp() * 1000)

# Tiger API返回的时间戳通常是毫秒级
timestamp_s = timestamp_ms / 1000
dt = datetime.fromtimestamp(timestamp_s)
```

### 2. 数据验证
```python
# 检查是否获取到真实数据的方法
if bars and len(bars) > 0:
    df = pd.DataFrame(bars)
    
    # 验证1：时间范围合理性
    if 'time' in df.columns:
        latest_time = pd.to_datetime(df['time'], unit='ms')
        if latest_time.max().year > 2025:
            print("⚠️ 可能是Mock数据（时间戳异常）")
    
    # 验证2：价格连续性
    price_change = df['close'].diff().abs()
    if price_change.max() > df['close'].mean() * 0.5:
        print("⚠️ 价格跳跃异常，可能是Mock数据")
    
    # 验证3：成交量合理性
    if 'volume' in df.columns and df['volume'].mean() < 10:
        print("⚠️ 成交量异常低，可能是Mock数据")
```

### 3. 错误处理
```python
try:
    bars = quote_client.get_future_bars(...)
    if not bars or len(bars) == 0:
        print("未获取到数据，可能原因：")
        print("1. 合约代码不正确")
        print("2. 时间范围超出可用数据")
        print("3. 缺少行情权限")
except Exception as e:
    print(f"API调用失败: {e}")
```

---

## 💡 本项目遇到的问题和解决方案

### 问题1：获取的数据是Mock而非真实数据
**症状**:
- 时间戳异常（1970年或2026年）
- 价格变化不合理（短时间大幅波动）
- 数据量少且连续性差

**根本原因**:
- API客户端未正确初始化（quote_client = None）
- 代码中的fallback逻辑使用了Mock数据生成器
- 配置文件路径不正确

**解决方案**:
```python
# 1. 确保配置文件正确
# 路径: /home/cx/openapicfg_dem/tiger_openapi_config.properties

# 2. 正确初始化客户端
client_config = TigerOpenClientConfig(props_path='./openapicfg_dem')
quote_client = QuoteClient(client_config)

# 3. 验证客户端有效性
if quote_client is None:
    raise Exception("API客户端初始化失败")

# 4. 测试API连接
try:
    quote = quote_client.get_market_quote(symbols=['SIL2503.US'])
    if quote:
        print("✅ API连接正常")
except:
    print("❌ API连接失败")
```

### 问题2：数据分布偏差（泛化性风险）
**用户发现的关键问题**：
> "当前市场真实数据一直在上涨，在这种条件下是不是很容易达到盈利率？就算一直持有也能搞定。市场真实数据的分布是不是有误导作用，万一白银开始长期下跌呢？"

**问题本质**:
- 如果训练数据只包含上涨行情，模型会过拟合"做多"策略
- 真实市场有牛市、熊市、震荡市，单一行情训练的模型泛化性差
- 需要包含多种市场状态的数据

**解决方案**（下一步实施）:
1. 采集更长时间跨度的数据（包含上涨、下跌、震荡）
2. 数据增强：对历史数据做镜像/反转处理模拟相反行情
3. 分段训练：识别不同市场状态，分别训练
4. 正则化和集成学习：防止过拟合

---

## 📝 最佳实践总结

### 1. 数据采集
```python
✅ 使用真实API而非Mock数据
✅ 采集足够长的时间跨度（至少3-6个月）
✅ 包含不同市场状态（上涨/下跌/震荡）
✅ 验证数据质量（时间戳、价格连续性、成交量）
✅ 保存原始数据和清洗后数据
```

### 2. API使用
```python
✅ 先用DEMO账户测试，验证通过后再用真实账户
✅ 处理API限流（添加sleep）
✅ 错误重试机制（网络异常、数据缺失）
✅ 记录API调用日志
✅ 定期检查行情权限是否过期
```

### 3. 风险控制
```python
✅ 在代码中明确标识使用的是DEMO还是真实账户
✅ 真实账户操作需要额外确认机制
✅ 设置最大仓位和单日亏损限制
✅ 使用止损订单保护
```

---

## 🔗 重要链接

- **新文档入口（推荐）**: https://quant.itigerup.com/openapi/zh/python/overview/introduction.html
- **收费及权限**: 查看官网详情
- **FAQ**: 常见问题解答
- **问题反馈**: 遇到问题时联系技术支持

---

## ⚡ 关键经验教训

1. **DEMO账户 ≠ Mock数据**
   - DEMO账户提供**真实行情**，只是下单执行是模拟的
   - 之前误以为DEMO账户只能获取Mock数据，导致数据采集问题

2. **配置路径很重要**
   - 相对路径 `./openapicfg_dem` 需要在正确的工作目录下运行
   - 绝对路径 `/home/cx/openapicfg_dem/` 更可靠

3. **数据验证必不可少**
   - 不能假设API返回的就是真实数据
   - 必须检查时间戳、价格范围、成交量等
   - 可视化检查（K线图）是最直观的验证方法

4. **行情权限需要购买**
   - 即使是DEMO账户，期货行情权限也需要单独购买
   - 没有权限时API可能返回空数据或降级到Mock数据

5. **泛化性比准确率更重要**
   - 单一市场状态下的高准确率可能是假象
   - 必须在多种市场环境下验证策略
   - 防止"牛市策略"在熊市中崩溃

---

**最后更新**: 2026-01-21  
**状态**: 文档已完成，待写入RAG系统
