# 原始 tiger1.py 与当前代码：下单与标的信息查询对比

## 一、原始版（根目录 tiger1.py，可下单成功）

### 1.1 配置与客户端

- **配置**：`TigerOpenClientConfig(props_path='./openapicfg_dem')`（demo 参数 `d`）。
- **客户端**：直接用 `QuoteClient(client_config)`、`TradeClient(client_config)`，**无 api_adapter 层**。
- **account**：`account_id = getattr(client_config, 'account', None)`，来自同一份 client_config。

### 1.2 标的信息查询（verify_api_connection）

- **交易所**：`quote_client.get_future_exchanges()`。
- **合约**：`quote_client.get_future_contracts('COMEX')`，用 `contracts.set_index('contract_code').loc['SIL2603']` 取 SIL2603。
- **行情**：`quote_client.get_future_brief(['SIL2603'])`、`quote_client.get_future_bars(['SIL2603'], ...)`。
- **结论**：标的信息全部用 **SIL2603**（简短格式），直接调 **QuoteClient**，无中间层。

### 1.3 下单（place_tiger_order）

- **account**：`account_id = getattr(client_config, 'account', None)`。
- **合约**：`contract_symbol = _to_api_identifier(FUTURE_SYMBOL)` → `'SIL2603'`；  
  `contract = future_contract(symbol=contract_symbol, currency=FUTURE_CURRENCY)`  
  **仅 symbol + currency，无 expiry、无 exchange**。
- **订单**：`order_obj = limit_order(account=account_id, contract=contract, action=side, limit_price=price, quantity=quantity)`。
- **提交**：**直接** `trade_client.place_order(order_obj)`，即 SDK 的 TradeClient 接收 Order 对象。

要点：**同一 client_config → 同一 account、同一 trade_client；合约只传 symbol + currency；直接 trade_client.place_order(order)。**

---

## 二、当前版（src/tiger1.py + api_adapter + OrderExecutor）

### 2.1 配置与客户端

- **配置**：仍是 `props_path='./openapicfg_dem'`（demo）。
- **客户端**：先建 `quote_client`、`trade_client`，再 `api_manager.initialize_real_apis(quote_client, trade_client, account=account_from_config)`，下单走 **RealTradeApiAdapter**（trade_api）。

### 2.2 标的信息查询

- **tiger1.verify_api_connection**：通过 `api_manager.quote_api`（封装了 quote_client），仍用 SIL2603。
- **OrderExecutor / 其他脚本**：若用 `api_manager.trade_api` 查单，account 来自 api_manager 初始化时的 account。

### 2.3 下单

- **place_tiger_order（src/tiger1.py）**：  
  `trade_api.place_order(symbol=symbol_for_api, side=..., order_type=..., quantity=..., time_in_force=..., limit_price=..., stop_price=None)`  
  即走 **RealTradeApiAdapter.place_order(symbol, side, ...)**。
- **RealTradeApiAdapter**：  
  - account：从 `self.account`（初始化时传入）或 client.config.account 或 api_manager._account 取。  
  - 合约：当前对 SIL2603 用了 `future_contract(symbol=..., currency=..., expiry='20260327', multiplier=1.0, exchange='COMEX')`，**比原始多了 expiry、exchange**。  
  - 订单：`limit_order(account=account, contract=contract, ...)`，再 `self.client.place_order(order)`（即同一 TradeClient）。

差异点：  
1. **合约**：原始只有 symbol+currency；当前对 SIL 加了 expiry、exchange。  
2. **account 来源**：原始直接 client_config.account；当前经 api_manager 初始化，若某路径未正确传入 account，可能不一致。

---

## 三、为何原始会成功、当前可能出问题

| 项目 | 原始（成功） | 当前 |
|------|--------------|------|
| account | 直接 client_config.account | 经 api_manager，多路径可能不一致 |
| 合约 | future_contract(symbol='SIL2603', currency=USD) | future_contract(..., expiry=..., exchange='COMEX') |
| 调用链 | tiger1 → trade_client.place_order(order) | tiger1 → trade_api.place_order(...) → adapter 建 Order → client.place_order(order) |

可能原因简述：

1. **合约格式**：原始只用 symbol+currency 即可成功；当前对 SIL 多加 expiry/exchange，若 SDK 或服务端对合约序列化/校验更严，有可能导致拒单或“账户未授权”等误导性错误。
2. **account 一致性**：若某处未用 openapicfg_dem 的 account 初始化 api_manager，或用了错误 account，会出现授权问题。

---

## 四、已做修改（对齐原始可成功行为）

1. **合约**（已做）：`api_adapter.py` 下单时与原始一致，仅用 **symbol + currency**：  
   `future_contract(symbol=symbol_to_use, currency=Currency.USD)`，不传 expiry、exchange。
2. **account**（已做）：`src/tiger1.py` 初始化 api_manager 时**优先用 client_config.account**（与原始一致），不再依赖 trade_client.config.account（可能为 None）。  
   - 模块加载时：`account_from_config = getattr(client_config, 'account', None) or getattr(trade_client.config, 'account', None)`  
   - `place_tiger_order` 内重新初始化时：同样优先 `client_config.account`。

这样当前链路在 account、合约、最终调用 TradeClient.place_order(order) 上与原始版一致。
