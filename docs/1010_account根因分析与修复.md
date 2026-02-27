# API 1010 account 为空 - 根因与修复

## 根因（已定位）

**Tiger SDK 的 `TradeClient.config` 为 `None`。**

验证：
```python
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.trade.trade_client import TradeClient
cfg = TigerOpenClientConfig(props_path='openapicfg_dem')
tc = TradeClient(cfg)
# client_config.account 有值，但 trade_client.config 为 None
assert getattr(cfg, 'account', None)  # 有值
assert getattr(tc, 'config', None)    # None
```

**错误用法**：`OrderExecutor._ensure_api_initialized` 以及 execute_buy/execute_sell 里“trade_api 为 None 时重新初始化”的逻辑，用 `getattr(t1.trade_client.config, 'account', None)` 取 account。因为 `trade_client.config` 为 None，得到的一直是 **None**，传入 `initialize_real_apis(account=None)`，fallback 若再失败则 `final_account` 为空，下单即 1010。

**为何真实 API 测试能过、DEMO 会挂**：测试里是 `client_config = TigerOpenClientConfig(...); account_to_use = client_config.account` 再 `initialize_real_apis(..., account=account_to_use)`，走的是 **client_config.account**，没有用 trade_client.config。DEMO 里若某次走到 OrderExecutor 的“重新初始化”路径（例如 trade_api 曾为 None），就会用 trade_client.config.account → None，导致 1010。

## 修复

在 `src/executor/order_executor.py` 中，所有“用 tiger1 的 client 初始化/重新初始化 API”的地方，**优先用 `t1.client_config.account`**，仅当没有 client_config 或 account 仍为空时，再回退到 `trade_client.config.account`：

- `_ensure_api_initialized`：account 先取 `getattr(t1.client_config, 'account', None)`，再按需用 trade_client.config
- execute_buy / execute_sell 中“trade_api 为 None 时重新初始化”的 account 逻辑同上

这样无论谁先初始化、是否走重新初始化，account 都来自 **client_config（openapicfg_dem）**，与真实 API 测试一致，避免 1010。
