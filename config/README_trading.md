# 交易配置 trading.json

- **trading_backend**：交易后端，如 `tiger`（老虎）、后续可扩展 `futu` 等。
- **symbol**：交易标的，如 `SIL.COMEX.202603`（白银期货）。
- **tick_size**：最小变动价位，如 COMEX 白银 `0.005`。

环境变量优先：`TRADING_BACKEND`、`TRADING_SYMBOL`、`TICK_SIZE`。  
示例见 `trading.json.example`。
