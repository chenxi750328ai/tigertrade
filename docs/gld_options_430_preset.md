# GLD 期权：三档卖 Call 预设 + 老虎拉线

与 `scripts/backtest_gld_options_futures_proxy.py` 一致：用 **Black-Scholes + 历史波动** 做简化回测；**欧式期末结算**，不含提前指派。

## 滚动（不是钉死 430）

- 口头行权价 **400 / 430 / 450** 是在 **报价参考价 S_ref≈430**（`--anchor-spot`）下说的。
- 脚本先算 **固定** `m_i = K_i/S_ref - 1`；**每个重开日**用 **当日 GLD 收盘价 S** 算 **`K = S×(1+m)`**，因此行权价**随 GLD 涨跌滚动**；S_ref **不**等于「每天 GLD 都是 430」。

## 经验口径

- 三档卖 Call：上述美元行权在 S_ref 时的比例；每桶相同张数（`--contracts-per-bucket`）。
- 时间价值经验：**约 $4/股/15 天**（卖方净收与脚本里「纯外在价值」会有偏差）。

## 数据：老虎 API（推荐）

1. 配置 `openapicfg` properties，并导出路径，例如：

   `export TIGEROPEN_PROPS=$HOME/tiger_openapi_config.properties`

2. 拉 GLD 日线到 CSV：

   ```bash
   python scripts/fetch_gld_daily_tiger.py -o data/gld_daily.csv --start 2010-01-01
   ```

3. 回测（或直接 `--tiger-props` 不写 CSV）：

   ```bash
   python scripts/backtest_gld_options_futures_proxy.py --preset-near430 \
     --tiger-props "$TIGEROPEN_PROPS" \
     --align-15d-cycle --print-anchor-extrinsic --calibrate-sigma-for-extrinsic \
     --extrinsic-target 4 --extrinsic-horizon-days 15 --contracts-per-bucket 2
   ```

- 数据源优先级：**`--csv` > `--tiger-props` > yfinance**。
- `--align-15d-cycle`：`dte=15`、`rebalance=10` 交易日。
- `--calibrate-sigma-for-extrinsic`：只影响 **打印用 σ**；**回测仍用滚动历史 σ**。
- 可选 `--preset-long-put-strike 400 --put-contracts 2` 叠保护腿。

## 网格优化

```bash
python scripts/optimize_gld_options_moneyness.py --tiger-props "$TIGEROPEN_PROPS" \
  --align-15d-cycle --include-430-ladder --strict-order \
  --objective calmar --top 15
```

`--include-430-ladder` 在 **仅 Short Call 三档**（未开 `--with-put-thirds`）时，还会在随机子采样后保证评估 **S_ref=430 时等价 400/430/450** 的 m 组合。

## 模型局限

见回测脚本顶部说明：GLD 代理期货、基差 `basis-drag-bps`、无指派跳空精细模拟等。
