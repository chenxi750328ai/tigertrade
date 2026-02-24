# 订单日志导出与分析

**生成时间**: 2026-02-24 16:09:41
**数据源**: `run/order_log.jsonl`
**总条数**: 1947

## 一、结论（是否「真的」在老虎后台）

| 类型 | 含义 | 是否会在老虎后台出现 |
|------|------|----------------------|
| **mode=mock** | 模拟单，未调用老虎 API | **不会** |
| **mode=real, status=success** | 已成功提交至老虎 API | **会**（请在老虎 DEMO/实盘账户对应时间、合约下查询） |
| **mode=real, status=fail** | 调用老虎 API 但被拒绝（如非交易时段、账户限制） | **不会** |

因此：若后台查不到订单，请先看下方统计中 **mode=real 且 status=success** 的数量与时间；若多为 mock 或 real 多为 fail，则后台无对应记录是预期行为。

**⚠️ 关于「与老虎后台对不上」**：当前 mode=real 且 status=success 的条数中，**多数 order_id 为 Mock、TEST_、ORDER_ 等**，说明来自**测试或 mock 环境**（当时未真正走老虎 API，或 API 被 mock 后把 mock 返回值写进了 order_log）。这类记录**不是老虎真实成交**，与老虎后台对不上是预期。只有 order_id 为**纯数字且较长**（老虎返回格式）的，才可能是真实单。详见下方「real success 中：疑似测试/mock 与可能真实单」。

## 二、统计汇总

### 按 mode 与 status

- mode=**real**, status=**fail**: 897 条
- mode=**real**, status=**success**: 638 条
- mode=**mock**, status=**success**: 412 条

### mode=real 且 status=success 中：疑似测试/mock 与可能真实单

| 类型 | 条数 | 说明 |
| --- | --- | --- |
| 疑似测试/mock（order_id 含 Mock、TEST_、ORDER_ 等） | 64 | 来自测试或 mock 环境，**非老虎真实成交**，与老虎后台对不上是预期。 |
| 可能为老虎真实单（order_id 为纯数字） | 574 | 才可能在老虎后台查到；可用 `scripts/verify_demo_orders_against_tiger.py` 核对。 |

### 按 source

- source=**auto**: 1908 条
- source=**manual**: 39 条

### 按日期（订单数）

| 日期 | 总条数 | real成功（会出现在老虎） | real失败 | mock成功 |
| --- | --- | --- | --- | --- |
| 2026-02-24 | 3 | 1 | 2 | 0 |
| 2026-02-11 | 616 | 568 | 47 | 1 |
| 2026-02-10 | 45 | 4 | 29 | 12 |
| 2026-02-09 | 82 | 4 | 56 | 22 |
| 2026-02-08 | 404 | 14 | 280 | 110 |
| 2026-02-06 | 160 | 4 | 112 | 44 |
| 2026-02-05 | 82 | 4 | 56 | 22 |
| 2026-02-03 | 85 | 5 | 57 | 23 |
| 2026-02-02 | 313 | 25 | 158 | 130 |
| 2026-02-01 | 157 | 9 | 100 | 48 |

*收益率按日统计需老虎后台成交明细或 API 拉取；本表仅订单条数按日。*

### mode=real 且 status=fail 的典型错误（前 15 条）

- `ALLOW_REAL_TRADING!=1`: 808 次
- `持仓硬顶 pos=999>=3`: 17 次
- `order_id无效(来自mock/测试)，未记入实盘成功`: 16 次
- `API Error 1010: biz param error(field 'account' cannot be empty)`: 16 次
- `cannot import name 'OrderSide' from 'tigeropen.common.consts' (/root/minicond...`: 13 次
- `code=1200 msg=standard account response error(bad_request:Orders cannot be pl...`: 10 次
- `下单失败: 无法创建Order对象或调用API - Order创建错误: account不能为空，无法创建订单。self.account=None, cl...`: 10 次
- `Order failed`: 7 次

### mode=real 且 status=success 最近 10 条时间戳（供与老虎后台核对）

- 2026-02-11T16:10:13.878784
- 2026-02-11T16:10:58.780421
- 2026-02-11T16:11:52.067872
- 2026-02-11T16:13:07.324301
- 2026-02-11T16:13:07.549711
- 2026-02-11T16:13:07.765699
- 2026-02-11T16:13:47.291553
- 2026-02-11T16:13:49.577313
- 2026-02-11T16:13:51.869034
- 2026-02-24T16:04:46.959026

## 三、说明

- 完整明细已导出为 CSV：`run/order_log_export.csv`（或通过 `--csv` 指定路径）。
- DEMO 运行（`tiger1 d moe`）时：若 SDK 初始化成功则使用真实 API（openapicfg_dem），订单为 mode=real；若初始化失败则走模拟，订单为 mode=mock。
- 老虎后台请使用 **DEMO 账户** 对应账户与时间范围查询；实盘账户与 DEMO 账户订单分离。
