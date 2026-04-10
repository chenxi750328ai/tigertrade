# 订单日志导出与分析

**生成时间**: 2026-04-10 16:26:41
**数据源**: `run/order_log.jsonl`
**总条数**: 1993

## 一、结论（是否「真的」在老虎后台）

| 类型 | 含义 | 是否会在老虎后台出现 |
|------|------|----------------------|
| **mode=mock** | 模拟单，未调用老虎 API | **不会** |
| **mode=real, status=success** | order_log 记录（本系统认为已提交） | **未必会**：含测试/mock 污染，与老虎后台对不上是常态，**成功数仅以老虎后台为准** |
| **mode=real, status=fail** | 调用老虎 API 但被拒绝（如非交易时段、账户限制） | **不会** |

因此：若后台查不到订单，请先看下方统计中 **mode=real 且 status=success** 的数量与时间；若多为 mock 或 real 多为 fail，则后台无对应记录是预期行为。

**⚠️ 关于「与老虎后台对不上」**：当前 mode=real 且 status=success 的条数中，**多数 order_id 为 Mock、TEST_、ORDER_ 等**，说明来自**测试或 mock 环境**（当时未真正走老虎 API，或 API 被 mock 后把 mock 返回值写进了 order_log）。这类记录**不是老虎真实成交**，与老虎后台对不上是预期。只有 order_id 为**纯数字且较长**（老虎返回格式）的，才可能是真实单。详见下方「real success 中：疑似测试/mock 与可能真实单」。

## 二、统计汇总

### 按 mode 与 status

- mode=**real**, status=**fail**: 919 条
- mode=**real**, status=**success**: 662 条
- mode=**mock**, status=**success**: 412 条

### mode=real 且 status=success 中：疑似测试/mock 与可能真实单

| 类型 | 条数 | 说明 |
| --- | --- | --- |
| 疑似测试/mock（order_id 含 Mock、TEST_、ORDER_ 等） | 64 | 来自测试或 mock 环境，**非老虎真实成交**，与老虎后台对不上是预期。 |
| 可能为老虎真实单（order_id 为纯数字） | 598 | **未与老虎核对**，可能含测试污染；真实成功数仅以老虎后台为准。 |

### 按 source

- source=**auto**: 1953 条
- source=**manual**: 40 条

### 按日期（订单数）

| 日期 | 总条数 | order_log 记成功（未核对） | real失败 | mock成功 |
| --- | --- | --- | --- | --- |
| 2026-04-02 | 4 | 4 | 0 | 0 |
| 2026-04-01 | 4 | 4 | 0 | 0 |
| 2026-03-31 | 22 | 12 | 10 | 0 |
| 2026-03-26 | 2 | 0 | 2 | 0 |
| 2026-02-25 | 11 | 3 | 8 | 0 |
| 2026-02-24 | 6 | 2 | 4 | 0 |
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

### mode=real 且 status=fail 分类

- **本地风控拒绝**（未发 API）：851 条（如 ALLOW_REAL_TRADING、持仓硬顶）
- **真实 API 被拒**（老虎拒绝）：68 条（如 1010/1200/account 为空）

### `ALLOW_REAL_TRADING!=1` 按 run_env 分解（新日志才有 run_env）

| run_env | 条数 | 说明 |
| --- | --- | --- |
| 未记录 | 808 | 旧日志无 `run_env` 字段，或 pytest/CI 混入 |

**说明**：若你**一直用 DEMO**（`python src/tiger1.py d ...`），`RUN_ENV=sandbox`，**代码路径不会**产生 `ALLOW_REAL_TRADING!=1`。汇总里的 808 次主要来自**未记录**、**production** 或**历史/测试**写入，并非 DEMO 正常下单。

### mode=real 且 status=fail 的典型错误（前 15 条）

- `ALLOW_REAL_TRADING!=1`: 808 次
- `order_id无效(来自mock/测试)，未记入实盘成功`: 26 次
- `API Error 1010: biz param error(field 'account' cannot be empty)`: 26 次
- `持仓硬顶 pos=999>=3`: 17 次
- `cannot import name 'OrderSide' from 'tigeropen.common.consts' (/root/minicond...`: 13 次
- `code=1200 msg=standard account response error(bad_request:Orders cannot be pl...`: 10 次
- `下单失败: 无法创建Order对象或调用API - Order创建错误: account不能为空，无法创建订单。self.account=None, cl...`: 10 次
- `Order failed`: 7 次
- `订单提交后后台未查到，可能被拒或延迟，请核对后台`: 1 次
- `订单提交后 8 秒内后台未查到，可能被拒或延迟，请核对后台`: 1 次

### mode=real 且 status=success 最近 10 条时间戳（供与老虎后台核对）

- 2026-03-31T21:21:25.863838
- 2026-03-31T21:21:26.068565
- 2026-04-01T23:26:37.475571
- 2026-04-01T23:28:59.933539
- 2026-04-01T23:30:39.641394
- 2026-04-01T23:35:17.978695
- 2026-04-02T10:18:28.639922
- 2026-04-02T18:39:06.104102
- 2026-04-02T22:15:10.025454
- 2026-04-02T22:15:19.639210

## 三、说明

- 完整明细已导出为 CSV：`run/order_log_export.csv`（或通过 `--csv` 指定路径）。
- DEMO 运行（`tiger1 d moe`）时：若 SDK 初始化成功则使用真实 API（openapicfg_dem），订单为 mode=real；若初始化失败则走模拟，订单为 mode=mock。
- 老虎后台请使用 **DEMO 账户** 对应账户与时间范围查询；实盘账户与 DEMO 账户订单分离。
