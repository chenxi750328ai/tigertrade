# API 被拒根因分析与修复方案

**日期**: 2026-02-24  
**问题**: 为何还有 API 被拒？状态页显示「失败（含 API 被拒）897 笔」

## 一、失败分类（order_log 统计）

| 错误类型 | 次数 | 是否真实 API 被拒 | 根因 |
|----------|------|-------------------|------|
| ALLOW_REAL_TRADING!=1 | 808 | **否** | 本地风控：RUN_ENV=production 且未设环境变量，**未发 API** |
| 持仓硬顶 pos>=3 | 17 | **否** | 本地风控：超仓拦截，**未发 API** |
| order_id 无效(来自 mock/测试) | 16 | **否** | 测试/mock 污染，非实盘 |
| **API Error 1010: account cannot be empty** | 16 | **是** | account 未正确传入 Order |
| **cannot import OrderSide** | 13 | **是** | tigeropen 版本/路径变更，导入失败 |
| **code=1200: Orders cannot be placed at this moment** | 10 | **是** | COMEX 非交易时段，老虎拒绝 |
| account 为空/Order 创建错误 | 10 | **是** | 同 1010 |
| Order failed | 7 | **是** | 需查具体原因 |

**结论**：897 条失败中，**约 808 条为本地风控拒绝**（未发 API），**约 89 条为真实 API 被拒**。当前统计口径「失败（含 API 被拒）」将两者混在一起，易误导。

## 二、修复方案

### 2.1 区分统计口径（立即实施）

- 在 `order_execution_status.json` 中增加 `real_api_reject_count`（真实 API 被拒）与 `local_reject_count`（本地风控拒绝）
- 状态页展示时区分：`失败：本地风控 N 笔，API 被拒 M 笔`

### 2.2 真实 API 被拒修复

| 错误 | 修复措施 |
|------|----------|
| **1010 / account 为空** | 确保 `api_manager.initialize_real_apis(account=...)` 在 tiger1 初始化时正确传入；api_adapter 已有 fallback，需验证 openapicfg_dem 解析 |
| **OrderSide 导入失败** | 在 tiger1/order_executor 中增加 `OrderSide` 的兼容 fallback（与 OrderType 一致） |
| **1200 非交易时段** | 在发单前增加 COMEX 交易时段检查；非时段直接本地拒绝并记录「非交易时段」，避免无谓 API 调用 |
| **Order failed** | 增强错误日志，记录完整 code/msg 便于后续排查 |

### 2.3 DEMO 启动保障

- `run_20h_demo.sh` 启动前设置 `ALLOW_REAL_TRADING=1`（虽然 sandbox 不检查，但保证一致性）
- 确保 `openapicfg_dem` 存在且 account 可解析

## 三、实施优先级

1. **P0**：区分统计口径，避免误导
2. **P1**：OrderSide 兼容导入、account 初始化验证
3. **P2**：COMEX 交易时段检查（减少 1200 无效调用）
