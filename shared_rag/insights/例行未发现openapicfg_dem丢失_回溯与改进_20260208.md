# 回溯：openapicfg_dem 丢失为何每日例行未发现

**日期**: 2026-02-08  
**问题**: `/home/cx/openapicfg_dem` 目录及 `tiger_openapi_config.properties` 丢失，20h DEMO 与真实 API 依赖该配置，但每日例行仍显示「通过」，未暴露问题。

---

## 一、为何例行「通过」了却没发现问题

### 1. CI/CD 测试（第 1 项）

- 例行命令：`pytest tests/ -m "not real_api"`。
- **所有依赖真实 API 的用例被排除**，不读 `openapicfg_dem`，不连 Tiger。
- 结果：无配置也能全绿，**无法发现配置丢失**。

### 2. 覆盖率统计（第 2 项）

- 同样用 `-m "not real_api"` 或等价，不触达真实 API。
- **无法发现配置丢失**。

### 3. 问题解决（第 3 项）

- 只处理「已暴露」的失败用例与 Bug。
- 没有自动化步骤检查「DEMO 配置是否存在」。
- **无法发现配置丢失**。

### 4. 20h DEMO 运行（第 4 项）

- 例行描述是「**查看或启动**状态」。
- 若执行时只做「查看」（看进程、看日志），而 DEMO 当前未在跑，则**不会执行「启动」**，也就不会调用 `TigerOpenClientConfig(props_path='./openapicfg_dem')`，不会读配置。
- 结果：配置丢了，只要没人在该次例行里真正去**启动** DEMO，就不会报错。
- **只有「启动 DEMO」时才会暴露配置缺失**；仅「查看状态」不会。

### 5. 数据刷新 / 收益率与算法优化 / 状态页（第 5–7 项）

- 数据与报告脚本多从本地日志、JSON、报告文件读数据，不强制依赖 `openapicfg_dem` 存在。
- 状态页只更新 HTML 并推送。
- **无法发现配置丢失**。

---

## 二、根因归纳

1. **CI 主动排除 real_api**，真实 API 与 DEMO 配置从未在 CI 路径被校验。
2. **20h DEMO 例行只要求「查看或启动」**，未强制「每次至少尝试一次启动或配置校验」；若只查看且 DEMO 未跑，就不会读配置，问题隐藏。
3. **没有「DEMO 配置存在且有效」的硬性前置检查**，例行通过条件里不包含「openapicfg_dem 存在且 account 可读」。

---

## 三、已做恢复与建议改进

### 已做

- 已重新创建 **`/home/cx/openapicfg_dem`**，并放入 **`tiger_openapi_config.properties`** 模板（占位符需你从 Tiger 后台替换为真实 tiger_id、private_key_path、account）。
- **`tigertrade/openapicfg_dem`** 符号链接指向 `/home/cx/openapicfg_dem`，链接已恢复有效。

### 建议改进（例行与清单）

1. **在「20h DEMO 运行」中增加前置检查**  
   - 执行「查看或启动」前，先检查：`openapicfg_dem/tiger_openapi_config.properties` 存在，且能解析出非空的 `account`（或至少存在且可读）。  
   - 若不通过：本次例行标记为异常并写入 RAG/状态，不得视为「通过」。

2. **在例行工作清单中写清**  
   - 「20h DEMO」项注明：若本机负责启动 DEMO，必须先通过「openapicfg_dem 存在且配置有效」检查，否则报错/告警，不得跳过。

3. **可选：CI 或每日流水增加轻量校验**  
   - 仅检查「配置文件存在且格式可读」（不读敏感内容），用于尽早发现配置被误删或路径错误。

---

*本文档存入 shared_rag/insights，供后续例行设计与复盘使用。*
