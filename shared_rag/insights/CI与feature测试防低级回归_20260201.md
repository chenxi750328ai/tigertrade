# CI/CD 与 Feature 测试如何防「低级」回归

**日期**: 2026-02-01  
**类别**: insights  
**标签**: ci, regression, order_executor, __main__, check_risk_control

---

## 问题：为何「AttributeError check_risk_control」没被拦住？

- **现象**：20h DEMO 以 `python src/tiger1.py d moe` 启动时，出现 `AttributeError: module '__main__' has no attribute 'check_risk_control'`。
- **原因**：tiger1 作为 **脚本**（`__main__`）运行时，OrderExecutor 可能拿到的是 `__main__` 作为 risk_manager；在部分初始化顺序下，`__main__` 尚未定义 `check_risk_control`，导致属性查找失败。
- **为何 CI/Feature 没发现**：
  1. **单元/Feature 测试**里创建的是 `OrderExecutor(t1)`，其中 `t1` 是 **导入的** `src.tiger1` 模块，始终有 `check_risk_control`，从未覆盖「risk_manager 无 check_risk_control」的回退路径。
  2. **没有以真实入口启动**：没有在 CI 里用 `python src/tiger1.py d moe` 跑过哪怕几秒的冒烟，所以 __main__ 路径从未被触发。

---

## 已补的测试与 CI

1. **OrderExecutor 回退路径**（`tests/test_order_executor_comprehensive.py`）  
   - 用例：`test_order_executor_fallback_when_risk_manager_lacks_check_risk_control`  
   - 做法：传入一个**没有** `check_risk_control` 的 risk_manager（模拟 __main__ 尚未定义），创建 OrderExecutor，再调用 `execute_buy`。  
   - 断言：不抛 `AttributeError`，且 `_risk_fallback` 被设置（回退到 `src.tiger1`）。

2. **tiger1 __main__ 冒烟**（`tests/test_run_moe_demo_integration.py`）  
   - 用例：`test_tiger1_main_entry_no_attribute_error`  
   - 做法：subprocess 执行 `python src/tiger1.py d moe`，等待约 6 秒后终止，收集 stdout/stderr。  
   - 断言：输出中不得出现 `AttributeError` 与 `check_risk_control`（或其它 AttributeError）。

3. **CI 显式跑上述用例**（`.github/workflows/ci.yml`）  
   - 步骤名：`Regression guard (OrderExecutor 回退 + tiger1 __main__ 冒烟)`  
   - 单独跑这两个用例，失败即 CI 失败，便于定位。

---

## 小结

| 之前缺口 | 补丁 |
|----------|------|
| 只测 OrderExecutor(t1)，不测「无 check_risk_control」 | 单测：无 check_risk_control 的 risk_manager → 回退到 t1、不抛错 |
| 从未用 `python src/tiger1.py` 跑过 | 冒烟：subprocess 启动 tiger1 __main__，检查启动阶段无 AttributeError |
| CI 未显式覆盖该路径 | CI 增加 Regression guard 步骤，必跑上述两用例 |

后续改 OrderExecutor 或 tiger1 入口时，跑本地 `pytest tests/test_order_executor_comprehensive.py tests/test_run_moe_demo_integration.py` 以及 CI 即可拦住同类低级回归。
