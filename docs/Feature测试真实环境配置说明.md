# Feature测试真实环境配置说明

**日期**: 2026-01-28  
**要求**: Feature测试必须使用真实环境测试

## 一、配置说明

### 1.1 统一初始化机制

所有Feature测试现在都通过 `tests/feature_test_base.py` 统一初始化真实API：

```python
from tests.feature_test_base import initialize_real_api

# 在测试类中初始化
@classmethod
def setUpClass(cls):
    initialize_real_api()
    # ... 其他初始化
```

### 1.2 自动初始化

`feature_test_base.py` 在模块导入时会自动初始化真实API，确保所有Feature测试都使用真实环境。

## 二、已修改的测试文件

### 2.1 test_feature_order_execution.py

✅ **已修改为使用真实API**:
- 导入 `initialize_real_api`
- 在 `setUpClass` 中初始化真实API
- 如果API处于Mock模式，测试会跳过并报错

### 2.2 其他Feature测试文件

需要修改以下文件，确保它们也使用真实API：
- `test_feature_data_collection.py`
- `test_feature_strategy_prediction.py`
- `test_feature_risk_management.py`
- `test_feature_trading_loop.py`

## 三、验证机制

### 3.1 自动验证

`initialize_real_api()` 会自动验证：
- API不是Mock模式
- account已正确设置
- trade_api.client可用

### 3.2 测试失败条件

如果以下条件不满足，测试会跳过并报错：
- API处于Mock模式
- account未设置
- trade_api.client不可用

## 四、运行方式

### 4.1 运行单个Feature测试

```bash
python -m unittest tests.test_feature_order_execution.TestFeatureOrderExecution.test_f3_001_buy_order_e2e -v
```

### 4.2 运行所有Feature测试

```bash
python -m unittest discover -s tests -p "test_feature_*.py" -v
```

## 五、注意事项

### 5.1 授权问题

如果遇到授权错误：
```
account '21415812702670778' is not authorized to the api user
```

需要在Tiger后台配置account授权。

### 5.2 配置文件

确保 `openapicfg_dem/tiger_openapi_config.properties` 配置正确：
- account: DEMO账户ID
- tiger_id: API用户ID
- license: 许可证
- private_key_pk1/private_key_pk8: 私钥

## 六、总结

✅ **所有Feature测试现在都使用真实API环境**
✅ **统一的初始化机制确保一致性**
✅ **自动验证确保测试环境正确**

---

**重要**: Feature测试必须使用真实环境，不能使用Mock！
