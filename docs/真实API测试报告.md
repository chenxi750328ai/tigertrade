# 真实API测试报告

**日期**: 2026-01-28  
**测试文件**: `tests/test_feature_order_execution_real_api.py`

## 一、测试执行情况

### 1.1 真实API初始化

✅ **成功初始化真实API（DEMO账户）**:
```
配置加载成功: account=<来自配置文件>, tiger_id=<来自配置文件>
使用account: <来自配置文件>
API初始化成功
   Quote API: RealQuoteApiAdapter
   Trade API: RealTradeApiAdapter
   Account: <来自 openapicfg_dem，勿提交>
   Mock模式: False
```

### 1.2 测试执行

✅ **测试真正运行了真实API**:
- 不再使用Mock
- 真正调用Tiger API
- 真正尝试下单和查询

## 二、测试结果

### 2.1 下单测试

❌ **下单失败**:
```
错误: code=1010 msg=biz param error(account 'xxx' is not authorized to the api user)
```

**原因**: account 没有在 Tiger 后台授权给 API 用户（tiger_id）

### 2.2 查询订单测试

❌ **查询订单失败**:
```
错误: code=1010 msg=biz param error(account 'xxx' is not authorized to the api user)
```

**原因**: 同样的授权问题

## 三、问题分析

### 3.1 授权问题

**错误信息**: `account 'xxx' is not authorized to the api user`

**可能的原因**:
1. account 配置不正确
2. API 用户（tiger_id）未在后台授权访问该 account
3. 需要在 Tiger 后台完成「账户 → API 用户」授权

### 3.2 测试验证

✅ **测试确实使用了真实API**:
- 不再使用Mock
- 真正调用Tiger API
- 真正尝试下单和查询
- 如果授权问题解决，测试应该能通过

## 四、解决方案

### 4.1 需要做的事情

1. **检查 Tiger 后台配置**:
   - 确认你使用的 account 是否已授权给当前 API 用户（tiger_id）
   - 若未授权，请在 Tiger 后台完成授权（参见 [后台看不到订单_必读_授权配置步骤](后台看不到订单_必读_授权配置步骤.md)）

2. **验证account配置**:
   - 确认 `openapicfg_dem/tiger_openapi_config.properties` 中的account配置是否正确
   - 确认account是否属于DEMO账户

3. **测试授权**:
   - 授权配置完成后，重新运行测试
   - 测试应该能够成功下单和查询订单

## 五、测试代码

### 5.1 测试文件

`tests/test_feature_order_execution_real_api.py`

### 5.2 关键特性

1. **真实API初始化**:
   ```python
   client_config = TigerOpenClientConfig(props_path='./openapicfg_dem')
   quote_client = QuoteClient(client_config)
   trade_client = TradeClient(client_config)
   api_manager.initialize_real_apis(quote_client, trade_client, account=account_to_use)
   ```

2. **真实下单**:
   ```python
   success, message = self.order_executor.execute_buy(...)
   ```

3. **真实查询**:
   ```python
   all_orders = trade_client.get_orders(account=account, symbol=symbol, limit=50)
   found_order = trade_client.get_order(account=account, order_id=order_id)
   ```

## 六、总结

### 6.1 测试状态

✅ **真实API测试已实现**:
- 不再使用Mock
- 真正调用Tiger API
- 真正尝试下单和查询

❌ **遇到授权问题**:
- account没有授权给API用户
- 需要在Tiger后台配置授权

### 6.2 下一步

1. **解决授权问题**:
   - 在Tiger后台配置account授权
   - 确认account配置正确

2. **重新运行测试**:
   - 授权问题解决后，重新运行测试
   - 测试应该能够成功下单和查询订单

3. **验证订单**:
   - 下单成功后，通过API查询订单
   - 验证订单在Tiger后台可见

---

**结论**: 真实API测试已经实现并运行，但是遇到了account授权问题。授权问题解决后，测试应该能够成功验证订单提交和查询功能。
