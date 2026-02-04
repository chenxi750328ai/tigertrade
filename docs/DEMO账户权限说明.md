# DEMO账户权限说明

**日期**: 2026-01-29  
**重要**: DEMO账户权限信息

## 一、权限购买说明

**后台的权限是需要购买的**，DEMO用户已购买以下权限：

### 1.1 白银相关权限
- ✅ **白银期货（SIL）**：有权限
- 交易所：COMEX（纽约商品交易所）
- 合约代码格式：`SIL2603`, `SIL2503` 等

### 1.2 SG人民币相关权限
- ✅ **新加坡人民币期货**：有权限
- 交易所：SGX（新加坡交易所）
- 合约代码格式：待确认

## 二、当前代码使用的标的

### 2.1 白银期货（SIL）

**配置位置**: `/home/cx/tigertrade/src/tiger1.py`

```python
FUTURE_SYMBOL = "SIL.COMEX.202603"
```

**转换后格式**: `SIL2603`（用于API下单）

**状态**: ✅ **有权限**，可以使用

### 2.2 其他标的

**注意**: 如果使用其他标的（如原油CL、黄金GC等），需要确认是否有权限。

## 三、权限验证

### 3.1 下单前检查

1. **确认symbol是否有权限**：
   - ✅ SIL相关：有权限
   - ✅ SG人民币相关：有权限
   - ❓ 其他标的：需要确认

2. **授权检查**：
   - account必须授权给API用户（tiger_id）
   - 这是之前发现的授权失败问题的根本原因

### 3.2 常见错误

1. **授权失败**：
   ```
   account 'xxx' is not authorized to the api user（需在 Tiger 后台完成授权）
   ```
   - **原因**: account没有授权给API用户
   - **解决**: 需要在Tiger后台配置授权

2. **权限不足**：
   - **原因**: 标的没有购买权限
   - **解决**: 使用有权限的标的（SIL或SG人民币）

## 四、代码修改建议

### 4.1 确保使用有权限的标的

```python
# ✅ 正确：使用有权限的SIL期货
FUTURE_SYMBOL = "SIL.COMEX.202603"  # 转换为 SIL2603

# ❌ 错误：使用没有权限的标的（如原油CL）
# FUTURE_SYMBOL = "CL.COMEX.202603"  # 可能没有权限
```

### 4.2 下单前验证权限

```python
def check_symbol_permission(symbol):
    """检查symbol是否有权限"""
    # 有权限的标的列表
    authorized_symbols = [
        'SIL',  # 白银
        'SG',   # 新加坡人民币（待确认具体代码）
    ]
    
    # 检查symbol是否在有权限列表中
    for authorized in authorized_symbols:
        if symbol.startswith(authorized):
            return True
    return False
```

## 五、测试建议

### 5.1 使用有权限的标的测试

1. **白银期货（SIL）**：
   - Symbol: `SIL2603`
   - 状态: ✅ 有权限，可以使用

2. **SG人民币期货**：
   - Symbol: 待确认具体代码
   - 状态: ✅ 有权限，可以使用

### 5.2 避免使用没有权限的标的

- ❌ 原油（CL）：可能没有权限
- ❌ 黄金（GC）：可能没有权限
- ❌ 其他未购买的标的

## 六、相关文档

- **API文档**: https://quant.itigerup.com/openapi/zh/python/operation/trade/placeOrder.html
- **权限问题分析**: `/home/cx/tigertrade/docs/订单提交问题分析.md`
- **授权问题分析**: `/home/cx/tigertrade/docs/订单提交问题根本原因分析.md`

---

**重要提醒**：
1. ✅ DEMO账户有权限的标的：SIL（白银）、SG人民币
2. ⚠️ 使用其他标的前，需要确认是否有权限
3. ⚠️ account必须授权给API用户（tiger_id）
