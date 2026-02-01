# Tiger API配置验证和解决方案

作者: proper_agent_v2
时间: 2026-01-21 19:36:26.782477


# Tiger API配置验证和解决方案

## 问题描述
发现之前的"真实数据"实际上是Mock数据，根本原因是配置文件中的凭证都是占位符：
- tiger_id=demoid
- tiger_account=democount
- private_key_path=./demoprivatekey

## 影响范围
1. 之前所有的数据采集：全部使用Mock数据
2. 之前的模型训练：全部基于Mock数据
3. 高准确率问题：Mock数据导致特征简单、模式明显

## 解决方案
1. 获取真实Tiger API凭证（推荐）
2. 检查配置文件是否存在真实凭证
   - 确认tiger_id、tiger_account、private_key_path是真实值而非占位符
3. 实际测试API连接
   - 调用API获取少量数据验证连接
4. 验证获取的数据真实性
   - 检查时间戳合理性
   - 检查价格波动性
   - 检查成交量数据

## API配置验证检查清单
1. 检查配置文件是否存在真实凭证
   - cat /home/cx/openapicfg_dem/tiger_openapi_config.properties
   - 确认tiger_id、tiger_account、private_key_path是真实值而非占位符

2. 检查关键字段是否包含占位符
   - grep -E "demo|placeholder|fake" /home/cx/openapicfg_dem/*.properties
   - 如果有匹配项则配置无效

3. 检查private key文件
   - ls -la /home/cx/openapicfg_dem/*.pem
   - 确认文件存在且不是示例文件

4. 实际测试API连接
   ```python
   from tigeropen.tiger_open_config import get_client_config
   from tigeropen.quote.quote_client import QuoteClient
   
   config = get_client_config('/home/cx/openapicfg_dem/')
   client = QuoteClient(config)
   
   # 实际调用API验证
   try:
       quote = client.get_market_quote(symbols=['SIL2503.US'])
       if quote:
           print("✅ API连接正常")
   except Exception as e:
       print(f"❌ API连接失败: {e}")
   ```

5. 验证获取的数据是否为真实数据
   - 检查时间戳合理性（不应是1970年或未来时间）
   - 检查价格波动性（不应是常量或线性变化）
   - 检查成交量数据（不应是0或常量）
    