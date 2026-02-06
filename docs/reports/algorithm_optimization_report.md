# 算法优化和收益率分析报告

生成时间: 2026-02-06T10:06:38.485998

**算法版本**: 2.0（重大变更见 [algorithm_versions.md](../algorithm_versions.md)）

## 效果数据来源（本次例行用了啥）

- 收益率/胜率：API 历史订单 暂无或未解析
- DEMO：多日志汇总（同次运行，四策略共用统计；订单成功、止损止盈等）
- 网格/BOLL：回测（data/processed/test.csv）产出最优参数与 return_pct/win_rate

## 优化后的参数

### grid

```json
{
  "rsi_buy": 30,
  "rsi_sell": 60,
  "ma_short": 5,
  "ma_long": 20,
  "use_or": false
}
```

### boll

```json
{
  "rsi_buy": 30,
  "rsi_sell": 60,
  "ma_short": 5,
  "ma_long": 20,
  "use_or": false
}
```

