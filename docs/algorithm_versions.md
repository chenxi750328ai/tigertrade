# 算法版本说明

重大算法变更时在此区分版本，便于训练产出与报告对比。

---

## 当前版本

| 版本 | 说明 | 主要变更 |
|------|------|----------|
| **2.0** | 因果掩码 + Focal Loss | 见下 |
| 1.0 | 初始 | 见下 |

---

## 版本明细

### v2.0（当前）

- **MoE Transformer**
  - 自注意力启用**因果掩码**（`causal_mask=True`）：位置 t 仅可关注 j≤t，避免未来信息泄露。
  - 与局部窗口 `window_size` 同时使用时，仅可关注「过去」窗口内位置。
  - 动作分类损失改为 **Focal Loss**（`use_focal_loss=True`），缓解涨/跌/平类别不平衡。
- **数据**
  - `DataNormalizer.fit_transform_rolling(window=60)`：滚动窗口 Z-score 归一化，避免未来信息泄露。
- **参考**：策略设计文档、多头注意力建模金融数据（时序掩码、Focal Loss）。

### v1.0

- MoE/Transformer 自注意力无因果掩码（存在理论上的未来信息泄露风险）。
- MoE 动作损失为 LabelSmoothingCrossEntropy。
- 归一化为全局 fit_transform（或策略内逐样本均值/方差）。

---

## 如何对比版本

1. **训练产出**  
   训练脚本会按版本保存模型，例如：  
   - `best_moe_transformer_v1.0.pth`  
   - `best_moe_transformer_v2.0.pth`  
   同一数据与划分下，对比两版的验证准确率、收益头 MAE、回测收益/夏普即可。

2. **报告**  
   `algorithm_optimization_report.json` 与策略报告中会包含 `algorithm_version` 字段，标明本次运行使用的版本。

3. **切换版本**  
   修改 `src/algorithm_version.py` 中 `CURRENT_VERSION` 后重新训练即可产出该版本的 checkpoint；历史版本通过加载对应 `*_vX.X.pth` 做推理或回测对比。

---

## 版本递增约定

在以下情况考虑递增版本（并更新本文档与 `src/algorithm_version.py` 的 `VERSION_HISTORY`）：

- 注意力机制变更（如因果/非因果、稀疏、窗口）
- 损失函数变更（如 CE → Focal、权重方案）
- 归一化/预处理方式变更（如全局 → 滚动）
- 模型结构变更（层数、头数、MoE 专家数等对可比性影响较大的改动）
