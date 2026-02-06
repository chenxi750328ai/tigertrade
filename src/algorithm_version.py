"""
算法版本标识 - 重大算法变更时递增，便于训练/报告对比。

使用方式：
- 训练保存：checkpoint 与文件名带版本，如 best_moe_transformer_v2.0.pth
- 报告：algorithm_optimization_report.json 写入 algorithm_version，便于对比
- 版本说明：见 docs/algorithm_versions.md
"""

# 当前算法版本（重大变更时递增，如因果掩码、Focal Loss、归一化方式等）
CURRENT_VERSION = "2.0"

# 各版本简要说明（详细见 docs/algorithm_versions.md）
VERSION_HISTORY = {
    "1.0": "初始：MoE/Transformer 无因果掩码，动作损失为 LabelSmoothingCrossEntropy",
    "2.0": "因果掩码(causal_mask=True)防未来泄露；MoE 动作损失改为 Focal Loss；DataNormalizer 支持滚动窗口归一化",
}


def get_current_version():
    return CURRENT_VERSION


def get_version_info(version=None):
    """返回某版本的说明，默认当前版本。"""
    v = version or CURRENT_VERSION
    return VERSION_HISTORY.get(v, "未知版本")
