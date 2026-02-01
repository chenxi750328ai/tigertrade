# 变更日志

## [2.1.1] - 2026-01-27

### 文档更新
- 📝 更新PROTOCOL.md：添加项目归属说明，明确基于agentfuture框架
- 📝 更新版本历史：记录文档整理变更
- 📝 创建项目整理方案文档（PROJECT_ORGANIZATION.md）
- 📝 创建根目录项目总览（README.md）

### 改进
- 🔄 明确PROTOCOL.md的项目归属和用途
- 🔄 统一文档格式，添加项目说明

---

## [2.0.0] - 2026-01-20

### 重大变更
- 🚀 系统完全通用化，支持任意期货标的（不再限于SIL2603）
- ✅ 修正模型数量：从6个更新为7个（添加ModelComparisonStrategy）
- 🧹 大规模文件整理：删除4个冗余文件，归档11个旧文件

### 新增
- ✨ 添加 --symbol 参数支持任意标的
- ✨ 自动生成基于标的的输出目录
- ✨ 新增系统扩展性测试
- ✨ 创建archive目录用于归档旧版本
- ✨ 添加第7个模型：模型对比策略（LSTM+Transformer）

### 改进
- 🔄 重命名所有文件，移除硬编码的sil2603命名
- 🔄 优化目录结构，分类更清晰
- 🔄 更新所有文档为通用版本
- 🔄 改进启动脚本，支持灵活配置

### 删除
- ❌ debug_train.py（测试脚本）
- ❌ simple_train_test.py（测试脚本）
- ❌ src/train_with_logging.py（被详细版替代）
- ❌ scripts/data_collection/data_collection_analyzer.py（重复）

### 归档
- 📦 3个旧训练脚本 → archive/old_training_scripts/
- 📦 3个旧采集脚本 → archive/old_collection_scripts/
- 📦 5个旧文档 → archive/

### 测试
- ✅ 所有测试通过（6/6）
- ✅ 7个模型全部验证可用
- ✅ 支持多标的训练验证通过

### 文档
- 📝 更新README为通用版本
- 📝 创建文件整理计划文档
- 📝 创建系统重构总结文档
- 📝 创建整理完成报告

---

## [1.0.0] - 2026-01-20 (早期版本)

### 初始功能
- 📥 SIL2603数据下载（支持>20000条）
- 🤖 6个模型训练（遗漏了第7个）
- 📊 测试评估和报告生成
- 🚀 一键启动脚本
- 📚 详细文档

### 限制
- ⚠️ 硬编码SIL2603，不支持其他标的
- ⚠️ 遗漏ModelComparisonStrategy模型
- ⚠️ 文件重复和混乱
