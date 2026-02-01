# tiger2 是什么

**tiger2** 是**另一个项目**里的模块，不在本仓库（tigertrade）里。

- **位置**：若存在，一般在 `/home/cx/tigertrade1` 项目中，主入口为 `tiger2.py`（或 `tiger2` 包）。
- **关系**：可理解为「tiger1 = 本仓库主程序，tiger2 = 另一仓库的优化版/实验版」；本仓库不包含、不依赖 tiger2。
- **测试**：`tests/test_optimized_strategy.py` 和部分旧测试会引用 tiger2；在未安装 tigertrade1/未提供 tiger2 时，这些用例会**被跳过**（`@unittest.skipIf(not HAS_TIGER2, ...)`），属预期行为。

无需在本仓库内安装或配置 tiger2；若你只有 tigertrade，直接忽略或跳过相关测试即可。
