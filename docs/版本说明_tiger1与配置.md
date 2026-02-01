# tiger1 版本与配置说明

## 最原始的 tiger1.py 去哪儿了

- **原来位置**：项目**根目录** `/home/cx/tigertrade/tiger1.py`（根目录下的 `tiger1.py`）。
- **后来**：项目重组时，主入口迁到 `src/tiger1.py`，根目录的 `tiger1.py` 在仓库里被删掉了（git 里是 `D tiger1.py`）。
- **恢复出来的“原始版”**：已从 git 提交 `5f13dd8` 取出，保存为：
  - **`archive/original_tiger1_from_root.py`**（973 行，即当时根目录那份能下单成功的版本）。

如需用“最原始”的那份逻辑，可直接看或跑 `archive/original_tiger1_from_root.py`。当前主入口仍是 `src/tiger1.py`。

## 账号和 KEY 配置（从未改过）

- **一直用的配置目录**：**`openapicfg_dem`**（项目里是 `./openapicfg_dem`，即 `/home/cx/tigertrade/openapicfg_dem/`；若你本机是 `/cx/openapicfg_dem`，也是同一套配置思路）。
- 根目录原始 `tiger1.py` 里就是：
  - `client_config = TigerOpenClientConfig(props_path='./openapicfg_dem')`（demo 模式）。
- 现在的 `src/tiger1.py` 在 demo 模式（参数 `d`）下同样是：
  - `TigerOpenClientConfig(props_path='./openapicfg_dem')`。
- **没有改过账号/KEY**：代码里一直读的都是 `openapicfg_dem` 下面的配置，可以下单成功的那套就是这份。

## 各版本对应关系（简要）

| 文件 | 说明 |
|------|------|
| **archive/original_tiger1_from_root.py** | 从 git 恢复的、根目录的原始 tiger1.py（用 openapicfg_dem，可下单成功的那版） |
| **src/tiger1.py** | 当前主入口，仍用 openapicfg_dem（demo） |
| **src/tiger1_legacy.py** | 后续整理的“旧版”逻辑备份 |
| **tiger1_v2.py** | 根目录下的 V2 模块化版本 |
| **archive/temp_reports/tiger1.py.backup** | 某次备份快照，也用的 openapicfg_dem |

总结：**最原始的 tiger1.py 就是根目录那份，已恢复到 `archive/original_tiger1_from_root.py`；账号和 KEY 一直用的是 openapicfg_dem 下的配置，从未改过，可以下单成功。**
