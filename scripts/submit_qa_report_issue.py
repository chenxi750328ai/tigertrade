#!/usr/bin/env python3
"""向 tigertrade 仓库提交 QA 评审报告 Issue，指派陈正霞（Tiger 项目 Leader）处理。"""
import os
import json
import urllib.request
from pathlib import Path

def load_env():
    for p in [Path("/home/cx/.env"), Path("/home/cx/agentfuture/.env"), Path("/home/cx/tigertrade/.env")]:
        if p.exists():
            for line in open(p):
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k in ("GITHUB_TOKEN", "GITHUB_PAT") and v:
                        os.environ.setdefault("GITHUB_TOKEN", v)
                        return
                    if k == "GITHUB_PAT" and v:
                        os.environ.setdefault("GITHUB_TOKEN", v)
                        return

def main():
    load_env()
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print("未设置 GITHUB_TOKEN，无法通过 API 提交 Issue。")
        print("请手动打开 https://github.com/chenxi750328ai/tigertrade/issues/new 提交。")
        return 1

    title = "[QA] Tiger 项目全面评审报告 — 请陈正霞处理"
    body = """**提交人**：陈正孤（项目 QA）  
**评审日期**：2026-02-08  
**报告位置**：`docs/reports/QA评审报告_Tiger项目_陈正孤.md`

---

## 摘要

对 Tiger 项目的**需求/设计文档、代码、测试用例、报告**进行了全面评审，并补充了**深层次问题**与**用户视角验收测试**：
- 主报告：`docs/reports/QA评审报告_Tiger项目_陈正孤.md`
- 深层次问题与用户验收：`docs/reports/QA评审报告_深层次问题与用户验收_陈正孤.md`

### 高优先级（建议本迭代处理）

| 编号 | 摘要 |
|------|------|
| **D1** | 回测与实盘逻辑不一致（grid/boll），回测参考价值受限 |
| **T1** | TC-F6-001（20h 运行）未自动化，需明确由稳定性流水线执行并检查结果 |

### 中优先级

D2（Feature 7 无测试覆盖）、D3（20h 与 CI 触发方式）、C1（TODO 跟踪）、T2/T3、R1 等见报告正文。

### 请陈正霞

1. 确认高优先级项的处理计划与责任人；
2. 将中/低优先级项纳入项目计划或 backlog；
3. 用户验收测试清单可作为发布前检查项。

完整内容请查看仓库内报告文档。"""

    url = "https://api.github.com/repos/chenxi750328ai/tigertrade/issues"
    data = json.dumps({"title": title, "body": body}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            out = json.loads(resp.read().decode())
            issue_url = out.get("html_url", "")
            issue_number = out.get("number", "")
            print(f"Issue 已提交: #{issue_number}")
            print(f"链接: {issue_url}")
            return 0
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"提交失败 HTTP {e.code}: {body[:500]}")
        return 1
    except Exception as e:
        print(f"提交失败: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
