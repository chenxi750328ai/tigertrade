#!/usr/bin/env python3
"""
Tiger QA 辅助：检测本机 OpenClaw Gateway；可选通过 SDK 下发简短任务（需 pip install openclaw-sdk）。

用法：
  python scripts/openclaw_qa_helper.py              # 仅 HTTP 探活
  python scripts/openclaw_qa_helper.py --delegate   # 向 main agent 发一条 QA 摘要任务（需 SDK）

环境变量（与仓库根目录 juejin_publish_via_openclaw.py 一致）：
  OPENCLAW_GATEWAY_WS_URL  默认 ws://127.0.0.1:18789/gateway
  OPENCLAW_API_KEY / OPENCLAW_GATEWAY_TOKEN  默认 openclaw-local-dev
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

DEFAULT_HTTP = os.environ.get("OPENCLAW_GATEWAY_HTTP", "http://127.0.0.1:18789/")


def check_gateway_http(url: str = DEFAULT_HTTP) -> bool:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as r:
            return getattr(r, "status", 200) == 200
    except Exception as e:
        print(f"[openclaw_qa_helper] Gateway HTTP 不可达 ({url}): {e}", file=sys.stderr)
        return False


def build_delegate_prompt() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return f"""你是 Tiger 交易项目 QA 助手。请根据以下约束给出 3～5 条**可执行**的测试或验收建议（不要泛泛而谈）：

1. 项目目标包含：胜率、夏普、回撤、月化收益、20h 稳定性等（见仓库 README / docs/reports/QA_从项目目标把控质量）。
2. 仓库路径（若需读文件）：{root}
3. 优先关注：下单路径（api_adapter、order_executor）、order_log 实盘成功门禁、place_tiger_order 与回测一致性。

请直接输出编号列表即可。"""


async def _delegate_async() -> None:
    try:
        from openclaw_sdk import OpenClawClient
        from openclaw_sdk.core.config import ClientConfig
    except ImportError:
        print(
            "未安装 openclaw-sdk：pip install openclaw-sdk cryptography",
            file=sys.stderr,
        )
        sys.exit(2)

    ws = os.environ.get("OPENCLAW_GATEWAY_WS_URL", "ws://127.0.0.1:18789/gateway")
    token = os.environ.get("OPENCLAW_API_KEY") or os.environ.get(
        "OPENCLAW_GATEWAY_TOKEN", "openclaw-local-dev"
    )
    config = ClientConfig(gateway_ws_url=ws, api_key=token, timeout=120)
    prompt = build_delegate_prompt()
    conn = OpenClawClient.connect(**config.model_dump())
    async with (await conn) as client:
        agent = client.get_agent("main")
        try:
            stream = await agent.execute_stream(prompt)
            full_text = []
            async for event in stream:
                if isinstance(event, dict):
                    payload = event.get("payload") or event
                    if isinstance(payload, dict) and "data" in payload:
                        d = payload["data"]
                        if isinstance(d, dict) and "text" in d:
                            full_text.append(d.get("delta") or d["text"])
            if full_text:
                print("".join(full_text))
            else:
                result = await agent.execute(prompt)
                print(result.content if result else "无返回内容")
        except Exception as e:
            print(f"Stream 异常: {e}", file=sys.stderr)
            result = await agent.execute(prompt)
            print(result.content if result else "无返回内容")


def main() -> None:
    p = argparse.ArgumentParser(description="OpenClaw Gateway 探活与可选 QA 委派")
    p.add_argument(
        "--delegate",
        action="store_true",
        help="通过 WebSocket 向 main agent 下发一条 QA 摘要任务（需 openclaw-sdk）",
    )
    args = p.parse_args()

    if not check_gateway_http():
        sys.exit(1)
    print(f"[openclaw_qa_helper] Gateway OK: {DEFAULT_HTTP}")

    if args.delegate:
        import asyncio

        asyncio.run(_delegate_async())


if __name__ == "__main__":
    main()
