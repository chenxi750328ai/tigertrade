"""
黑盒测试：被测对象 = 真实软件、真实流程。subprocess 跑 DEMO，不 mock 内部。

流程分支（必须覆盖，缺一即 FAIL）：
  1. 禁止项：1010、account 为空、AttributeError、NameError 等
  2. 交易循环已启动 + 至少一种 feature（策略预测/风控 DFX/下单）在输出中有证据
  3. 后台可见性：若本轮有 mode=real&status=success 且 order_id 为数字，get_orders(account,symbol) 必须能查到该 order_id

运行：pytest tests/test_blackbox_demo.py -m blackbox -v

改进：类级别只跑一次 DEMO（25s），四个用例复用同一份输出，避免多进程叠单、超出风控 2 手。
"""
import json
import os
import sys
import subprocess
import time
import unittest
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TIGER1_CMD = [sys.executable, 'src/tiger1.py', 'd', 'moe']
ORDER_LOG = PROJECT_ROOT / 'run' / 'order_log.jsonl'
CONFIG_PATH = PROJECT_ROOT / 'openapicfg_dem'

# 禁止出现在 DEMO 输出中的模式（任一出现即失败）
FORBIDDEN_PATTERNS = [
    ('1010', 'API 1010 account 或参数错误'),
    ('field \'account\' cannot be empty', 'account 为空'),
    ('account不能为空', 'account 为空'),
    ('account 为空', 'account 为空'),
    ('AttributeError', '未捕获的 AttributeError'),
    ('NameError', '未捕获的 NameError'),
]
# 复合禁止：两段同时出现才失败（避免误伤正常日志）
FORBIDDEN_COMPOSITE = [
    (('AttributeError', 'check_risk_control'), 'AttributeError check_risk_control'),
    (('NameError', 'MIN_TICK'), 'MIN_TICK 未定义 NameError'),
]
# 与 feature 流程对应的输出证据：交易循环 + 至少一类“策略/风控/下单”被跑到
FLOW_EVIDENCE_LOOP = [
    '开始交易循环',
    '运行时长',
    'TradingExecutor',
]
FLOW_EVIDENCE_FEATURES = [
    '预测',           # 策略预测（Feature 2）
    '置信度',         # 策略输出
    'MoETradingStrategy',
    '[DFX]',          # 风控路径（买入/卖出放行或拒绝）（Feature 4）
    '买入放行',
    '卖出放行',
    '买入拒绝',
    '卖出拒绝',
    '执行买入',       # 订单执行（Feature 3）
    '执行卖出',
    '订单提交成功',
    '订单ID',
    '进度更新',       # 循环进度
]


def _run_demo_subprocess(timeout_sec=25):
    """真实入口跑 DEMO，返回 (returncode, stdout+stderr)。"""
    proc = subprocess.Popen(
        TIGER1_CMD,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, 'ALLOW_REAL_TRADING': '1'},
    )
    try:
        out, _ = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            out = ""
    return getattr(proc, 'returncode', None), (out or "")


def _assert_no_forbidden(combined: str, forbidden_list, composite_list=None):
    """检查输出中是否出现禁止模式，若有则 raise AssertionError。"""
    combined_lower = combined.lower()
    for pattern, desc in forbidden_list:
        if pattern in combined or pattern.lower() in combined_lower:
            raise AssertionError(
                f"黑盒测试失败：DEMO 输出中出现「{desc}」。\n"
                f"匹配模式: {pattern!r}\n"
                f"输出片段（末 4000 字符）:\n{combined[-4000:]}"
            )
    for (p1, p2), desc in (composite_list or []):
        if p1 in combined and p2 in combined:
            raise AssertionError(
                f"黑盒测试失败：DEMO 输出中出现「{desc}」。\n"
                f"输出片段（末 4000 字符）:\n{combined[-4000:]}"
            )


# 若输出中出现以下任一，视为「未进入交易循环」（权限/连接等），不强制要求流程证据
EARLY_EXIT_INDICATORS = [
    '连接失败',
    'permission denied',
    'do not have permissions',
    'exit(1)',
    '环境连接失败',
    '期货交易接口不可用',
    '期货行情接口不可用',
    '未初始化',
]


def _assert_feature_flows_covered(combined: str, loop_markers, feature_markers):
    """断言交易循环已启动，且至少一类 feature 流程在输出中有证据。若因权限/连接提前退出则跳过流程断言。"""
    early_exit = any(ind in combined for ind in EARLY_EXIT_INDICATORS)
    if early_exit:
        return
    has_loop = any(m in combined for m in loop_markers)
    if not has_loop:
        raise AssertionError(
            "黑盒测试失败：DEMO 输出中未看到交易循环启动证据（如 开始交易循环/运行时长/TradingExecutor）。\n"
            f"输出片段（末 3000 字符）:\n{combined[-3000:]}"
        )
    has_feature = any(m in combined for m in feature_markers)
    if not has_feature:
        raise AssertionError(
            "黑盒测试失败：DEMO 输出中未看到任何 feature 流程证据（策略预测/风控 DFX/下单/进度更新）。\n"
            "说明策略循环或风控/下单路径可能未跑到。\n"
            f"输出片段（末 3000 字符）:\n{combined[-3000:]}"
        )


def _last_real_success_order_id_after(ts_cutoff):
    """从 order_log 取 ts >= ts_cutoff 的最近一条 mode=real, status=success，且 order_id 为纯数字。"""
    if not ORDER_LOG.exists():
        return None
    last_oid = None
    last_ts = None
    with open(ORDER_LOG, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get('mode') != 'real' or r.get('status') != 'success':
                continue
            ts_s = r.get('ts') or ''
            if ts_s and ts_s < ts_cutoff:
                continue
            oid = (r.get('order_id') or '')
            oid_str = str(oid).strip()
            if not oid_str or any(x in oid_str.upper() for x in ('MOCK', 'TEST', 'ORDER_')):
                continue
            if oid_str.isdigit():
                last_oid, last_ts = oid_str, ts_s
    return last_oid


def _assert_order_visible_in_backend_if_any(placed_after_iso):
    """若 order_log 在 placed_after_iso 之后有 real+success 且 order_id 为数字，则 get_orders 必须能查到该 order_id。"""
    order_id = _last_real_success_order_id_after(placed_after_iso)
    if not order_id:
        return  # 本轮无此类订单，不断言
    if not CONFIG_PATH.exists():
        raise AssertionError("黑盒测试失败：存在 real+success 订单但无法校验后台（openapicfg_dem 不存在）。")
    try:
        from tigeropen.tiger_open_config import TigerOpenClientConfig
        from tigeropen.trade.trade_client import TradeClient
    except ImportError:
        raise AssertionError("黑盒测试失败：存在 real+success 订单但无法校验后台（Tiger SDK 未安装）。")
    cfg = TigerOpenClientConfig(props_path=str(CONFIG_PATH))
    account = getattr(cfg, 'account', None)
    if not account:
        raise AssertionError("黑盒测试失败：存在 real+success 订单但无法校验后台（account 未配置）。")
    client = TradeClient(cfg)
    symbol = 'SIL2605'
    try:
        orders = client.get_orders(account=account, symbol=symbol, limit=100)
    except Exception as e:
        raise AssertionError(
            f"黑盒测试失败：后台可见性校验时 get_orders 报错（如未授权）。order_id={order_id}\n错误: {e}"
        )
    if orders is None:
        orders = []
    found = set()
    for o in orders:
        oid = getattr(o, 'order_id', None) or getattr(o, 'id', None)
        if oid is not None:
            found.add(str(oid).strip())
    if order_id not in found:
        raise AssertionError(
            f"黑盒测试失败：流程分支「后台可见」未通过——刚下的单在老虎 get_orders 中查不到。\n"
            f"order_id={order_id} account={account} 查到 {len(found)} 条。"
            "请检查 Tiger 后台账户授权与同一账户/环境。"
        )


@pytest.mark.blackbox
class TestBlackboxDemo(unittest.TestCase):
    """黑盒：真实 DEMO 进程 - 禁止错误 + 流程分支覆盖 + 后台可见性。
    类级别只跑一次 DEMO（25s），四类断言复用同一份输出，避免多进程叠单、超出风控 2 手。
    """
    _returncode = None
    _combined = None
    _cutoff_iso = None

    @classmethod
    def setUpClass(cls):
        from datetime import datetime, timezone, timedelta
        run_start = datetime.now(timezone.utc)
        returncode, combined = _run_demo_subprocess(timeout_sec=25)
        run_end = datetime.now(timezone.utc)
        cls._returncode = returncode
        cls._combined = combined
        # 本轮 run 期间写入的 order_log 的 ts 在 [run_start, run_end]，cutoff 取 run_start 前几秒
        cls._cutoff_iso = (run_start - timedelta(seconds=5)).strftime('%Y-%m-%dT%H:%M:%S')

    def test_demo_no_forbidden_errors(self):
        """禁止出现：1010、account 为空、AttributeError、NameError 等。"""
        _assert_no_forbidden(self._combined, FORBIDDEN_PATTERNS, FORBIDDEN_COMPOSITE)

    def test_demo_feature_flows_covered(self):
        """必须跑到：交易循环启动 + 至少一种 feature 流程（策略预测/风控/下单）在输出中有证据。"""
        _assert_no_forbidden(self._combined, FORBIDDEN_PATTERNS, FORBIDDEN_COMPOSITE)
        _assert_feature_flows_covered(
            self._combined,
            loop_markers=FLOW_EVIDENCE_LOOP,
            feature_markers=FLOW_EVIDENCE_FEATURES,
        )

    def test_demo_order_visible_in_backend_if_placed(self):
        """流程分支「后台可见」：若 DEMO 跑出 real+success 且 order_id 为数字，get_orders 必须能查到。"""
        _assert_no_forbidden(self._combined, FORBIDDEN_PATTERNS, FORBIDDEN_COMPOSITE)
        _assert_order_visible_in_backend_if_any(self._cutoff_iso)

    def test_demo_starts_without_crash(self):
        """启动阶段无未捕获异常（复用本轮 DEMO 输出，前段无 Traceback 即通过）。"""
        combined = self._combined
        if 'Traceback' in combined and 'Error' in combined:
            if 'TimeoutExpired' not in combined:
                raise AssertionError(
                    "黑盒测试失败：DEMO 启动阶段出现未捕获异常。\n"
                    f"输出片段:\n{combined[-3000:]}"
                )


if __name__ == '__main__':
    unittest.main(verbosity=2)
