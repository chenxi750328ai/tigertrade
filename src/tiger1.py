
import sys
import os
import time
import random
import json
import logging
import hmac
import hashlib
import math
from datetime import datetime, timedelta, date, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import talib
import traceback
from collections import deque
from dotenv import load_dotenv
import csv

# 从项目根 .env 加载 ALLOW_REAL_TRADING 等（避免只改 shell 未 export 导致综合账户路径全拒单）
_TIGER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_TIGER_ROOT, ".env"))

# Tiger Open API imports
from tigeropen.common.consts import Language, Market, BarPeriod, QuoteRight
from tigeropen.common.consts import OrderStatus, Currency, SecurityType
from tigeropen.common.util.contract_utils import stock_contract
from tigeropen.trade.trade_client import TradeClient

try:
    from .tigeropen_consts_compat import OrderType, OrderSide, TimeInForce
except ImportError:
    try:
        from src.tigeropen_consts_compat import OrderType, OrderSide, TimeInForce
    except ImportError:
        # 直接运行 python src/tiger1.py 时无包上下文：从 src/ 目录加载 compat
        _td = os.path.dirname(os.path.abspath(__file__))
        if _td not in sys.path:
            sys.path.insert(0, _td)
        from tigeropen_consts_compat import OrderType, OrderSide, TimeInForce

# 导入API适配器
try:
    from .api_adapter import api_manager
except ImportError:
    try:
        # 如果相对导入失败，尝试绝对导入
        from src.api_adapter import api_manager
    except ImportError:
        # 如果作为脚本直接运行，需要添加当前目录到sys.path
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from api_adapter import api_manager

# 导入策略模块
try:
    from .strategies import llm_strategy
    from .strategies import large_model_strategy
    from .strategies import huge_transformer_strategy
    from .strategies import model_comparison_strategy
    from .strategies import large_transformer_strategy
    from .strategies import enhanced_transformer_strategy
    from .strategies import rl_trading_strategy
except ImportError:
    try:
        from strategies import llm_strategy
        from strategies import large_model_strategy
        from strategies import huge_transformer_strategy
        from strategies import model_comparison_strategy
        from strategies import large_transformer_strategy
        from strategies import enhanced_transformer_strategy
        from strategies import rl_trading_strategy
    except ImportError:
        # 如果导入失败，打印警告但继续运行
        print("⚠️ 警告：无法导入策略模块，某些功能可能不可用")
        llm_strategy = None
        large_model_strategy = None
        huge_transformer_strategy = None
        model_comparison_strategy = None
        large_transformer_strategy = None
        enhanced_transformer_strategy = None
        rl_trading_strategy = None

try:
    from .strategies import data_driven_optimization
except ImportError:
    try:
        from strategies import data_driven_optimization
    except ImportError:
        print("⚠️ 警告：无法导入data_driven_optimization模块")
        data_driven_optimization = None

# 导入时段自适应策略模块
try:
    from . import order_log
except ImportError:
    try:
        from src import order_log
    except ImportError:
        order_log = None

# 合约最小变动价位（必须在 place_tiger_order / verify_api 之前定义，否则 MIN_TICK 未定义报错）
try:
    from src.config import TradingConfig as _TC
    FUTURE_TICK_SIZE = _TC.TICK_SIZE
except Exception:
    try:
        from config import TradingConfig as _TC
        FUTURE_TICK_SIZE = _TC.TICK_SIZE
    except Exception:
        FUTURE_TICK_SIZE = float(os.getenv("TICK_SIZE", "0.005"))
MIN_TICK = FUTURE_TICK_SIZE

try:
    from .strategies import time_period_strategy
    TIME_PERIOD_STRATEGY_AVAILABLE = True
except ImportError:
    try:
        from strategies import time_period_strategy
        TIME_PERIOD_STRATEGY_AVAILABLE = True
    except ImportError:
        print("⚠️ 警告：无法导入time_period_strategy模块，时段自适应功能将不可用")
        TIME_PERIOD_STRATEGY_AVAILABLE = False
        time_period_strategy = None

# 全局数据收集器
class DataCollector:
    """数据收集器"""
    
    def __init__(self, data_dir='/home/cx/trading_data'):
        from datetime import datetime
        today = datetime.now().strftime('%Y-%m-%d')
        # 按照规范创建日期文件夹结构
        self.data_dir = os.path.join(data_dir, today)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 文件名包含日期
        self.data_file = os.path.join(self.data_dir, f'trading_data_{today}.csv')
        
        self.fields = [
            'timestamp', 'price_current', 'grid_lower', 'grid_upper', 'atr', 
            'rsi_1m', 'rsi_5m', 'buffer', 'threshold', 'near_lower', 
            'rsi_ok', 'trend_check', 'rebound', 'vol_ok', 'final_decision',
            'take_profit_price', 'stop_loss_price', 'position_size', 'side',
            'deviation_percent', 'atr_multiplier', 'min_buffer_val', 'market_regime',
            'boll_upper', 'boll_mid', 'boll_lower'
        ]
        
        # 初始化CSV文件
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()
    
    def collect_data_point(self, **kwargs):
        """收集数据点"""
        # 获取当前时间戳
        current_timestamp = datetime.now().isoformat()
        data_point = {
            'timestamp': current_timestamp,
            **kwargs
        }
        with open(self.data_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(data_point)
        


# 创建全局数据收集器实例
data_collector = DataCollector()

# 初始化配置和客户端
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.common.util.signature_utils import read_private_key
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.trade.trade_client import TradeClient

# module logger
logger = logging.getLogger(__name__)
from tigeropen.trade.trade_client import TradeClient

# module logger
logger = logging.getLogger(__name__)

# 模块说明（中文）
# 本模块实现了一个简单的期货网格交易策略原型，包含：
# - 行情数据获取与时区/格式标准化（`get_kline_data`）
# - 技术指标计算（BOLL, ATR, RSI 等，`calculate_indicators`）
# - 风控检查（`check_risk_control`）
# - 下单封装（`place_tiger_order`、`place_take_profit_order`）
# - 几种策略实现：`grid_trading_strategy`, `grid_trading_strategy_pro1`, `boll1m_grid_strategy`
#
# 设计要点（中文总结）:
# - 在 import 时尽量保持轻量（避免在模块导入阶段触发真实网络/文件IO）
# - 在 sandbox 环境（模拟）下，失败的下单会被模拟为成功以方便回测/开发
# - 对于行情时间戳做了健壮的解析与时区转换（默认假定返回为 UTC）
# - 针对止盈单提交增加了对最小变动价位（tick size）的自动修正与重试逻辑
#
# 算法总体与实现细节（中文详解）
# 下面的内容给出策略核心算法的逐步说明，便于阅读与后续维护：
#
# 1) 目标与设计：
#    - 目标：基于 Bollinger Bands 与 ATR 的多层确认机制实现稳健的期货网格开仓/平仓逻辑，
#      兼顾成交概率与风控（止损/单笔亏损/当日亏损/仓位限制）。
#    - 设计原则：尽量保持运行时可控（sandbox 模式下模拟下单），并对第三方 SDK 的
#      多种返回格式（DataFrame/iterable/by-page）做兼容处理。
#
# 2) 核心数据流：
#    - 从行情端获取 1 分钟与 5 分钟 K 线（`get_kline_data`），做时区与时间单位归一化（UTC -> Asia/Shanghai），
#      并保证最少数据量阈值（`MIN_KLINES`）以避免空值、短期样本失真。
#    - 基于 5 分钟数据计算 Bollinger Bands 与 ATR（`calculate_indicators`），并基于 1 分钟数据
#      计算短周期 RSI 与成交量，用于入场/退出的即时判断。
#
# 3) 网格确定与动态调整：
#    - 使用 5 分钟 Boll 中轨/上轨/下轨作为基准网格边界（`grid_lower`, `grid_upper`），
#      并结合 ATR 做微调以考虑当前波动率。
#    - 在 BOLL 发散或 ATR 放大时，调整网格以减少频繁进出导致的滑点与手续费损耗。
#
# 4) 开仓逻辑（Buy 条件示例）：
#    - 基线：价格接近或低于 `grid_lower` 且 1 分钟 RSI 处于低位（不同趋势下阈值不同）；
#    - 额外允许条件（pro1）：短期 RSI 反转、价格/RSI 背离、或成交量突增之一可放宽入场；
#    - 最终进入前执行 `check_risk_control`：校验仓位上限、单笔可能损失（基于 ATR 与合约乘数）、当日亏损上限等。
#
# 5) 止盈/止损策略：
#    - 止损：基于 ATR 倍数并对低波动加设 ATR 下限，同时在 BOLL 下轨下方留出结构缓冲；
#    - 止盈：以 `grid_upper` 减去基于 ATR 的偏移量或至少一个最小 tick 设置目标价，提高可成交概率；
#    - 止盈单提交：若主单无法直接包含利润腿，会调用 `place_take_profit_order` 单独下 TP，
#      并具备对被拒绝（例如 tick-size 不匹配）时的自动向最近 tick 对齐并重试一次的容错逻辑。
#
# 6) 下单容错与模拟：
#    - 优先使用 SDK 的合约/下单帮助函数（若可用），否则构造 `SimpleNamespace` 来兼容 `trade_client.place_order` 的参数。
#    - 在 sandbox 环境中，下单失败会被模拟为成功（以便离线开发/回测），而 production 模式下若未开启
#      `ALLOW_REAL_TRADING=1` 则拒绝真实下单以避免误操作。
#
# 7) 可测试性与工程实践：
#    - 函数尽量保持副作用可控（例如通过模块级变量保存简要状态），并在测试中通过 monkeypatch 模拟
#      `quote_client` / `trade_client` 的行为来验证不同路径。
#    - 对时间戳解析、分页逻辑、以及不同 SDK 返回格式增加了兼容性代码与日志，便于排查线上差异。

# Read command-line mode when running as a script, but be import-safe for tests
count_type = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ('d', 'c') else 'd'

client_config = None
quote_client = None
trade_client = None

# 根因修复：使用绝对路径加载配置，避免 subprocess/cron 等场景下 cwd 非项目根导致 1010
def _abs_config_path(rel_name):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    return str(root / rel_name)

# Only try to instantiate real client objects when running with explicit args
if len(sys.argv) > 1:
    if count_type == 'd':
        try:
            client_config = TigerOpenClientConfig(props_path=_abs_config_path('openapicfg_dem'))
        except Exception:
            client_config = None
    elif count_type == 'c':
        try:
            client_config = TigerOpenClientConfig(props_path=_abs_config_path('openapicfg_com'))
        except Exception:
            client_config = None
    else:
        print(f"错误：不支持的参数 '{count_type}'，仅支持 d 或 c")
        # When running as a script we will exit later in main; do not sys.exit on import
        client_config = None

# 说明：
# - 本脚本通过命令行参数选择运行模式：'d' 表示 demo/sandbox，'c' 表示 production。
# - 在模块导入阶段不会主动触发实盘/网络操作；仅当明确传入参数时才尝试创建 SDK 客户端。
# - 这样在进行单元测试或作为库被导入时，不会因为缺少配置或网络导致导入失败。

# Try to build clients if we have a config; fail gracefully for import-time safety
if client_config is not None:
    try:
        quote_client = QuoteClient(client_config)  # 行情客户端
        trade_client = TradeClient(client_config)  # 交易客户端
        
        # 如果成功创建了真实客户端，初始化api_manager使用真实API
        # 与原始 tiger1 一致：account 直接来自 client_config（openapicfg_dem）
        if not hasattr(api_manager, '_account') or not api_manager._account:
            account_from_config = getattr(client_config, 'account', None) or (getattr(trade_client.config, 'account', None) if hasattr(trade_client, 'config') else None)
            api_manager.initialize_real_apis(quote_client, trade_client, account=account_from_config)
        # 如果account已设置，跳过重新初始化，避免覆盖
    except Exception:
        quote_client = None
        trade_client = None
        # 如果 SDK 初始化失败（例如缺少凭证/网络），保持 None 以便测试时注入模拟对象
        # 同时确保api_manager处于模拟模式
        api_manager.initialize_mock_apis()

# another method 
# def get_client_config():
#    client_config = TigerOpenClientConfig()
#    # 如果是windowns系统，路径字符串前需加 r 防止转义， 如 read_private_key(r'C:\Users\admin\tiger.pem')
#    client_config.private_key = read_private_key('填写私钥PEM文件的路径')
#    client_config.tiger_id = '替换为tigerid'
#    client_config.account = '替换为账户，建议使用模拟账户'
#    client_config.language = Language.zh_CN  #可选，不填默认为英语'
#    # client_config.timezone = 'US/Eastern' # 可选时区设置
#    return client_config
# 调用上方定义的函数生成用户配置ClientConfig对象
# client_config = get_client_config()

# 合约配置（当前 SIL2605：COMEX白银2026年5月期货，见 config/trading.json）
# 老虎证券期货合约格式：{品种}.{交易所}.{到期月}，需确认实际合约代码
# 交易标的：优先 config/trading.json（TradingConfig），其次环境变量
try:
    from src.config import TradingConfig
    FUTURE_SYMBOL = TradingConfig.SYMBOL
except Exception:
    try:
        from config import TradingConfig
        FUTURE_SYMBOL = TradingConfig.SYMBOL
    except Exception:
        FUTURE_SYMBOL = os.getenv("TRADING_SYMBOL", "SIL.COMEX.202605")
FUTURE_CURRENCY = Currency.USD
FUTURE_MULTIPLIER = 1000  # 白银期货每手1000盎司

# 网格策略核心参数（匹配之前讨论的规则）
GRID_MAX_POSITION = 2          # 最大持仓手数（默认）
DEMO_MAX_POSITION = 2          # DEMO/沙箱硬顶，不可被时段自适应覆盖；与 OrderExecutor HARD_MAX 一致，避免多进程/多轮测试叠单
GRID_ATR_PERIOD = 14           # ATR计算周期
GRID_BOLL_PERIOD = 20          # BOLL带周期
GRID_BOLL_STD = 2              # BOLL标准差
GRID_RSI_PERIOD_1M = 14        # 1分钟RSI周期
GRID_RSI_PERIOD_5M = 14        # 5分钟RSI周期

# 风控参数（6万美元账户适配，已优化放宽）
DAILY_LOSS_LIMIT = 2000         # 日亏损上限（美元，从1200放宽到2000）
# 合约时间限制：通知日/到期日风险与展期提示（可环境变量覆盖：NOTICE_DAYS, EXPIRY_BLOCK_DAYS）
try:
    NOTICE_DAYS = int(os.getenv("NOTICE_DAYS", "5"))
except (TypeError, ValueError):
    NOTICE_DAYS = 5
try:
    EXPIRY_BLOCK_DAYS = int(os.getenv("EXPIRY_BLOCK_DAYS", "2"))
except (TypeError, ValueError):
    EXPIRY_BLOCK_DAYS = 2
SINGLE_TRADE_LOSS = 3000        # 单笔最大亏损（美元，从1000放宽到3000）
STOP_LOSS_MULTIPLIER = 1.2     # 止损倍数（ATR）
STOP_LOSS_ATR_FLOOR = float(os.getenv('STOP_LOSS_ATR_FLOOR', 0.25))  # 低波动时的 ATR 下限，避免止损过近
STOP_LOSS_STRUCT_MULTIPLIER = float(os.getenv('STOP_LOSS_STRUCT_MULTIPLIER', 0.35))  # 相对下轨的结构缓冲（ATR 前数）
MIN_KLINES = 10                 # 最少K线条数阈值（用于get_kline_data）

# 网格周期参数
GRID_PERIOD = 20                # 网格计算所需的历史K线数量

# 新增：风控函数中使用的常量
STOP_LOSS_ATR_FACTOR = 2.0      # 止损ATR倍数因子
MAX_SINGLE_LOSS = 5000          # 单笔最大损失（从3000放宽到5000）
MAX_OPEN_ORDERS = 10            # 最大开放订单数量
ALLOW_REAL_TRADING = 0           # 是否允许真实交易（0为不允许，1为允许）

# 止盈参数（可通过命令行参数或环境变量调整）
TAKE_PROFIT_ATR_OFFSET = 0.2    # 止盈相对上轨的ATR余量比例（提高成交概率）
TAKE_PROFIT_MIN_OFFSET = 0.02   # 止盈最小绝对余量（价格单位）

# 行情判断阈值
BOLL_DIVERGENCE_THRESHOLD = 0.2  # BOLL发散阈值（轨道间距扩大≥20%）
ATR_AMPLIFICATION_THRESHOLD = 0.3 # ATR放大≥30%判定波动加剧

# 策略全局变量
current_position = 0           # 当前多头持仓手数（仅 long）
current_short_position = 0     # 当前空头持仓手数（卖出开仓导致，仅 short；期货 API 负数量表示空）
daily_loss = 0                 # 当日累计亏损
grid_upper = 0                 # 网格上轨
grid_lower = 0                 # 网格下轨
last_boll_width = 0            # 上一次BOLL轨道间距
atr_5m = 0                     # 5分钟ATR值
is_boll_divergence = False     # 是否BOLL发散

# 新增订单状态跟踪
open_orders = {}               # 记录待平仓的买单 {order_id: {'quantity': qty, 'price': price, 'timestamp': ts, 'tech_params': {}, 'reason': ''}}
closed_positions = {}          # 已平仓的交易记录 {order_id: {'buy_order_id': id, 'sell_order_id': id, 'buy_price': bp, 'sell_price': sp, 'analysis': {...}}, ...}


# 新增止盈相关全局变量
position_entry_times = {}      # 记录每个持仓的入场时间 {position_id: timestamp}
position_entry_prices = {}     # 记录每个持仓的入场价格 {position_id: entry_price}
active_take_profit_orders = {} # 跟踪已提交的止盈单 {position_id: {'target_price': price, 'submit_time': timestamp}}
# 最近一次买入成功时间（用于后台持仓同步延迟保护，防止短时回写导致超仓）
_last_buy_success_ts = 0.0

# 止盈参数（可通过命令行参数或环境变量调整）
TAKE_PROFIT_TIMEOUT = 15       # 止盈单超时（分钟）
MIN_PROFIT_RATIO = float(0.02) # 最低主动止盈比例（2%）

# 运行环境标识（用于日志/模拟下单提示），以及今日日期用于每日亏损重置
RUN_ENV = 'sandbox' if count_type == 'd' else 'production'
if RUN_ENV == 'production' and os.getenv('ALLOW_REAL_TRADING', '0') != '1':
    print("⚠️ 【配置】RUN_ENV=production（综合账户 c）且 ALLOW_REAL_TRADING!=1：place_tiger_order 将拒绝真实下单。")
    print("   修复：export ALLOW_REAL_TRADING=1 或在项目根 .env 写入 ALLOW_REAL_TRADING=1（见 .env.example）。")
    print("   DEMO 账户请用：python src/tiger1.py d ...（RUN_ENV=sandbox，不检查此项）。")
today = datetime.now().date()

# 初始化时段自适应策略（如果可用）
time_period_strategy_instance = None
if TIME_PERIOD_STRATEGY_AVAILABLE and time_period_strategy:
    try:
        time_period_strategy_instance = time_period_strategy.TimePeriodStrategy(
            symbol=FUTURE_SYMBOL,
            use_reference_rules=True
        )
        print("✅ 时段自适应策略已初始化")
    except Exception as e:
        print(f"⚠️ 时段自适应策略初始化失败: {e}，将使用默认网格参数")
        time_period_strategy_instance = None

# ====================== 核心工具函数 ======================
def get_timestamp():
    """生成API签名所需的时间戳"""
    return str(int(time.time() * 1000))  # 返回字符串而不是整数

def calculate_indicators(df_1m, df_5m):
    """
    # 完整检查：确保DataFrame有所需列
    required_cols = ["open", "high", "low", "close", "volume"]
    
    # 检查并修复1分钟数据
    if len(df_1m) == 0 or not all(col in df_1m.columns for col in required_cols):
        # 数据无效，返回默认值
        return {
            "1m": {"close": 0, "high": 0, "low": 0, "open": 0, "volume": 0, "rsi": 50, "atr": 0},
            "5m": {"close": 0, "high": 0, "low": 0, "open": 0, "volume": 0, "rsi": 50, "atr": 0,
                   "boll_upper": 0, "boll_lower": 0, "boll_middle": 0, "boll_mid": 0}
        }
    
    # 检查并修复5分钟数据
    if len(df_5m) == 0 or not all(col in df_5m.columns for col in required_cols):
        # 5分钟数据无效，使用1分钟数据替代
        latest_1m = df_1m.iloc[-1]
        return {
            "1m": {"close": latest_1m["close"], "high": latest_1m["high"], "low": latest_1m["low"],
                   "open": latest_1m["open"], "volume": latest_1m["volume"], "rsi": 50, "atr": 0},
            "5m": {"close": latest_1m["close"], "high": latest_1m["high"], "low": latest_1m["low"],
                   "open": latest_1m["open"], "volume": latest_1m["volume"], "rsi": 50, "atr": 0,
                   "boll_upper": latest_1m["close"] * 1.02, "boll_lower": latest_1m["close"] * 0.98,
                   "boll_middle": latest_1m["close"], "boll_mid": latest_1m["close"]}
        }
    
    计算技术指标
    :param df_1m: 1分钟K线数据
    :param df_5m: 5分钟K线数据
    :return: 包含技术指标的字典
    """
    if df_1m is None or df_5m is None:
        return {
            "1m": {"close": 0, "high": 0, "low": 0, "open": 0, "volume": 0, "rsi": 50, "atr": 0},
            "5m": {"close": 0, "high": 0, "low": 0, "open": 0, "volume": 0, "rsi": 50, "atr": 0,
                   "boll_upper": 0, "boll_lower": 0, "boll_middle": 0, "boll_mid": 0}
        }
    # 初始化返回结构
    indicators = {
        '1m': {},
        '5m': {}
    }

    # 为1分钟数据计算指标（缺列时用 close 或 0 回退，兼容测试/精简数据）
    if len(df_1m) > 0:
        latest_1m = df_1m.iloc[-1]
        close_1m = latest_1m.get('close', 0)
        indicators['1m']['close'] = close_1m
        indicators['1m']['high'] = latest_1m.get('high', close_1m)
        indicators['1m']['low'] = latest_1m.get('low', close_1m)
        indicators['1m']['open'] = latest_1m.get('open', close_1m)
        indicators['1m']['volume'] = latest_1m.get('volume', 0)

        # 计算1分钟RSI
        if len(df_1m) >= 15:
            delta = df_1m['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['1m']['rsi'] = rsi.iloc[-1] if len(rsi) > 0 else 50
        else:
            indicators['1m']['rsi'] = 50

    # 为5分钟数据计算指标
    if len(df_5m) > 0:
        latest_5m = df_5m.iloc[-1]
        indicators['5m']['close'] = latest_5m.get('close', 0)
        indicators['5m']['high'] = latest_5m.get('high', 0)
        indicators['5m']['low'] = latest_5m.get('low', 0)
        indicators['5m']['open'] = latest_5m.get('open', latest_5m.get('close', 0))
        indicators['5m']['volume'] = latest_5m.get('volume', 0)

        # 计算5分钟RSI
        if len(df_5m) >= 15 and 'close' in df_5m.columns:
            delta = df_5m['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['5m']['rsi'] = rsi.iloc[-1] if len(rsi) > 0 else 50
        else:
            indicators['5m']['rsi'] = 50

        # 计算BOLL指标 (使用20周期)
        if len(df_5m) >= 20 and 'close' in df_5m.columns:
            rolling_close = df_5m['close'].rolling(window=20)
            ma = rolling_close.mean()
            std = rolling_close.std()
            boll_upper = ma + 2 * std
            boll_lower = ma - 2 * std
            boll_middle = ma
            
            indicators['5m']['boll_upper'] = boll_upper.iloc[-1] if len(boll_upper) > 0 else latest_5m.get('close', 0)
            indicators['5m']['boll_lower'] = boll_lower.iloc[-1] if len(boll_lower) > 0 else latest_5m.get('close', 0)
            indicators['5m']['boll_middle'] = boll_middle.iloc[-1] if len(boll_middle) > 0 else latest_5m.get('close', 0)
            # alias expected by some tests
            indicators['5m']['boll_mid'] = indicators['5m']['boll_middle']
        else:
            # 如果数据不足，使用默认值
            indicators['5m']['boll_upper'] = latest_5m.get('close', 0) * 1.02
            indicators['5m']['boll_lower'] = latest_5m.get('close', 0) * 0.98
            indicators['5m']['boll_middle'] = latest_5m.get('close', 0)

        # 计算ATR指标
        if len(df_5m) >= 2 and 'high' in df_5m.columns and 'low' in df_5m.columns and 'close' in df_5m.columns:
            high_low = df_5m['high'] - df_5m['low']
            high_close = abs(df_5m['high'] - df_5m['close'].shift())
            low_close = abs(df_5m['low'] - df_5m['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1] if len(tr.rolling(window=14).mean()) >= 14 else 0
            indicators['5m']['atr'] = atr
        else:
            indicators['5m']['atr'] = 0

    return indicators


def judge_market_trend(indicators):
    """
    判断市场趋势
    :param indicators: 技术指标字典
    :return: 趋势类型 ('bullish', 'bearish', 'sideways')
    """
    # 获取5分钟数据的指标并返回测试套件期望的几个标签
    if '5m' in indicators and 'close' in indicators['5m']:
        # use boll_mid if available
        boll_middle = indicators['5m'].get('boll_middle') or indicators['5m'].get('boll_mid')
        current_price = indicators['5m']['close']
        rsi_5m = indicators['5m'].get('rsi', 50)

        if boll_middle is None or boll_middle == 0:
            return 'osc_normal'

        price_position = (current_price - boll_middle) / boll_middle

        # strong bull / bear
        if price_position > 0.02 and rsi_5m > 60:
            return 'bull_trend'
        if price_position < -0.02 and rsi_5m < 40:
            return 'bear_trend'

        # less decisive regimes
        if rsi_5m > 55:
            return 'osc_bull'
        if rsi_5m < 45:
            return 'osc_bear'

        return 'osc_normal'
    else:
        # 如果没有5分钟数据，默认为横盘
        return 'osc_normal'


def adjust_grid_interval(trend, indicators):
    """
    根据市场趋势调整网格间隔（整合时段自适应逻辑）
    :param trend: 市场趋势类型 ('bullish', 'bearish', 'sideways')
    :param indicators: 技术指标字典
    """
    global grid_upper, grid_lower, atr_5m, GRID_MAX_POSITION
    
    # 从指标中获取布林带值和当前价格
    if '5m' in indicators and 'boll_upper' in indicators['5m'] and 'boll_lower' in indicators['5m']:
        # 使用布林带的上下轨作为网格边界
        boll_upper = indicators['5m']['boll_upper']
        boll_lower = indicators['5m']['boll_lower']
        
        # 获取ATR值用于调整网格
        atr_value = indicators['5m'].get('atr', 0)
        current_price = indicators['5m'].get('close', indicators.get('1m', {}).get('close', 0))
        
        # 尝试使用时段自适应策略
        use_time_period_strategy = False
        if time_period_strategy_instance and current_price > 0:
            try:
                # 获取时段自适应网格参数（已经基于当前价格计算好了）
                grid_params = time_period_strategy_instance.get_grid_parameters(current_price)
                
                # 直接使用时段自适应策略返回的网格上下轨（已经基于当前价格计算）
                period_grid_upper = grid_params['grid_upper']
                period_grid_lower = grid_params['grid_lower']
                period_grid_step = grid_params['grid_step']
                
                # 确保网格区间合理（以当前价格为中心，但考虑布林带范围）
                # 如果时段自适应网格在布林带范围内，直接使用
                if period_grid_lower >= boll_lower and period_grid_upper <= boll_upper:
                    # 时段自适应网格在布林带范围内，直接使用
                    grid_upper = period_grid_upper
                    grid_lower = period_grid_lower
                elif period_grid_lower < boll_lower or period_grid_upper > boll_upper:
                    # 时段自适应网格超出布林带范围，需要调整
                    # 以当前价格为中心，使用时段自适应的间距，但限制在布林带范围内
                    grid_upper = min(period_grid_upper, boll_upper + period_grid_step)
                    grid_lower = max(period_grid_lower, boll_lower - period_grid_step)
                    
                    # 确保网格以当前价格为中心（如果可能）
                    grid_center = (grid_upper + grid_lower) / 2
                    if abs(grid_center - current_price) > period_grid_step:
                        # 如果网格中心偏离当前价格太多，重新以当前价格为中心计算
                        grid_upper = current_price + 2 * period_grid_step
                        grid_lower = current_price - 2 * period_grid_step
                        
                        # 但仍要确保在布林带合理范围内
                        if grid_upper > boll_upper * 1.1:  # 允许超出10%
                            grid_upper = boll_upper * 1.1
                        if grid_lower < boll_lower * 0.9:  # 允许超出10%
                            grid_lower = boll_lower * 0.9
                else:
                    # 默认情况：直接使用时段自适应网格
                    grid_upper = period_grid_upper
                    grid_lower = period_grid_lower
                
                # 更新最大仓位（时段自适应）；DEMO 不得被时段抬高，始终用 DEMO_MAX_POSITION
                raw_max = grid_params['max_position']
                GRID_MAX_POSITION = min(raw_max, DEMO_MAX_POSITION) if RUN_ENV == 'sandbox' else raw_max
                if RUN_ENV == 'sandbox' and raw_max > DEMO_MAX_POSITION:
                    logger.warning("DEMO 最大仓位已限制为 %s 手（时段配置为 %s 手）", DEMO_MAX_POSITION, raw_max)
                
                use_time_period_strategy = True
                period_name = grid_params['period_name']
                config_source = grid_params['config_source']
                
                print(f"📈 时段自适应网格 | 时段: {period_name} | 区间: [{grid_lower:.3f}, {grid_upper:.3f}], 仓位: {GRID_MAX_POSITION}手")
                
            except Exception as e:
                print(f"⚠️ 时段自适应策略获取失败: {e}，使用传统方法")
                use_time_period_strategy = False
        
        # 如果时段自适应策略不可用，使用传统方法
        if not use_time_period_strategy:
            # 根据趋势调整网格边界
            if trend == 'bullish':
                # 牛市中稍微扩大网格上轨
                grid_upper = boll_upper * (1 + 0.3 * (atr_value / boll_upper if boll_upper != 0 else 0))
                grid_lower = boll_lower * (1 - 0.05 * (atr_value / boll_lower if boll_lower != 0 else 0))
            elif trend == 'bearish':
                # 熊市中稍微缩小网格下轨
                grid_upper = boll_upper * (1 - 0.05 * (atr_value / boll_upper if boll_upper != 0 else 0))
                grid_lower = boll_lower * (1 - 0.1 * (atr_value / boll_lower if boll_lower != 0 else 0))
            else:
                # 横盘整理时使用布林带边界
                grid_upper = boll_upper
                grid_lower = boll_lower
            
            print(f"📈 传统网格参数 - 上轨: {grid_upper:.3f}, 下轨: {grid_lower:.3f}, ATR: {atr_value:.3f}")
        
        # 确保网格下轨不为0或负数
        if grid_lower <= 0:
            grid_lower = boll_lower if boll_lower > 0 else abs(boll_lower) + 0.01
        
        # 更新全局ATR值
        atr_5m = atr_value
        
    else:
        logger.debug("指标数据不足，使用默认网格参数")


def verify_api_connection():
    """本项目用期货，初始化校验必须包含期货：1) 期货交易接口（下单/查单）；2) 期货行情接口（brief+bars）。两项都可用才通过。"""
    if api_manager.is_mock_mode:
        print("🧪 运行在模拟模式下，跳过真实API连接验证")
        return True
    symbol = _to_api_identifier(FUTURE_SYMBOL)
    # 1) 期货交易接口必须可用
    trade_api = getattr(api_manager, "trade_api", None)
    if trade_api is None:
        print(f"❌ {count_type} 环境连接失败：trade_api 未初始化（请使用 python src/tiger1.py d 等方式启动）")
        return False
    try:
        trade_api.get_orders(limit=1)
    except Exception as e:
        print(f"❌ {count_type} 期货交易接口不可用：{e}")
        return False
    # 2) 期货行情接口必须可用（本项目用期货，校验必须包含期货）
    quote_api = getattr(api_manager, "quote_api", None)
    if quote_api is None:
        print(f"❌ {count_type} 环境连接失败：期货行情 quote_api 未初始化")
        return False
    # 优先用 get_future_brief 校验；若 permission denied 则降级到 get_future_bars
    # （DEMO 账户可能无 brief 权限但有 bars 权限，两者均可获取价格）
    try:
        quote_api.get_future_brief([symbol])
    except Exception as e:
        brief_err = str(e)
        if "permission denied" in brief_err.lower() or "4000" in brief_err:
            print(f"⚠️  {count_type} get_future_brief 无权限，降级用 get_future_bars 校验行情可用性")
        else:
            print(f"❌ {count_type} 期货行情接口不可用：{e}")
            return False
    try:
        quote_api.get_future_bars(
            [symbol],
            BarPeriod.ONE_MINUTE,
            -1,
            -1,
            2,
            None,
        )
        print(f"✅ {count_type} 期货行情 K 线接口可用")
    except Exception as e:
        print(f"❌ {count_type} 期货行情 K 线接口不可用：{e}")
        return False
    return True

# 说明：
# - `verify_api_connection` 主要用于手动/调试时快速验证 SDK 与网络连接是否正常，
#   会尝试调用行情与合约接口并打印返回样例。单元测试中一般会对 `quote_client` 做 Mock。

def get_future_brief_info(symbol):
    """获取期货简要信息（包括乘数、最小变动价位、到期日等）"""
    try:
        # 检查是否为模拟模式
        if api_manager.is_mock_mode:
            print("🧪 运行在模拟模式下，使用默认参数")
            return {
                "multiplier": FUTURE_MULTIPLIER,
                "min_tick": MIN_TICK,
                "expire_date": datetime.strptime(FUTURE_EXPIRE_DATE, "%Y-%m-%d").date() if FUTURE_EXPIRE_DATE != "2026-03-28" else date.today() + timedelta(days=90)
            }
        # 通过合约代码获取合约详情（此前误加提前 return 导致从未走 API，tick 一直用 0.01；COMEX 白银为 0.005）
        brief_info = api_manager.quote_api.get_future_brief([symbol])
        if not brief_info.empty and len(brief_info) > 0:
            row = brief_info.iloc[0]
            multiplier = getattr(row, "multiplier", FUTURE_MULTIPLIER)
            # API 可能返回 tick_size 或 min_tick
            min_tick = getattr(row, "tick_size", None) or getattr(row, "min_tick", None)
            if min_tick is None:
                min_tick = MIN_TICK
            else:
                min_tick = float(min_tick)
            
            expire_date_str = getattr(row, "expire_date", FUTURE_EXPIRE_DATE)
            expire_date = datetime.strptime(expire_date_str, "%Y-%m-%d").date() if expire_date_str != "2026-03-28" else date.today() + timedelta(days=90)
            return {
                "multiplier": multiplier,
                "min_tick": min_tick,
                "expire_date": expire_date
            }
        logger.debug("获取概要信息失败，使用默认参数")
        return {
            "multiplier": FUTURE_MULTIPLIER,
            "min_tick": MIN_TICK,
            "expire_date": datetime.strptime(FUTURE_EXPIRE_DATE, "%Y-%m-%d").date() if FUTURE_EXPIRE_DATE != "2026-03-28" else date.today() + timedelta(days=90)
        }
    except Exception as e:
        logger.debug("获取概要信息失败：%s，使用默认参数", e)
        return {
            "multiplier": FUTURE_MULTIPLIER,
            "min_tick": MIN_TICK,
            "expire_date": datetime.strptime(FUTURE_EXPIRE_DATE, "%Y-%m-%d").date() if FUTURE_EXPIRE_DATE != "2026-03-28" else date.today() + timedelta(days=90)
        }

def _to_api_identifier(symbol: str) -> str:
    """Convert known symbol patterns into the compact identifier expected by the
    quote by-page API.

    Examples:
      - 'SIL.COMEX.202603' -> 'SIL2603'
      - 'SIL2603' -> 'SIL2603' (unchanged)

    This is a best-effort helper to improve compatibility with different symbol
    naming conventions returned/used elsewhere in the codebase and SDK.
    """
    try:
        s = symbol.strip()
        # Already compact like SIL2603
        import re
        if re.match(r'^[A-Za-z]+\d{4}$', s):
            return s
        # Dotted format like 'SIL.COMEX.202603' -> base 'SIL', date '202603' -> 'SIL2603'
        if '.' in s:
            parts = s.split('.')
            base = parts[0]
            datepart = parts[-1]
            if len(datepart) == 6 and datepart.isdigit():
                year = datepart[:4]
                month = datepart[4:6]
                return f"{base}{year[-2:]}{month}"
        return s
    except Exception:
        return symbol

# 说明：
# - 一些场景下合约符号有多种表示法（例如 'SIL.COMEX.202603' vs 'SIL2603'），
#   本助手函数做尽可能的兼容性转换，优先返回 SDK/行情接口期望的紧凑表示法（如 'SIL2603'）。


def get_tick_data(symbol, count=100):
    """
    获取Tick级别的实时数据
    
    Parameters:
    - symbol: str or list-like of symbols
    - count: int, number of most-recent ticks to return
    
    Returns:
    - pandas.DataFrame with columns ['time', 'price', 'volume', 'side'] or empty DataFrame on error
    """
    try:
        # 检查是否为模拟模式
        if api_manager.is_mock_mode:
            # 在模拟模式下，生成模拟Tick数据
            now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8)))
            ticks = []
            base_price = 98.0  # 基准价格
            
            for i in range(count):
                tick_time = now - timedelta(seconds=i)
                # 模拟价格波动
                price = base_price + random.uniform(-0.1, 0.1)
                volume = random.randint(1, 10)
                side = random.choice(['BUY', 'SELL'])
                
                ticks.append({
                    'time': tick_time,
                    'price': price,
                    'volume': volume,
                    'side': side
                })
            
            if ticks:
                df = pd.DataFrame(ticks)
                df.set_index('time', inplace=True)
                return df
            return pd.DataFrame()
        else:
            # 实际API调用（如果API支持Tick数据）
            if 'quote_client' in globals() and quote_client is not None:
                try:
                    # 尝试使用Tiger API获取最新报价作为Tick数据
                    if isinstance(symbol, str):
                        symbol_list = [symbol]
                    else:
                        symbol_list = list(symbol)
                    
                    # 方法1: 尝试使用get_future_bars获取最新1条数据作为Tick
                    try:
                        latest_bars = quote_client.get_future_bars(
                            symbol_list,
                            BarPeriod.ONE_MINUTE,
                            -1,  # begin_time
                            -1,  # end_time
                            1,   # 只获取最新1条
                            None
                        )
                        if latest_bars is not None and not latest_bars.empty:
                            # 使用最新K线的收盘价作为Tick价格
                            latest_bar = latest_bars.iloc[-1]
                            now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8)))
                            tick_price = latest_bar.get('close', 0) if hasattr(latest_bar, 'get') else getattr(latest_bar, 'close', 0)
                            
                            ticks = [{
                                'time': now,
                                'price': tick_price,
                                'volume': latest_bar.get('volume', 0) if hasattr(latest_bar, 'get') else getattr(latest_bar, 'volume', 0),
                                'side': 'BUY'
                            }]
                            
                            df = pd.DataFrame(ticks)
                            df.set_index('time', inplace=True)
                            return df
                    except Exception as e1:
                        # 如果get_future_bars失败，尝试其他方法
                        pass
                    
                    # 方法2: 尝试使用get_future_brief获取最新报价
                    try:
                        brief_info = quote_client.get_future_brief(symbol_list)
                        if brief_info is not None and not brief_info.empty:
                            now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8)))
                            row = brief_info.iloc[0] if hasattr(brief_info, 'iloc') else brief_info
                            
                            # 尝试获取最新价格
                            tick_price = 0
                            for attr in ['last_price', 'close', 'price', 'latest_price']:
                                if hasattr(row, attr):
                                    tick_price = getattr(row, attr)
                                    break
                            
                            if tick_price > 0:
                                ticks = [{
                                    'time': now,
                                    'price': tick_price,
                                    'volume': 0,
                                    'side': 'BUY'
                                }]
                                df = pd.DataFrame(ticks)
                                df.set_index('time', inplace=True)
                                return df
                    except Exception as e2:
                        pass
                        
                except Exception as e:
                    # 所有方法都失败，使用模拟数据
                    pass
            
            # 如果无法获取Tick数据，使用模拟数据（基于最新K线价格）
            try:
                # 获取最新K线数据作为Tick数据的基准
                latest_kline = get_kline_data(symbol, '1min', count=1)
                if not latest_kline.empty:
                    base_price = latest_kline.iloc[-1]['close']
                else:
                    base_price = 98.0  # 默认价格
            except:
                base_price = 98.0
            
            # 生成模拟Tick数据
            now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8)))
            ticks = []
            for i in range(min(count, 10)):  # 限制数量
                tick_time = now - timedelta(seconds=i)
                price = base_price + random.uniform(-0.05, 0.05)  # 小幅波动
                volume = random.randint(1, 5)
                side = random.choice(['BUY', 'SELL'])
                
                ticks.append({
                    'time': tick_time,
                    'price': price,
                    'volume': volume,
                    'side': side
                })
            
            if ticks:
                df = pd.DataFrame(ticks)
                df.set_index('time', inplace=True)
                return df
            
            return pd.DataFrame()
    except Exception as e:
        logger.debug("获取Tick数据异常: %s", e)
        return pd.DataFrame()


def _make_synthetic_klines(count, base_price=90.0):
    """生成合成K线数据，用于 mock/异常 时保证策略有数据可跑。返回 index=时间(Asia/Shanghai)，列 open/high/low/close/volume。"""
    n = max(count, MIN_KLINES)
    price_changes = np.random.normal(0, 0.005, n)
    prices = base_price * (1 + price_changes).cumprod()
    opens = prices
    closes = prices * (1 + np.random.normal(0, 0.002, n))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.001, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.001, n)))
    volumes = np.random.randint(80, 120, n)
    idx = pd.date_range(start=datetime.now(timezone.utc), periods=n, freq='1min')
    idx = idx.tz_convert('Asia/Shanghai')
    return pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': volumes
    }, index=idx)


def load_klines_from_file(filepath: str, period: str, count: Optional[int] = None) -> pd.DataFrame:
    """从本地 CSV 加载 K 线，返回与 get_kline_data 相同格式的 DataFrame（index=time Asia/Shanghai，列 open/high/low/close/volume）。
    实盘与回测仅数据源不同，此处为「文件」分支；API 分支在 get_kline_data 内。
    CSV 需含列：time 或 timestamp 或 date（可选），open/high/low/close，volume（可选）。若仅有 close，则 open=high=low=close，volume=0。"""
    if not filepath or not os.path.isfile(filepath):
        return pd.DataFrame()
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return df
        # 时间列
        time_col = None
        for c in ('time', 'timestamp', 'date', 'datetime'):
            if c in df.columns:
                time_col = c
                break
        if time_col:
            df['time'] = pd.to_datetime(df[time_col], errors='coerce')
        else:
            df['time'] = pd.date_range(start='2020-01-01', periods=len(df), freq='1min' if period in ('1min', '1m') else '5min')
        df = df.dropna(subset=['time'])
        if df.empty:
            return pd.DataFrame()
        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize('UTC', ambiguous='infer')
        df['time'] = df['time'].dt.tz_convert('Asia/Shanghai')
        # OHLCV：缺列时用 close 补齐
        if 'close' not in df.columns:
            return pd.DataFrame()
        for col in ('open', 'high', 'low'):
            if col not in df.columns:
                df[col] = df['close']
        if 'volume' not in df.columns:
            df['volume'] = 0
        out = df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
        out = out.set_index('time').sort_index()
        if count is not None:
            out = out.tail(int(count))
        return out
    except Exception as e:
        logger.exception("load_klines_from_file %s: %s", filepath, e)
        return pd.DataFrame()


def load_ticks_from_file(filepath: str, count: Optional[int] = None) -> pd.DataFrame:
    """从本地 CSV 加载 Tick，返回 DataFrame（index=time Asia/Shanghai，列至少含 price）。
    训练有 tick 回测也应有；与实盘 get_tick_data 同粒度，仅数据源为文件。"""
    if not filepath or not os.path.isfile(filepath):
        return pd.DataFrame()
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return df
        time_col = None
        for c in ('time', 'timestamp', 'date', 'datetime'):
            if c in df.columns:
                time_col = c
                break
        if not time_col:
            return pd.DataFrame()
        df['time'] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=['time'])
        if df.empty or 'price' not in df.columns:
            return pd.DataFrame()
        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize('UTC', ambiguous='infer')
        df['time'] = df['time'].dt.tz_convert('Asia/Shanghai')
        out = df[['time', 'price']].copy()
        if 'volume' in df.columns:
            out['volume'] = df['volume']
        out = out.set_index('time').sort_index()
        if count is not None:
            out = out.tail(int(count))
        return out
    except Exception as e:
        logger.exception("load_ticks_from_file %s: %s", filepath, e)
        return pd.DataFrame()


def get_kline_data(symbol, period, count=100, start_time=None, end_time=None, from_file: Optional[str] = None):
    """Fetch K-line data (candles) and normalize to a pandas.DataFrame.
    数据源二选一：from_file 为路径时从文件加载（回测），否则从 API 获取（实盘）。其余处理与算法一致。
    Supports optional `start_time` and `end_time` (both `datetime` or epoch ms) and
    best-effort automatic paging using `QuoteClient.get_future_bars_by_page` for
    single-symbol time-range or large requests.

    Parameters
    - symbol: str or list-like of symbols
    - period: str one of {'1min','5min','1h','1d'}
    - count: int, number of most-recent bars to return
    - start_time, end_time: optional datetime or epoch ms (milliseconds since epoch)

    Returns
    - pandas.DataFrame indexed by timezone-aware `time` (Asia/Shanghai) with
      columns ['open','high','low','close','volume'] or an empty DataFrame on error.
    """
        # 中文说明：
        # 该函数从 `quote_client` 获取期货 K 线数据，并保证返回一个按北京时间（Asia/Shanghai）
        # 的 pandas.DataFrame，列为 ['open','high','low','close','volume']，索引为时间序列。
        # 兼容性要点：
        # - 支持传入单个合约或合约列表；当请求为单合约且需要大范围/时间段时尝试使用按页 API
        # - 能接受 pandas.DataFrame（含 time 列）或可迭代的 bar 对象（具有 .time/.open/.close 等属性）
        # - 对数字时间戳会尝试自动判断单位（s/ms/us/ns），并在 tz-naive 时默认视为 UTC
        # - 当获取到的数据少于 MIN_KLINES（默认10）时，会返回空 DataFrame，便于上层判定数据不足
    if from_file is not None:
        return load_klines_from_file(from_file, period, count)
    period_map = {
        "1min": BarPeriod.ONE_MINUTE,
        "3min": BarPeriod.THREE_MINUTES,
        "5min": BarPeriod.FIVE_MINUTES,
        "10min": BarPeriod.TEN_MINUTES,
        "15min": BarPeriod.FIFTEEN_MINUTES,
        "30min": BarPeriod.HALF_HOUR,
        "45min": BarPeriod.FORTY_FIVE_MINUTES,
        "1h": BarPeriod.ONE_HOUR,
        "2h": BarPeriod.TWO_HOURS,
        "3h": BarPeriod.THREE_HOURS,
        "4h": BarPeriod.FOUR_HOURS,
        "6h": BarPeriod.SIX_HOURS,
        "1d": BarPeriod.DAY,
        "1w": BarPeriod.WEEK,
        "1M": BarPeriod.MONTH,
        "1y": BarPeriod.YEAR,
    }
    if period not in period_map:
        logger.warning("不支持的周期：%s，使用合成数据兜底", period)
        return _make_synthetic_klines(count)
    
    try:
        # 检查是否为模拟模式
        if api_manager.is_mock_mode:
            # 在模拟模式下，使用模拟API
            klines = api_manager.quote_api.get_future_bars(
                symbol, 
                period, 
                start_time, 
                end_time, 
                count, 
                None
            )
            
            if klines is None or (hasattr(klines, 'empty') and klines.empty):
                return _make_synthetic_klines(count)
            if isinstance(klines, dict) and 'df' in klines:
                klines = klines['df']
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if isinstance(klines, pd.DataFrame):
                klines = klines.copy()
                for col in required_cols:
                    if col not in klines.columns:
                        if col == 'close' and 'open' in klines.columns:
                            klines[col] = klines['open']
                        elif col in ('low', 'high') and 'close' in klines.columns:
                            klines[col] = klines['close'] * (0.99 if col == 'low' else 1.01)
                        elif col == 'volume':
                            klines[col] = 100
                        else:
                            klines[col] = 90.0
                if len(klines) < MIN_KLINES:
                    return _make_synthetic_klines(count)
                if 'time' in klines.columns:
                    klines['time'] = pd.to_datetime(klines['time'], errors='coerce')
                    klines = klines.dropna(subset=['time'])
                    if klines.empty:
                        return _make_synthetic_klines(count)
                    if klines['time'].dt.tz is None:
                        klines['time'] = klines['time'].dt.tz_localize('UTC', ambiguous='infer')
                    klines['time'] = klines['time'].dt.tz_convert('Asia/Shanghai')
                    klines = klines.set_index('time')
                for c in list(klines.columns):
                    if c not in required_cols:
                        klines = klines.drop(columns=[c], errors='ignore')
                return klines.tail(max(count, MIN_KLINES))
            # 模拟模式下 API 可能返回 bar 对象列表，转为 DataFrame
            if hasattr(klines, '__iter__') and not isinstance(klines, (str, dict)):
                try:
                    klines_list = list(klines)
                    if klines_list and hasattr(klines_list[0], 'open'):
                        df = pd.DataFrame([{
                            'time': getattr(b, 'time', None),
                            'open': getattr(b, 'open', None),
                            'high': getattr(b, 'high', None),
                            'low': getattr(b, 'low', None),
                            'close': getattr(b, 'close', None),
                            'volume': getattr(b, 'volume', None)
                        } for b in klines_list])
                        if not df.empty and 'time' in df.columns:
                            df['time'] = pd.to_datetime(df['time'], errors='coerce')
                            if df['time'].dt.tz is None:
                                df['time'] = df['time'].dt.tz_localize('UTC')
                            df['time'] = df['time'].dt.tz_convert('Asia/Shanghai')
                            df = df.set_index('time')
                            if len(df) >= MIN_KLINES and all(c in df.columns for c in required_cols):
                                return df
                        if len(df) < MIN_KLINES:
                            return _make_synthetic_klines(count)
                except Exception:
                    pass
            return _make_synthetic_klines(count)
        else:
            # 实际API调用
            # 1. 统一 symbol 为 Tiger 期望的 compact 格式（如 SIL2603），SIL.COMEX.202603 需转换
            sym_list = [symbol] if isinstance(symbol, str) else list(symbol)
            identifier = _to_api_identifier(sym_list[0]) if sym_list else _to_api_identifier(FUTURE_SYMBOL)
            symbol_for_api = [identifier]
            # 2. 周末/休市时：若未指定时间，用上一交易日收盘作为 end，否则 API 可能返回空
            _end = end_time
            _start = start_time
            if _end is None and _start is None:
                now_utc = datetime.now(timezone.utc)
                weekday = now_utc.weekday()  # 0=Mon, 5=Sat, 6=Sun
                if weekday >= 5:  # 周六或周日，COMEX 休市，end 用上周五 17:00 ET ≈ 22:00 UTC
                    days_back = 1 if weekday == 5 else 2
                    _end = now_utc - timedelta(days=days_back)
                    _end = _end.replace(hour=22, minute=0, second=0, microsecond=0)
                    _start = _end - timedelta(hours=48)  # 往前 2 天确保有数据
                    logger.debug("周末请求K线，使用上一交易日 end=%s", _end)
            # Check if quote_client exists, otherwise try to initialize it
            if 'quote_client' not in globals() or quote_client is None:
                # Use the api_manager's quote_api as fallback
                klines = api_manager.quote_api.get_future_bars(
                    symbol_for_api,
                    period,
                    _start,
                    _end,
                    count,
                    None
                )
            else:
                now_utc = datetime.now(timezone.utc)
                weekday = now_utc.weekday()
                if weekday >= 5:  # 周末，用上一交易日收盘
                    days_back = 1 if weekday == 5 else 2
                    end_time = now_utc - timedelta(days=days_back)
                    end_time = end_time.replace(hour=22, minute=0, second=0, microsecond=0)
                    start_time = end_time - timedelta(hours=48)
                else:
                    end_time = now_utc
                    start_time = end_time - timedelta(hours=4) if period == "5min" else end_time - timedelta(hours=1)
                # 统一 symbol 为 compact 格式（SIL2603）
                sym_raw = symbol if isinstance(symbol, str) else (symbol[0] if symbol else FUTURE_SYMBOL)
                symbol1 = [_to_api_identifier(sym_raw)]
                logger.debug("get_kline_data request: symbol=%s period=%s count=%s start_time=%s end_time=%s", symbol1, period, count, start_time, end_time)

                # Convert optional start/end into epoch ms (UTC). Accept datetime (tz-aware or naive) or integer ms
                def _to_epoch_ms(t):
                    if t is None:
                        return None
                    if isinstance(t, (int, float)):
                        return int(t)
                    if isinstance(t, datetime):
                        # assume naive datetimes are UTC
                        if t.tzinfo is None:
                            t = t.replace(tzinfo=timezone.utc)
                        return int(t.astimezone(timezone.utc).timestamp() * 1000)
                    raise ValueError('start_time/end_time must be datetime or epoch ms')

                start_ms = _to_epoch_ms(start_time) if 'start_time' in locals() or 'start_time' in globals() else None
                end_ms = _to_epoch_ms(end_time) if 'end_time' in locals() or 'end_time' in globals() else None

                # If a time range or a large count is requested and we have a single symbol, try the paged API
                # 如果请求大于 SDK 单次返回上限，或用户显式提供时间范围，则尝试使用按页 API 获取历史数据
                use_paging = (start_ms is not None or end_ms is not None or count > 1000) and len(symbol1) == 1 and hasattr(quote_client, 'get_future_bars_by_page')

                if use_paging:
                    # fetch pages until done or we've collected `count` rows
                    all_pages = []
                    next_token = None
                    fetched = 0
                    while True:
                        # 说明：按页获取时我们需要处理多种 SDK 返回格式（DataFrame/tuple/dict/iterable）并
                        # 尽力提取 `next_page_token` 以持续分页，直到收集到足够的行或没有下一页为止。
                        try:
                            # API may accept (identifier, period, begin_time, end_time, total, page_size, time_interval)
                            identifier_for_api = _to_api_identifier(symbol1[0])
                            logger.debug("using identifier_for_api=%s for by-page call", identifier_for_api)
                            # prefer identifier string for by-page fetch
                            res = quote_client.get_future_bars_by_page(
                                identifier_for_api,
                                period_map[period],
                                start_ms if start_ms is not None else -1,
                                end_ms if end_ms is not None else -1,
                                count,
                                min(1000, max(100, count)),
                                2)
                        except TypeError:
                            # fall back to a simpler signature if needed
                            identifier_for_api = _to_api_identifier(symbol1[0])
                            res = quote_client.get_future_bars_by_page(identifier_for_api, period_map[period], start_ms or -1, end_ms or -1, count)

                        df_page = None
                        next_token = None
                        if isinstance(res, tuple) and len(res) == 2:
                            df_page, next_token = res
                        elif isinstance(res, dict):
                            df_page = res.get('df') or res.get('data') or pd.DataFrame(res)
                            next_token = res.get('next_page_token')
                        else:
                            df_page = res

                        token_from_column = False
                        if isinstance(df_page, pd.DataFrame):
                            # If the SDK returns next_page_token as a column, prefer that
                            if 'next_page_token' in df_page.columns:
                                # extract last non-null token
                                non_null = df_page['next_page_token'].dropna()
                                next_token = non_null.iloc[-1] if len(non_null) > 0 else None
                                # drop the token column from data we keep
                                df_page = df_page.drop(columns=['next_page_token'])
                                token_from_column = True
                            all_pages.append(df_page)
                            fetched += len(df_page)
                        else:
                            # If the page returned an iterable of bars, convert to DataFrame
                            try:
                                df_page = pd.DataFrame([{
                                    'time': getattr(bar, 'time', None),
                                    'open': getattr(bar, 'open', None),
                                    'high': getattr(bar, 'high', None),
                                    'low': getattr(bar, 'low', None),
                                    'close': getattr(bar, 'close', None),
                                    'volume': getattr(bar, 'volume', None)
                                } for bar in df_page])
                                all_pages.append(df_page)
                                fetched += len(df_page)
                            except Exception:
                                # give up if we cannot interpret page
                                break

                        if not next_token or fetched >= count:
                            break

                        # 否则继续循环并尽量传递 page token（不同 SDK 在参数签名上存在差异，需要兼容）
                        try:
                            logger.debug("paging: token=%s fetched=%s target=%s token_from_column=%s", next_token, fetched, count, token_from_column)
                            if token_from_column:
                                # When token came from a DataFrame column, prefer the simpler get_future_bars that accepts page_token
                                try:
                                    res = quote_client.get_future_bars(symbol1, period_map[period], -1, -1, count, next_token)
                                except Exception:
                                    # fall back to by-page with token if direct call fails
                                    logger.debug("get_future_bars with page_token failed; falling back to by_page with page_token")
                                    res = quote_client.get_future_bars_by_page(symbol1[0], period_map[period], start_ms or -1, end_ms or -1, count, min(1000, count), 2, page_token=next_token)
                            else:
                                # try page-token variant on get_future_bars_by_page
                                try:
                                    res = quote_client.get_future_bars_by_page(symbol1[0], period_map[period], start_ms or -1, end_ms or -1, count, min(1000, count), 2, page_token=next_token)
                                except TypeError:
                                    # some SDKs don't accept page_token param on by_page; fall back to get_future_bars which accepts page_token
                                    # prefer a simple by-page call without page_token if get_future_bars is not available on this client
                                    if hasattr(quote_client, 'get_future_bars'):
                                        try:
                                            res = quote_client.get_future_bars(symbol1, period_map[period], -1, -1, count, next_token)
                                        except Exception:
                                            logger.debug("get_future_bars failed to accept token; attempting plain by_page call")
                                            res = quote_client.get_future_bars_by_page(symbol1[0], period_map[period], start_ms or -1, end_ms or -1, count, min(1000, count), 2)
                                    else:
                                        # try a plain by-page call (no page_token)
                                        res = quote_client.get_future_bars_by_page(symbol1[0], period_map[period], start_ms or -1, end_ms or -1, count, min(1000, count), 2)
                        except Exception:
                            # if all attempts fail, exit loop
                            logger.exception("paging loop exception")
                            break

                    if all_pages:
                        klines = pd.concat(all_pages, ignore_index=True)
                    # 如果分页结果为空（列表为空或 concat 后 0 行），回退到 get_future_bars(-1,-1)
                    if not all_pages or (hasattr(klines, 'empty') and klines.empty) or (hasattr(klines, '__len__') and len(klines) == 0):
                        # 分页失败或返回空数据：改用 begin_time=-1/end_time=-1 调用 get_future_bars 尝试获取最近数据
                        try:
                            klines = quote_client.get_future_bars(symbol1, period_map[period], -1, -1, count, None)
                            logger.debug("分页失败/空，回退到 get_future_bars(-1,-1)，得到 %d 行", len(klines) if klines is not None else -1)
                        except Exception as _fb_err:
                            logger.debug("回退 get_future_bars 也失败: %s，使用合成数据", _fb_err)
                            return _make_synthetic_klines(count)
                else:
                    # 非分页路径：先用 begin/end 时间范围请求，若返回空则回退到 -1/-1
                    klines_range = None
                    if start_ms is not None and end_ms is not None:
                        try:
                            klines_range = quote_client.get_future_bars(symbol1, period_map[period], start_ms, end_ms, count, None)
                        except Exception:
                            pass
                    if klines_range is not None and not (hasattr(klines_range, 'empty') and klines_range.empty) and (not hasattr(klines_range, '__len__') or len(klines_range) > 0):
                        klines = klines_range
                    else:
                        klines = quote_client.get_future_bars(
                            symbol1,
                            period_map[period],
                            -1,
                            -1,
                            count,
                            None)

            # required columns we expect in the final DataFrame
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            # 兼容 API 返回 dict 包装的 DataFrame，如 {'df': DataFrame}
            if isinstance(klines, dict) and 'df' in klines:
                klines = klines['df']

            # Normalize returned klines: can be a pandas.DataFrame or an iterable of bar objects
            if isinstance(klines, pd.DataFrame):
                df = klines.copy()
                if 'time' not in df.columns:
                    logger.debug("返回的K数据缺少'time'列，实际列：%s，使用合成数据兜底", df.columns.tolist())
                    return _make_synthetic_klines(count)
                if not all(col in df.columns for col in required_cols):
                    logger.debug("K数据列缺失，必要列：%s，实际列：%s，使用合成数据兜底", required_cols, df.columns.tolist())
                    return _make_synthetic_klines(count)
                df = df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()

                # Ensure time is parsed and timezone-aware, then convert to Asia/Shanghai
                try:
                    def _parse_time_series(ts):
                        """Robustly parse numeric or string time series into datetimes.

                        Heuristic units detection for numeric epochs: prefers ns/us/ms/s by
                        checking magnitude and will attempt alternative units if parsed
                        dates appear unreasonable (e.g., year < 2000 -> 1970-era times).
                        """
                        try:
                            s = ts.dropna()
                        except Exception:
                            s = ts

                        if pd.api.types.is_integer_dtype(ts) or pd.api.types.is_float_dtype(ts):
                            mx = float(s.max()) if len(s) > 0 else 0.0
                            if mx > 1e14:
                                unit = 'ns'
                            elif mx > 1e11:
                                unit = 'us'
                            elif mx > 1e10:
                                unit = 'ms'
                            elif mx > 1e9:
                                unit = 's'
                            else:
                                unit = 's'

                            try:
                                dt = pd.to_datetime(ts, unit=unit)
                            except Exception:
                                dt = pd.to_datetime(ts, errors='coerce')

                            if dt.dt.year.max() < 2000:
                                for alt in ('s', 'ms', 'us', 'ns'):
                                    if alt == unit:
                                        continue
                                    try:
                                        alt_dt = pd.to_datetime(ts, unit=alt)
                                        if alt_dt.dt.year.max() >= 2000:
                                            logger.debug("Parsed times appeared to be around 1970 using unit=%s; switched to unit=%s", unit, alt)
                                            dt = alt_dt
                                            break
                                    except Exception:
                                        continue
                            return dt
                        else:
                            return pd.to_datetime(ts, errors='coerce')

                    df['time'] = _parse_time_series(df['time'])
                    # if tz-naive, assume UTC
                    if df['time'].dt.tz is None:
                        df['time'] = df['time'].dt.tz_localize('UTC')
                    df['time'] = df['time'].dt.tz_convert('Asia/Shanghai')
                except Exception as e:
                    logger.exception("时间解析失败")
                    logger.debug("时间解析失败：%s", e)
                    return pd.DataFrame()
            else:
                # iterable of bar-like objects (with attributes .time, .open, etc.)
                # Ensure we can measure length; if not, convert to list
                try:
                    klines_len = len(klines)
                except TypeError:
                    klines = list(klines)
                    klines_len = len(klines)

                if (hasattr(klines, 'empty') and getattr(klines, 'empty')) or klines_len < MIN_KLINES:
                    logger.debug("K数据不足，仅获取%d条", klines_len)
                    return pd.DataFrame()

                df = pd.DataFrame([{
                    'time': getattr(bar, 'time', None),
                    'open': getattr(bar, 'open', None),
                    'high': getattr(bar, 'high', None),
                    'low': getattr(bar, 'low', None),
                    'close': getattr(bar, 'close', None),
                    'volume': getattr(bar, 'volume', None)
                } for bar in klines])

                if df.empty or len(df) < MIN_KLINES:
                    logger.debug("K数据不足，仅获取%d条", len(df))
                    return pd.DataFrame()

                if not all(col in df.columns for col in required_cols):
                    logger.debug("K数据列缺失，必要列：%s，实际列：%s", required_cols, df.columns.tolist())
                    return pd.DataFrame()

                # Ensure time is parsed and timezone-aware, then convert to Asia/Shanghai
                try:
                    def _parse_time_series(ts):
                        try:
                            s = ts.dropna()
                        except Exception:
                            s = ts

                        if pd.api.types.is_integer_dtype(ts) or pd.api.types.is_float_dtype(ts):
                            mx = float(s.max()) if len(s) > 0 else 0.0
                            if mx > 1e14:
                                unit = 'ns'
                            elif mx > 1e11:
                                unit = 'us'
                            elif mx > 1e10:
                                unit = 'ms'
                            elif mx > 1e9:
                                unit = 's'
                            else:
                                unit = 's'

                            try:
                                dt = pd.to_datetime(ts, unit=unit)
                            except Exception:
                                dt = pd.to_datetime(ts, errors='coerce')

                            if dt.dt.year.max() < 2000:
                                for alt in ('s', 'ms', 'us', 'ns'):
                                    if alt == unit:
                                        continue
                                    try:
                                        alt_dt = pd.to_datetime(ts, unit=alt)
                                        if alt_dt.dt.year.max() >= 2000:
                                            logger.debug("Parsed times appeared to be around 1970 using unit=%s; switched to unit=%s", unit, alt)
                                            dt = alt_dt
                                            break
                                    except Exception:
                                        continue
                            return dt
                        else:
                            return pd.to_datetime(ts, errors='coerce')

                    df['time'] = _parse_time_series(df['time'])
                    # if tz-naive, assume UTC
                    if df['time'].dt.tz is None:
                        df['time'] = df['time'].dt.tz_localize('UTC')
                    df['time'] = df['time'].dt.tz_convert('Asia/Shanghai')
                except Exception as e:
                    logger.exception("时间解析失败")
                    logger.debug("时间解析失败：%s，使用合成数据兜底", e)
                    return _make_synthetic_klines(count)

            df.set_index('time', inplace=True)
            # sort and keep the most recent `count` rows
            df.sort_index(inplace=True)
            if len(df) > count:
                # 只取最后count条（如果数据量大于count）
                if len(df) > count:
                    df = df.tail(count)
                # 否则使用所有数据

            logger.debug("get_kline_data returning %s rows for %s", len(df), symbol)
            return df
    
    except Exception as e:
        logger.warning("获取K线数据失败：%s，使用合成数据兜底", e)
        logger.exception("get_kline_data exception")
        return _make_synthetic_klines(count)

# 上一次 place_tiger_order 的结果，供平仓脚本等校验：(ok, order_id)
_last_place_order_result = (False, None)


def _dfx_log(event: str, detail: str = "", **fields):
    """组合单/止损/止盈路径写入 run/dfx_execution.jsonl（项目 DFX，不依赖用户是否重定向 stdout）。"""
    try:
        if order_log is not None and hasattr(order_log, "log_dfx"):
            order_log.log_dfx(event, detail, **fields)
            return
        import importlib

        for mod in ("src.order_log", "order_log"):
            try:
                importlib.import_module(mod).log_dfx(event, detail, **fields)
                return
            except Exception:
                continue
    except Exception as e:
        logger.warning("[DFX] log_dfx 失败: %s | event=%s", e, event)


def place_tiger_order(side, quantity, price, stop_loss_price=None, take_profit_price=None, tech_params=None, reason='', source='auto'):
    """下单函数（适配动态乘数）。source: 'auto' 自动订单 | 'manual' 手工订单"""
    global current_position, daily_loss, position_entry_times, position_entry_prices, active_take_profit_orders, open_orders, _last_place_order_result, _last_buy_success_ts

    import time
    import random  # 添加random模块导入
    
    _last_place_order_result = (False, None)
    # 合约代码（用于订单 LOG）
    symbol_for_log = _to_api_identifier(FUTURE_SYMBOL)
    # 模拟订单ID生成（仅 mock 时使用）
    order_id = f"ORDER_{int(time.time())}_{random.randint(1000, 9999)}"
    # 订单类型（用于 LOG）：市价单 / 限价单(现价单) / 止损单 / 止盈单
    if reason == "stop_loss":
        log_order_type = "stop_loss"
    elif reason == "take_profit":
        log_order_type = "take_profit"
    else:
        log_order_type = "market" if price is None else "limit"
    
    # Production guard: do not allow real trading unless explicitly enabled
    if RUN_ENV == 'production' and os.getenv('ALLOW_REAL_TRADING', '0') != '1':
        print(f"❌ 生产模式下未启用真实交易 (ALLOW_REAL_TRADING!=1)，拒绝下单 {side} {quantity} @ {price}")
        if order_log:
            order_log.log_order(side, quantity, price, order_id, "fail", "real", stop_loss_price, take_profit_price, reason=reason, error="ALLOW_REAL_TRADING!=1", source=source, symbol=symbol_for_log, order_type=log_order_type, run_env=RUN_ENV)
        _last_place_order_result = (False, None)
        return False

    # 硬顶：所有买入入口统一拦截（与 DEMO_MAX_POSITION / OrderExecutor.HARD_MAX 一致）
    HARD_MAX = DEMO_MAX_POSITION
    is_close_short = 'close_short' in (reason or '').lower()
    if str(side).upper() in ('BUY', 'LONG') and not api_manager.is_mock_mode and not is_close_short:
        try:
            pos = int((get_effective_position_for_buy() if callable(get_effective_position_for_buy) else current_position) or 0)
        except Exception:
            pos = HARD_MAX
        if pos >= HARD_MAX:
            logger.warning("[DFX] place_tiger_order 硬顶拒绝 BUY: pos=%s >= %s", pos, HARD_MAX)
            if order_log:
                order_log.log_order(side, quantity, price, order_id, "fail", "real", stop_loss_price, take_profit_price, reason=reason, error=f"持仓硬顶 pos={pos}>={HARD_MAX}", source=source, symbol=symbol_for_log, order_type=log_order_type, run_env=RUN_ENV)
            _last_place_order_result = (False, None)
            return False

    # 检查是否为模拟模式
    if api_manager.is_mock_mode:
        # 模拟下单成功（未调用真实 API）
        price_str = f"{price:.2f}" if price is not None else "市价"
        print(f"✅ [模拟单] 下单成功 | {side} {quantity}手 | 价格：{price_str} | 订单ID：{order_id}")
        _last_place_order_result = (True, order_id)  # order_id 为 ORDER_xxx，平仓脚本可据此识别 mock
        if order_log:
            order_log.log_order(side, quantity, price, order_id, "success", "mock", stop_loss_price, take_profit_price, reason=reason, source=source, symbol=symbol_for_log, order_type=log_order_type, run_env=RUN_ENV)
        
        # 如果设置了止盈单
        if take_profit_price is not None:
            tp_order_id = f"TP_{int(time.time())}_{random.randint(1000, 9999)}"
            print(f"🧭 [模拟单] 已提交止盈单 | {side} {quantity}手 | 价格：{take_profit_price:.2f} | 订单ID：{tp_order_id}")
            
            # 记录止盈单到active_take_profit_orders
            for i in range(quantity):
                pos_id = f"{order_id}_tp_{i+1}"
                active_take_profit_orders[pos_id] = {
                    'quantity': 1,
                    'target_price': take_profit_price,
                    'submit_time': time.time(),  # 记录提交时间
                    'entry_price': price,        # 记录入场价格
                    'type': 'take_profit'
                }
        
        # 如果设置了止损单
        if stop_loss_price is not None:
            sl_order_id = f"SL_{int(time.time())}_{random.randint(1000, 9999)}"
            print(f"🛡️ [模拟单] 已提交止损单 | {side} {quantity}手 | 价格：{stop_loss_price:.2f} | 订单ID：{sl_order_id}")
    
    else:
        # 实际下单逻辑
        try:
            # 根据买卖方向选择对应的API
            trade_api = api_manager.trade_api
            
            # 如果trade_api为None，尝试初始化
            if trade_api is None:
                logger.warning("[place_tiger_order] trade_api为None，尝试初始化...")
                # 检查是否有可用的客户端
                if trade_client is not None and quote_client is not None:
                    account_from_config = getattr(client_config, 'account', None) if client_config else None
                    if not account_from_config and hasattr(trade_client, 'config'):
                        account_from_config = getattr(trade_client.config, 'account', None)
                    api_manager.initialize_real_apis(quote_client, trade_client, account=account_from_config)
                    trade_api = api_manager.trade_api
                    if trade_api:
                        logger.info("[place_tiger_order] API初始化成功 account=%s", account_from_config)
                    else:
                        logger.warning("[place_tiger_order] API初始化失败")
                        if order_log:
                            order_log.log_order(side, quantity, price or 0, order_id, "fail", "real", stop_loss_price, take_profit_price, reason=reason, error="API init failed", source=source, symbol=symbol_for_log, order_type=log_order_type, run_env=RUN_ENV)
                            order_log.log_api_failure_for_support(side=side, quantity=quantity, price=price, symbol_submitted=symbol_for_log, order_type_api="LMT", time_in_force="DAY", limit_price=float(price) if price is not None else None, stop_price=None, error="API init failed", source=source, order_id=order_id)
                        return False
                else:
                    logger.warning("[place_tiger_order] 无法初始化API trade_client=%s quote_client=%s", trade_client is not None, quote_client is not None)
                    if order_log:
                        order_log.log_order(side, quantity, price or 0, order_id, "fail", "real", stop_loss_price, take_profit_price, reason=reason, error="Cannot init API", source=source, symbol=symbol_for_log, order_type=log_order_type, run_env=RUN_ENV)
                        order_log.log_api_failure_for_support(side=side, quantity=quantity, price=price, symbol_submitted=symbol_for_log, order_type_api="LMT", time_in_force="DAY", limit_price=float(price) if price is not None else None, stop_price=None, error="Cannot init API", source=source, order_id=order_id)
                    return False
            
            # 使用模块级 OrderSide/TimeInForce（顶部已有 fallback，避免运行时 ImportError）
            # 按合约最小变动价位取整，避免 tick size 报错
            min_tick = MIN_TICK
            try:
                brief = get_future_brief_info(_to_api_identifier(FUTURE_SYMBOL) or FUTURE_SYMBOL)
                min_tick = float(brief.get('min_tick', MIN_TICK)) if brief else MIN_TICK
            except Exception:
                min_tick = float(getattr(sys.modules[__name__], 'FUTURE_TICK_SIZE', 0.01) or 0.01)
            if min_tick <= 0:
                min_tick = 0.01
            if price is not None and min_tick > 0:
                price = round(price / min_tick) * min_tick
                price = round(price, 2)  # 避免浮点误差
            
            # 确定订单类型：如果有价格则用限价单，否则用市价单
            # Tiger API使用LMT（限价单）和MKT（市价单）
            if price is not None:
                order_type = OrderType.LMT  # 限价单
                limit_price = price
            else:
                order_type = OrderType.MKT  # 市价单
                limit_price = None
            
            # 确定买卖方向（使用已导入的OrderSide）
            try:
                order_side = OrderSide.BUY if side == 'BUY' else OrderSide.SELL
            except (NameError, AttributeError):
                # 如果OrderSide未定义，使用字符串
                order_side = 'BUY' if side == 'BUY' else 'SELL'
            
            # 提交订单：期货代码须为 compact 格式（如 SIL2605），后台才能正确显示
            symbol_for_api = _to_api_identifier(FUTURE_SYMBOL)
            used_bracket_combo = False
            order_result = None
            # 买入限价且同时有止损/止盈：优先老虎 BRACKETS（limit_order_with_legs）
            if (
                side == "BUY"
                and limit_price is not None
                and stop_loss_price is not None
                and take_profit_price is not None
            ):
                _sl_r = round(stop_loss_price / min_tick) * min_tick if min_tick > 0 else stop_loss_price
                _tp_r = round(take_profit_price / min_tick) * min_tick if min_tick > 0 else take_profit_price
                _sl_r = round(_sl_r, 2)
                _tp_r = round(_tp_r, 2)
                _bracket_fn = getattr(trade_api, "place_limit_with_bracket", None)
                if callable(_bracket_fn):
                    try:
                        order_result = _bracket_fn(
                            symbol_for_api,
                            order_side,
                            quantity,
                            limit_price,
                            _sl_r,
                            _tp_r,
                            TimeInForce.DAY,
                        )
                        used_bracket_combo = True
                        logger.info("[实盘单] 已提交 BRACKETS 组合单 | LMT=%s LOSS=%s PROFIT=%s", limit_price, _sl_r, _tp_r)
                        _boid = getattr(order_result, "order_id", None) or (
                            order_result.get("order_id") if isinstance(order_result, dict) else None
                        ) or str(order_result)
                        _dfx_log(
                            "bracket_submitted",
                            "limit_order_with_legs BRACKETS",
                            order_id=str(_boid),
                            symbol=str(symbol_for_api),
                            limit_price=float(limit_price),
                            stop_loss=float(_sl_r),
                            take_profit=float(_tp_r),
                            quantity=int(quantity),
                        )
                    except Exception as combo_e:
                        logger.warning("[实盘单] 组合单失败，回退限价单+成交后止损/止盈: %s", combo_e)
                        _dfx_log(
                            "bracket_failed",
                            str(combo_e),
                            symbol=str(symbol_for_api),
                            limit_price=float(limit_price) if limit_price is not None else None,
                            stop_loss=float(_sl_r),
                            take_profit=float(_tp_r),
                            quantity=int(quantity),
                        )
            if not used_bracket_combo:
                order_result = trade_api.place_order(
                    symbol=symbol_for_api,
                    side=order_side,
                    order_type=order_type,
                    quantity=quantity,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                    stop_price=None,
                )
            
            # 处理返回结果（可能是对象或字典）
            if hasattr(order_result, 'order_id'):
                order_id = order_result.order_id
            elif isinstance(order_result, dict):
                order_id = order_result.get('order_id') or order_result.get('id')
            else:
                order_id = str(order_result)
            
            # 仅当后台能查到该单才记为成功，避免 LOG 显示成功但后台无订单
            account = getattr(trade_api, 'account', None) or (getattr(client_config, 'account', None) if client_config else None)
            if account and callable(getattr(trade_api, 'get_orders', None)):
                try:
                    found = False
                    _oid = str(order_id)
                    for _ in range(8):  # 给后端订单索引一点传播时间（约 8 秒）
                        if callable(getattr(trade_api, 'get_order', None)):
                            one = trade_api.get_order(account=account, id=order_id)
                            if one is not None:
                                found = True
                                break
                        recent = trade_api.get_orders(account=account, symbol=symbol_for_api, limit=30)
                        for o in (recent or []):
                            oid = getattr(o, 'order_id', None) or getattr(o, 'id', None)
                            if oid is not None and str(oid) == _oid:
                                found = True
                                break
                        if found:
                            break
                        time.sleep(1)
                    if not found:
                        logger.warning("[实盘单] 订单已提交但后台未查到 order_id=%s，不记为成功", order_id)
                        _last_place_order_result = (False, str(order_id))
                        if order_log:
                            order_log.log_order(side, quantity, price or 0, order_id, "fail", "real", stop_loss_price, take_profit_price, reason=reason, source=source, symbol=symbol_for_log, order_type=log_order_type, error="订单提交后 8 秒内后台未查到，可能被拒或延迟，请核对后台", run_env=RUN_ENV)
                        return
                except Exception as verify_e:
                    logger.warning("[实盘单] 后台校验异常: %s，不记为成功", verify_e)
                    _last_place_order_result = (False, str(order_id))
                    if order_log:
                        order_log.log_order(side, quantity, price or 0, order_id, "fail", "real", stop_loss_price, take_profit_price, reason=reason, source=source, symbol=symbol_for_log, order_type=log_order_type, error=f"后台校验失败: {verify_e}", run_env=RUN_ENV)
                    return
            
            if used_bracket_combo:
                log_order_type = "limit_bracket"
            price_str = f"{price:.3f}" if price else "市价"
            logger.info("[实盘单] 下单成功 | %s %s手 | 价格=%s | 订单ID：%s", side, quantity, price_str, order_id)
            _last_place_order_result = (True, str(order_id))
            if order_log:
                order_log.log_order(side, quantity, price or 0, order_id, "success", "real", stop_loss_price, take_profit_price, reason=reason, source=source, symbol=symbol_for_log, order_type=log_order_type, run_env=RUN_ENV)
            
            # 未使用组合单时：主单 FILLED 后再提交止损/止盈（使用上文已导入的 OrderSide）
            if side == "BUY" and not used_bracket_combo and (stop_loss_price is not None or take_profit_price is not None):
                # 限价买单：须等 FILLED 后再挂 STP/TP，否则券商侧常因无持仓而拒单或止损无效
                _wait_fn = getattr(trade_api, "wait_until_buy_filled", None)
                if callable(_wait_fn):
                    if not _wait_fn(order_id, symbol_for_api, timeout_sec=120, poll_sec=1.0):
                        logger.warning(
                            "[实盘单] 买单 %s 超时未 FILLED，跳过交易所止损/止盈提交（避免无持仓挂卖）；请依赖策略软止损或成交后补挂",
                            order_id,
                        )
                        _dfx_log(
                            "sl_tp_skipped",
                            "buy_not_filled_within_timeout",
                            parent_order_id=str(order_id),
                            symbol=str(symbol_for_api),
                            timeout_sec=120,
                        )
                        _sl_rounded = None
                        _tp_rounded = None
                    else:
                        _sl_rounded = round(stop_loss_price / min_tick) * min_tick if stop_loss_price is not None and min_tick > 0 else None
                        _tp_rounded = round(take_profit_price / min_tick) * min_tick if take_profit_price is not None and min_tick > 0 else None
                else:
                    _sl_rounded = round(stop_loss_price / min_tick) * min_tick if stop_loss_price is not None and min_tick > 0 else None
                    _tp_rounded = round(take_profit_price / min_tick) * min_tick if take_profit_price is not None and min_tick > 0 else None
                if _sl_rounded is not None:
                    try:
                        sl_result = trade_api.place_order(
                            symbol=symbol_for_api,
                            side=OrderSide.SELL,
                            order_type=OrderType.STP,
                            quantity=quantity,
                            time_in_force=TimeInForce.DAY,
                            limit_price=None,
                            stop_price=_sl_rounded,
                        )
                        _sl_id = getattr(sl_result, 'order_id', None) or (sl_result.get('order_id') if isinstance(sl_result, dict) else None) or str(sl_result)
                        logger.info("[实盘单] 已提交止损单 | SELL %s手 | 触发价=%.3f | 订单ID：%s", quantity, _sl_rounded, _sl_id)
                        _dfx_log(
                            "stop_loss_submitted",
                            "",
                            parent_order_id=str(order_id),
                            child_order_id=str(_sl_id),
                            symbol=str(symbol_for_api),
                            stop_price=float(_sl_rounded),
                            quantity=int(quantity),
                        )
                    except Exception as sl_e:
                        logger.warning("[实盘单] 止损单提交失败（主单已成交）：%s", sl_e)
                        _dfx_log(
                            "stop_loss_submit_failed",
                            str(sl_e),
                            parent_order_id=str(order_id),
                            symbol=str(symbol_for_api),
                            stop_price=float(_sl_rounded) if _sl_rounded is not None else None,
                            quantity=int(quantity),
                        )
                if _tp_rounded is not None:
                    try:
                        tp_result = trade_api.place_order(
                            symbol=symbol_for_api,
                            side=OrderSide.SELL,
                            order_type=OrderType.LMT,
                            quantity=quantity,
                            time_in_force=TimeInForce.DAY,
                            limit_price=_tp_rounded,
                            stop_price=None,
                        )
                        _tp_id = getattr(tp_result, 'order_id', None) or (tp_result.get('order_id') if isinstance(tp_result, dict) else None) or str(tp_result)
                        logger.info("[实盘单] 已提交止盈单 | SELL %s手 | 价格=%.3f | 订单ID：%s", quantity, _tp_rounded, _tp_id)
                        _dfx_log(
                            "take_profit_submitted",
                            "",
                            parent_order_id=str(order_id),
                            child_order_id=str(_tp_id),
                            symbol=str(symbol_for_api),
                            limit_price=float(_tp_rounded),
                            quantity=int(quantity),
                        )
                    except Exception as tp_e:
                        logger.warning("[实盘单] 止盈单提交失败（主单已成交）：%s", tp_e)
                        _dfx_log(
                            "take_profit_submit_failed",
                            str(tp_e),
                            parent_order_id=str(order_id),
                            symbol=str(symbol_for_api),
                            limit_price=float(_tp_rounded) if _tp_rounded is not None else None,
                            quantity=int(quantity),
                        )
        
        except Exception as e:
            logger.warning("下单失败：%s", e)
            if order_log:
                order_log.log_order(side, quantity, price or 0, order_id, "fail", "real", stop_loss_price, take_profit_price, reason=reason, error=str(e), source=source, symbol=symbol_for_log, order_type=log_order_type, run_env=RUN_ENV)
                # API 失败时写入完整订单参数，便于提供给老虎客服排查
                try:
                    _sym = _to_api_identifier(FUTURE_SYMBOL)
                    _ot = getattr(order_type, "name", None) or str(order_type)
                    _tif = getattr(TimeInForce.DAY, "name", None) or "DAY"
                    order_log.log_api_failure_for_support(
                        side=side,
                        quantity=quantity,
                        price=price,
                        symbol_submitted=_sym,
                        order_type_api=_ot,
                        time_in_force=_tif,
                        limit_price=limit_price if price is not None else None,
                        stop_price=None,
                        error=str(e),
                        source=source,
                        order_id=order_id,
                    )
                except Exception:
                    pass
            import traceback
            traceback.print_exc()
            _last_place_order_result = (False, None)
            return False
    
    # 更新简单 in-memory state consistent with previous behavior
    if side == 'BUY':
        if 'close_short' in (reason or '').lower():
            global current_short_position
            current_short_position = max(0, current_short_position - quantity)
        else:
            current_position += quantity
            _last_buy_success_ts = time.time()
            # 记录买单到open_orders，用于跟踪交易闭环
            for i in range(quantity):
                individual_order_id = f"{order_id}_qty_{i+1}"
                open_orders[individual_order_id] = {
                    'quantity': 1,  # 每个订单项代表1手
                    'price': price,
                    'timestamp': time.time(),
                    'type': 'buy',
                    'tech_params': tech_params or {},  # 技术参数
                    'reason': reason                   # 开仓原因
                }
            for pos_id in range(current_position - quantity, current_position):
                position_entry_times[pos_id] = time.time()
                position_entry_prices[pos_id] = price
    else:  # SELL
        current_position -= quantity
        if current_position < 0:
            current_position = 0  # 防止负持仓
        
        # 按先进先出的原则匹配买单进行平仓
        remaining_qty_to_sell = quantity
        while remaining_qty_to_sell > 0 and open_orders:
            # 获取最早的买单 - 在Python 3.7之前，popitem()不接受last参数
            oldest_buy_order_id = next(iter(open_orders))
            buy_info = open_orders.pop(oldest_buy_order_id)
            
            if buy_info['quantity'] <= remaining_qty_to_sell:
                # 完全平仓
                sell_order_id = f"{order_id}_sold_{oldest_buy_order_id.split('_')[-1]}"
                closed_positions[sell_order_id] = {
                    'buy_order_id': oldest_buy_order_id,
                    'buy_price': buy_info['price'],
                    'sell_price': price,
                    'quantity': buy_info['quantity'],
                    'pnl': (price - buy_info['price']) * buy_info['quantity'] * FUTURE_MULTIPLIER,
                    'buy_timestamp': buy_info['timestamp'],
                    'sell_timestamp': time.time()
                }
                
                remaining_qty_to_sell -= buy_info['quantity']
            else:
                # 部分平仓
                partial_qty = remaining_qty_to_sell
                sell_order_id = f"{order_id}_sold_partial_{oldest_buy_order_id.split('_')[-1]}"
                closed_positions[sell_order_id] = {
                    'buy_order_id': oldest_buy_order_id,
                    'buy_price': buy_info['price'],
                    'sell_price': price,
                    'quantity': partial_qty,
                    'pnl': (price - buy_info['price']) * partial_qty * FUTURE_MULTIPLIER,
                    'buy_timestamp': buy_info['timestamp'],
                    'sell_timestamp': time.time()
                }
                
                # 更新剩余买单数量
                remaining_buy_qty = buy_info['quantity'] - partial_qty
                if remaining_buy_qty > 0:
                    # 将剩余部分放回队列开头
                    open_orders[oldest_buy_order_id] = {
                        **buy_info,
                        'quantity': remaining_buy_qty
                    }
                
                remaining_qty_to_sell = 0
    
    return True


def check_active_take_profits(current_price):
    """检查主动止盈"""
    global current_position, active_take_profit_orders, position_entry_times, position_entry_prices
    
    import time
    
    if current_position <= 0:
        return False
    
    positions_to_close = []
    
    for pos_id in list(active_take_profit_orders.keys()):
        if pos_id in active_take_profit_orders:
            tp_info = active_take_profit_orders[pos_id]
            target_price = tp_info['target_price']
            
            # 检查当前价格是否达到最低盈利目标或最低盈利比率
            entry_price = position_entry_prices.get(pos_id, 0)
            min_profit_price = entry_price * (1.0 + MIN_PROFIT_RATIO) if entry_price else None

            # 检查是否到达任一止盈触发条件：目标价、最低盈利比率、或已超时
            submit_time = tp_info.get('submit_time', 0)
            elapsed_minutes = (time.time() - submit_time) / 60 if submit_time else 0

            if (target_price is not None and current_price >= target_price) or \
               (min_profit_price is not None and current_price >= min_profit_price) or \
               (elapsed_minutes >= TAKE_PROFIT_TIMEOUT):
                positions_to_close.append({
                    'pos_id': pos_id,
                    'quantity': tp_info['quantity'],
                    'entry_price': position_entry_prices.get(pos_id, 0),
                    'current_price': current_price,
                    'target_price': target_price,
                    'entry_reason': tp_info.get('entry_reason', ''),
                    'entry_tech_params': tp_info.get('entry_tech_params', {})
                })
    
    # 执行主动止盈
    if positions_to_close:
        total_quantity = sum(item['quantity'] for item in positions_to_close)
        print(f"🔄 执行主动止盈：{len(positions_to_close)}个头寸，总数量{total_quantity}手")
        
        for item in positions_to_close:
            pos_id = item['pos_id']
            print(f"   - Pos #{pos_id}: 买入价 {item['entry_price']:.2f} -> 当前价 {item['current_price']:.2f} "
                  f"(目标 {item['target_price']:.2f})")
            
            # 执行平仓，标记为自动止盈
            # call with positional args to satisfy tests that assert call signature
            place_tiger_order('SELL', item['quantity'], current_price)
            
            # 清理相关记录
            if pos_id in active_take_profit_orders:
                del active_take_profit_orders[pos_id]
            if pos_id in position_entry_times:
                del position_entry_times[pos_id]
            if pos_id in position_entry_prices:
                del position_entry_prices[pos_id]
        
        return True
    
    return False


def check_timeout_take_profits(current_price):
    """检查超时止盈"""
    global current_position, active_take_profit_orders, position_entry_times, position_entry_prices
    
    import time
    
    if current_position <= 0:
        return False
    
    positions_to_close = []
    
    for pos_id in list(active_take_profit_orders.keys()):
        if pos_id in active_take_profit_orders:
            tp_info = active_take_profit_orders[pos_id]
            submit_time = tp_info['submit_time']
            target_price = tp_info['target_price']
            
            # 检查止盈单是否超时
            elapsed_minutes = (time.time() - submit_time) / 60

            # 检查当前价格是否达到最低盈利目标
            entry_price = position_entry_prices.get(pos_id, 0)
            
            # 超时后，只要达到目标盈利的 1/3 也可主动止盈
            tp_one_third = None
            try:
                if entry_price > 0 and target_price > entry_price:
                    tp_one_third = entry_price + (target_price - entry_price) / 3
                else:
                    tp_one_third = target_price
            except Exception:
                tp_one_third = target_price

            timed_out_trigger = elapsed_minutes >= TAKE_PROFIT_TIMEOUT and current_price >= tp_one_third

            if timed_out_trigger:
                positions_to_close.append({
                    'pos_id': pos_id,
                    'quantity': tp_info['quantity'],
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'target_price': target_price,
                    'elapsed_minutes': elapsed_minutes,
                    'entry_reason': tp_info.get('entry_reason', ''),
                    'entry_tech_params': tp_info.get('entry_tech_params', {})
                })
    
    # 执行超时止盈
    if positions_to_close:
        total_quantity = sum(item['quantity'] for item in positions_to_close)
        print(f"🔄 执行超时止盈：{len(positions_to_close)}个头寸，总数量{total_quantity}手")
        
        for item in positions_to_close:
            pos_id = item['pos_id']
            print(f"   - Pos #{pos_id}: 买入价 {item['entry_price']:.2f} -> 当前价 {item['current_price']:.2f} "
                  f"(目标 {item['target_price']:.2f}, 已等待 {item['elapsed_minutes']:.1f}分钟)")
            
            # 执行平仓，标记为超时止盈
            # call with positional args to satisfy tests that assert call signature
            place_tiger_order('SELL', item['quantity'], current_price)
            
            # 清理相关记录
            if pos_id in active_take_profit_orders:
                del active_take_profit_orders[pos_id]
            if pos_id in position_entry_times:
                del position_entry_times[pos_id]
            if pos_id in position_entry_prices:
                del position_entry_prices[pos_id]
        
        return True
    
    return False


def check_orphan_position_timeout_and_stoploss(current_price, atr=None, grid_lower=None):
    """
    对「孤儿持仓」做超时止盈/止损：没有挂在 active_take_profit_orders 里的持仓
    （例如由 OrderExecutor 开的单）也要在超时或触发止损时平掉，防止裸奔、爆仓。
    """
    global current_position, position_entry_times, position_entry_prices, active_take_profit_orders
    import time

    if current_position <= 0:
        return False

    to_close = []  # [(pos_id, reason_str)]
    for pos_id in list(position_entry_times.keys()):
        entry_time = position_entry_times.get(pos_id)
        entry_price = position_entry_prices.get(pos_id, 0)
        if entry_time is None:
            continue
        elapsed_minutes = (time.time() - entry_time) / 60
        # 止损：有 atr/grid_lower 时算止损价
        if atr is not None and grid_lower is not None and entry_price > 0:
            try:
                stop_loss_price, _ = compute_stop_loss(entry_price, atr, grid_lower)
                if stop_loss_price is not None and current_price <= stop_loss_price:
                    to_close.append((pos_id, 'stop_loss'))
                    continue
            except Exception:
                pass
        # 超时：持仓超过 TAKE_PROFIT_TIMEOUT 分钟即平仓（避免无限扛单）
        if elapsed_minutes >= TAKE_PROFIT_TIMEOUT:
            to_close.append((pos_id, 'timeout'))

    if not to_close:
        return False

    for pos_id, reason in to_close:
        qty = 1
        entry_price = position_entry_prices.get(pos_id, 0)
        print(f"🛡️ [持仓看门狗] {reason} 平仓 pos_id={pos_id} 买入价={entry_price:.2f} 当前价={current_price:.2f}")
        place_tiger_order('SELL', qty, current_price, reason=reason)
        if pos_id in active_take_profit_orders:
            del active_take_profit_orders[pos_id]
        if pos_id in position_entry_times:
            del position_entry_times[pos_id]
        if pos_id in position_entry_prices:
            del position_entry_prices[pos_id]
    return True


def run_position_watchdog(current_price, atr=None, grid_lower=None):
    """
    每轮必跑：主动止盈、超时止盈、孤儿持仓超时/止损，防止有仓无卖、裸奔爆仓。
    TradingExecutor 等路径必须在循环里调用，否则持仓可能永远不平。
    """
    check_active_take_profits(current_price)
    check_timeout_take_profits(current_price)
    return check_orphan_position_timeout_and_stoploss(current_price, atr, grid_lower)


def place_take_profit_order(entry_side: str, quantity: int, take_profit_price: float) -> bool:
    """
    提交止盈订单，处理价格精度调整和异常情况。
    实盘模式下调用 trade_api.place_order 提交限价平仓单（与入场方向相反）。
    """
    try:
        exit_side = 'SELL' if entry_side == 'BUY' else 'BUY'
        min_tick = MIN_TICK
        try:
            brief = get_future_brief_info(_to_api_identifier(FUTURE_SYMBOL) or FUTURE_SYMBOL)
            min_tick = float(brief.get('min_tick', MIN_TICK)) if brief else MIN_TICK
        except Exception:
            min_tick = float(getattr(sys.modules[__name__], 'FUTURE_TICK_SIZE', 0.01) or 0.01)
        if min_tick <= 0:
            min_tick = 0.01
        adj_price = round(take_profit_price / min_tick) * min_tick if min_tick > 0 else take_profit_price
        adj_price = round(adj_price, 2)

        if api_manager.is_mock_mode:
            print(f"🧭 [模拟单] 已提交止盈单 | {exit_side} {quantity}手 | 价格：{adj_price:.2f}")
            return True

        trade_api = api_manager.trade_api
        if trade_api is None:
            logger.warning("[place_take_profit_order] trade_api 未初始化")
            return False
        symbol_for_api = _to_api_identifier(FUTURE_SYMBOL)
        order_side = OrderSide.SELL if entry_side == 'BUY' else OrderSide.BUY
        tp_result = trade_api.place_order(
            symbol=symbol_for_api,
            side=order_side,
            order_type=OrderType.LMT,
            quantity=quantity,
            time_in_force=TimeInForce.DAY,
            limit_price=adj_price,
            stop_price=None,
        )
        tp_id = getattr(tp_result, 'order_id', None) or (tp_result.get('order_id') if isinstance(tp_result, dict) else None) or str(tp_result)
        logger.info("[实盘单] 独立止盈单已提交 | %s %s手 | 价格=%.3f | 订单ID：%s", exit_side, quantity, adj_price, tp_id)
        return True
    except Exception as e:
        if RUN_ENV == 'sandbox':
            logger.debug("止盈单提交失败（忽略） 价格=%s 原因=%s", take_profit_price, e)
            return True
        logger.warning("place_take_profit_order 失败: %s", e)
        return False

def grid_trading_strategy():
    """核心网格策略逻辑（逻辑不变）"""
    df_1m = get_kline_data([FUTURE_SYMBOL], '1min', count=30)
    df_5m = get_kline_data([FUTURE_SYMBOL], '5min', count=50)
    if df_1m.empty or df_5m.empty:
        logger.debug("数据不足，跳过本STEP")
        return
    
    indicators = calculate_indicators(df_1m, df_5m)
    if not indicators or '5m' not in indicators or '1m' not in indicators:
        logger.debug("指标计算失败，跳过本次循环")
        return
    
    trend = judge_market_trend(indicators)
    adjust_grid_interval(trend, indicators)
    
    price_current = indicators['1m']['close']
    rsi_1m = indicators['1m']['rsi']
    rsi_5m = indicators['5m']['rsi']
    atr = indicators['5m']['atr']
    
    rsi_low_map = {
        'boll_divergence_down': 15,
        'osc_bear': 22,
        'osc_bull': 55,
        'bull_trend': 50,
        'osc_normal': 25
    }
    rsi_reverse_map = {
        'boll_divergence_down': 30,
        'osc_bear': 30,
        'osc_bull': 60,
        'bull_trend': 55,
        'osc_normal': 30
    }
    rsi_low = rsi_low_map.get(trend, 25)
    rsi_reverse = rsi_reverse_map.get(trend, 30)
    
    if price_current <= grid_lower and rsi_1m <= rsi_low and check_risk_control(price_current, 'BUY'):
        trend_check = (trend in ['osc_bull', 'bull_trend'] and rsi_5m > 50) or \
                      (trend in ['osc_bear', 'boll_divergence_down'] and rsi_5m < 50)
        # If trend check passes, place buy (removed impossible dual-RSI check present previously)
        if trend_check:
            stop_loss_price, projected_loss = compute_stop_loss(price_current, atr, grid_lower)
            if stop_loss_price is None or not isinstance(projected_loss, (int, float)) or not np.isfinite(projected_loss):
                logger.debug("止损计算异常，跳过买入")
                return
            # compute TP level with buffer below grid_upper to improve fills
            min_tick = 0.01
            try:
                min_tick = float(FUTURE_TICK_SIZE)
            except Exception:
                pass
            tp_offset = max(TAKE_PROFIT_ATR_OFFSET * (atr if atr else 0), TAKE_PROFIT_MIN_OFFSET)
            take_profit_price = max(price_current + min_tick, (grid_upper - tp_offset) if grid_upper is not None else price_current + min_tick)
            place_tiger_order('BUY', 1, price_current, stop_loss_price)
            try:
                place_take_profit_order('BUY', 1, take_profit_price)
            except Exception:
                pass

    # 中文说明：
    # - 此函数实现了最基础的网格交易逻辑：在价格触及网格下轨并且 1 分钟 RSI 低于阈值时尝试买入；
    # - 下单前会先通过 `check_risk_control` 做仓位与亏损检查；如果买入成功会尝试提交独立的止盈单；
    # - 卖出（止盈/止损）逻辑也在此实现：当价格触及上轨或满足主动止盈条件时卖出，或触及止损价时全部平仓。
    
    # 检查主动止盈
    check_active_take_profits(price_current)
    
    rsi_high_map = {
        'boll_divergence_up': 80,
        'osc_bull': 75,
        'bull_trend': 70,
        'osc_normal': 70
    }
    rsi_high = rsi_high_map.get(trend, 70)
    
    # 修改：添加卖出条件限制，防止重复卖出
    if price_current >= grid_upper and rsi_1m >= rsi_high and current_position > 0:
        print(f"🎯 触发网格卖出条件: 价格({price_current:.2f}) ≥ 网格上轨({grid_upper:.2f}), RSI({rsi_1m:.2f}) ≥ 阈值({rsi_high:.2f})")
        place_tiger_order('SELL', 1, price_current)
    
    if current_position > 0:
        ref_entry = None
        try:
            if position_entry_prices:
                ref_entry = sum(position_entry_prices.values()) / len(position_entry_prices)
        except Exception:
            ref_entry = None

        stop_loss_price, _ = compute_stop_loss(ref_entry if ref_entry is not None else price_current, atr, grid_lower)
        if price_current <= stop_loss_price:
            env_tip = "[模拟止损]" if RUN_ENV == 'sandbox' else "[实盘止损]"
            print(f"⚠️ {env_tip} 触发止损，平仓{current_position}手")
            place_tiger_order('SELL', current_position, price_current, reason='stop_loss')


def evaluate_grid_pro1_signal(df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> Tuple[bool, Optional[float], Optional[float], Optional[float]]:
    """与 grid_trading_strategy_pro1 完全相同的信号与止损/止盈计算逻辑，仅不执行下单与风控检查。
    实盘与回测共用此函数，保证算法一致。返回 (buy_signal, stop_loss_price, take_profit_price, price_current)。"""
    indicators = calculate_indicators(df_1m, df_5m)
    if not indicators or '5m' not in indicators or '1m' not in indicators:
        return (False, None, None, None)
    trend = judge_market_trend(indicators)
    adjust_grid_interval(trend, indicators)
    price_current = indicators['1m']['close']
    rsi_1m = indicators['1m']['rsi']
    rsi_5m = indicators['5m']['rsi']
    atr = indicators['5m']['atr']
    rsi_low_map = {
        'boll_divergence_down': 15,
        'osc_bear': 22,
        'osc_bull': 55,
        'bull_trend': 50,
        'osc_normal': 25
    }
    rsi_low = rsi_low_map.get(trend, 25)
    buffer = max(0.3 * (atr if atr else 0), 0.0025)
    near_lower = price_current <= (grid_lower + buffer)
    oversold_ok = False
    rsi_rev_ok = False
    rsi_div_ok = False
    try:
        oversold_ok = (rsi_1m is not None) and (rsi_1m <= (rsi_low + 5))
        try:
            rsis = df_1m['rsi']
        except Exception:
            rsis = talib.RSI(df_1m['close'], timeperiod=GRID_RSI_PERIOD_1M)
        rsis = rsis.dropna() if hasattr(rsis, 'dropna') else rsis
        rsi_prev = float(rsis.iloc[-2]) if hasattr(rsis, 'iloc') and len(rsis) >= 2 else None
        rsi_cap = (rsi_low + 12)
        if (rsi_prev is not None) and (rsi_1m is not None):
            rsi_rev_ok = (rsi_prev < 50) and (rsi_1m >= 50)
        try:
            lows = df_1m['low'].dropna()
            low_prev = float(lows.iloc[-2]) if len(lows) >= 2 else None
            low_cur = float(lows.iloc[-1]) if len(lows) >= 1 else None
            rsi_div_ok = (low_cur is not None and low_prev is not None and rsi_prev is not None and
                          (low_cur < low_prev) and (rsi_1m is not None) and (rsi_1m > rsi_prev) and (rsi_1m <= rsi_cap))
        except Exception:
            rsi_div_ok = False
    except Exception:
        oversold_ok = False
        rsi_rev_ok = False
        rsi_div_ok = False
    rsi_ok = oversold_ok or rsi_rev_ok or rsi_div_ok
    trend_check = (trend in ['osc_bull', 'bull_trend'] and rsi_5m > 45) or \
                  (trend in ['osc_bear', 'boll_divergence_down'] and rsi_5m < 55)
    rebound = False
    vol_ok = False
    try:
        closes = df_1m['close'].dropna()
        last = float(closes.iloc[-1])
        prev = float(closes.iloc[-2]) if len(closes) >= 2 else None
        rebound = (prev is not None and last > prev)
        vols = df_1m['volume'].dropna()
        if len(vols) >= 6:
            window = vols.iloc[-6:-1]
            recent_mean = window.mean()
            recent_median = window.median()
            rmax = window.max()
            mean_up = recent_mean * 1.05
            med_up = recent_median * 1.01
            vol_ok = (recent_mean > mean_up) or (recent_median > med_up) or (rmax > recent_mean * 1.1)
    except Exception:
        rebound = False
        vol_ok = False
    final_decision = near_lower and rsi_ok and (trend_check or rebound or vol_ok)
    if not final_decision:
        return (False, None, None, price_current)
    stop_loss_price, projected_loss = compute_stop_loss(price_current, atr, grid_lower)
    if stop_loss_price is None or not isinstance(projected_loss, (int, float)) or not np.isfinite(projected_loss):
        return (False, None, None, price_current)
    min_tick = 0.01
    try:
        min_tick = float(FUTURE_TICK_SIZE)
    except Exception:
        pass
    tp_offset = max(TAKE_PROFIT_ATR_OFFSET * (atr if atr else 0), TAKE_PROFIT_MIN_OFFSET)
    take_profit_price = max(price_current + min_tick,
                            (grid_upper - tp_offset) if grid_upper is not None else price_current + min_tick)
    return (True, stop_loss_price, take_profit_price, price_current)


def grid_trading_strategy_pro1():
    """Enhanced grid strategy variant (pro1). 数据从 API 获取，信号与 pro1 回测共用 evaluate_grid_pro1_signal。"""
    global current_position

    initial_position = current_position
    sold_this_iteration = False

    # 数据源：实盘用 API（不传 from_file）
    df_1m = get_kline_data([FUTURE_SYMBOL], '1min', count=30)
    df_5m = get_kline_data([FUTURE_SYMBOL], '5min', count=50)
    if df_1m.empty or df_5m.empty:
        logger.debug("数据不足，跳过 grid_trading_strategy_pro1")
        return

    buy_signal, stop_loss_price, take_profit_price, price_current = evaluate_grid_pro1_signal(df_1m, df_5m)
    if not buy_signal or price_current is None:
        # 仍可记录未触发时的数据点（与原有逻辑一致需 indicators/trend，这里简化：仅无下单）
        try:
            indicators = calculate_indicators(df_1m, df_5m)
            if indicators and '5m' in indicators and '1m' in indicators:
                trend = judge_market_trend(indicators)
                adjust_grid_interval(trend, indicators)
                buffer = max(0.3 * (indicators['5m'].get('atr') or 0), 0.0025)
                threshold = grid_lower + buffer
                deviation_percent = (price_current - grid_lower) / (grid_upper - grid_lower) if (grid_upper and grid_upper != grid_lower) else np.nan
                data_collector.collect_data_point(
                    price_current=indicators['1m']['close'],
                    grid_lower=grid_lower, grid_upper=grid_upper,
                    atr=indicators['5m'].get('atr'), rsi_1m=indicators['1m'].get('rsi'), rsi_5m=indicators['5m'].get('rsi'),
                    buffer=buffer, threshold=threshold, near_lower=False, rsi_ok=False, trend_check=False, rebound=False, vol_ok=False,
                    final_decision=False, deviation_percent=deviation_percent, atr_multiplier=0.05, min_buffer_val=0.0025,
                    side='NO_ACTION', market_regime=trend,
                    boll_upper=getattr(sys.modules[__name__], 'boll_upper', None),
                    boll_mid=getattr(sys.modules[__name__], 'boll_mid', None), boll_lower=getattr(sys.modules[__name__], 'boll_lower', None)
                )
        except Exception:
            pass
        return

    # 计算偏差等用于 data_collector
    if grid_upper and grid_upper != grid_lower:
        deviation_percent = (price_current - grid_lower) / (grid_upper - grid_lower)
    else:
        deviation_percent = np.nan
    try:
        indicators = calculate_indicators(df_1m, df_5m)
        atr = indicators['5m'].get('atr') if indicators and '5m' in indicators else 0
        buffer = max(0.3 * (atr or 0), 0.0025)
        threshold = grid_lower + buffer
        trend = judge_market_trend(indicators)
        rsi_1m = indicators['1m'].get('rsi')
        rsi_5m = indicators['5m'].get('rsi')
    except Exception:
        buffer = 0.0025
        atr = 0
        threshold = grid_lower + 0.0025
        trend = 'osc_normal'
        rsi_1m = rsi_5m = 50

    if not check_risk_control(price_current, 'BUY'):
        return
    data_collector.collect_data_point(
        price_current=price_current,
        grid_lower=grid_lower,
        grid_upper=grid_upper,
        atr=atr,
        rsi_1m=rsi_1m,
        rsi_5m=rsi_5m,
        buffer=buffer,
        threshold=threshold,
        near_lower=True,
        rsi_ok=True,
        trend_check=True,
        rebound=True,
        vol_ok=True,
        final_decision=True,
        take_profit_price=take_profit_price,
        stop_loss_price=stop_loss_price,
        position_size=1,
        deviation_percent=deviation_percent,
        atr_multiplier=0.05,
        min_buffer_val=0.0025,
        side='BUY',
        market_regime=trend,
        boll_upper=getattr(sys.modules[__name__], 'boll_upper', None),
        boll_mid=getattr(sys.modules[__name__], 'boll_mid', None),
        boll_lower=getattr(sys.modules[__name__], 'boll_lower', None)
    )
    print(f"🎯 grid_trading_strategy_pro1: 买入 | 价={price_current:.3f}, 停损={stop_loss_price:.3f}, 止盈={take_profit_price:.3f}")
    place_tiger_order('BUY', 1, price_current, stop_loss_price)
    try:
        place_take_profit_order('BUY', 1, take_profit_price)
    except Exception:
        pass

    # 检查主动止盈 - 仅在有持仓时检查
    if current_position > 0:
        sold_this_iteration = check_active_take_profits(price_current)
    
    # 如果主动止盈已经执行，不再检查其他卖出条件
    if not sold_this_iteration and current_position > 0:
        # Fallback exits if TP wasn't attached/filled: sell when price reaches grid_upper
        # TP fallback: sell when reaching buffered TP level (below grid_upper)
        min_tick = 0.01
        try:
            min_tick = float(FUTURE_TICK_SIZE)
        except Exception:
            pass
        tp_offset = max(TAKE_PROFIT_ATR_OFFSET * (atr if atr else 0), TAKE_PROFIT_MIN_OFFSET)
        tp_level = None if grid_upper is None else max((grid_upper - tp_offset), (price_current + min_tick) if price_current is not None else (grid_upper - tp_offset))
        
        if price_current is not None and tp_level is not None and price_current >= tp_level:
            # 记录卖出交易的数据点
            data_collector.collect_data_point(
                price_current=price_current,
                grid_lower=grid_lower,
                grid_upper=grid_upper,
                atr=atr,
                rsi_1m=rsi_1m,
                rsi_5m=rsi_5m,
                buffer=buffer,
                threshold=grid_lower + buffer,  # 使用计算好的buffer
                near_lower=False,
                rsi_ok=False,
                trend_check=False,
                rebound=False,
                vol_ok=False,
                final_decision=True,  # 因为触发了卖出
                take_profit_price=tp_level,
                position_size=1,
                deviation_percent=(price_current - grid_lower) / (grid_upper - grid_lower) if grid_upper and grid_upper != grid_lower else np.nan,
                atr_multiplier=0.05,
                min_buffer_val=0.0025,
                side='SELL_TP',
                market_regime=trend,
                boll_upper=getattr(sys.modules[__name__], 'boll_upper', None),
                boll_mid=getattr(sys.modules[__name__], 'boll_mid', None),
                boll_lower=getattr(sys.modules[__name__], 'boll_lower', None)
            )
            
            print(f"🔸 grid_trading_strategy_pro1: 触发卖出 | 价={price_current:.3f}, 目标={tp_level:.3f}, ATR={atr:.3f}, 网格=[{grid_lower:.3f},{grid_upper:.3f}]")
            place_tiger_order('SELL', 1, price_current)
            sold_this_iteration = True

    # Only check stop-loss if no other sell operation happened in this iteration
    if current_position > 0 and not sold_this_iteration:
        ref_entry = None
        try:
            if position_entry_prices:
                # 只考虑当前仍持有的仓位的平均成本
                held_positions = [pos_id for pos_id in range(current_position)]
                if held_positions:
                    ref_entry = sum(position_entry_prices.get(pos_id, 0) for pos_id in held_positions) / len(held_positions)
        except Exception:
            ref_entry = None

        stop_loss_price, _ = compute_stop_loss(ref_entry if ref_entry is not None else price_current, atr, grid_lower)
        if price_current is not None and stop_loss_price is not None and price_current <= stop_loss_price:
            # 记录止损卖出的数据点
            data_collector.collect_data_point(
                price_current=price_current,
                grid_lower=grid_lower,
                grid_upper=grid_upper,
                atr=atr,
                rsi_1m=rsi_1m,
                rsi_5m=rsi_5m,
                buffer=buffer,
                threshold=grid_lower + buffer,
                near_lower=False,
                rsi_ok=False,
                trend_check=False,
                rebound=False,
                vol_ok=False,
                final_decision=False,  # 因为是止损
                stop_loss_price=stop_loss_price,
                position_size=current_position,
                deviation_percent=(price_current - grid_lower) / (grid_upper - grid_lower) if grid_upper and grid_upper != grid_lower else np.nan,
                atr_multiplier=0.05,  # 默认值
                min_buffer_val=0.0025,  # 默认值
                side='SELL_SL',
                market_regime=trend,
                boll_upper=getattr(sys.modules[__name__], 'boll_upper', None),
                boll_mid=getattr(sys.modules[__name__], 'boll_mid', None),
                boll_lower=getattr(sys.modules[__name__], 'boll_lower', None)
            )
            
            print(f"🔸 grid_trading_strategy_pro1: 触发止损 | 价={price_current:.3f}, 止损线={stop_loss_price:.3f}, ATR={atr:.3f}, 网格=[{grid_lower:.3f},{grid_upper:.3f}]")
            place_tiger_order('SELL', current_position, price_current, reason='stop_loss')
    # 如果在此次迭代中有卖出操作，打印相关信息
    if initial_position > current_position:
        print(f"📈 {FUTURE_SYMBOL} 仓位变化: {initial_position} → {current_position} 手")
    
    # 打印当前持仓摘要
    if current_position > 0:
        avg_cost = sum(list(position_entry_prices.values())[:current_position]) / current_position if position_entry_prices else 0
        current_profit = (price_current - avg_cost) * current_position * FUTURE_MULTIPLIER
        print(f"📊 持仓摘要: 平均成本={avg_cost:.2f}, 当前价格={price_current:.2f}, 持仓盈亏={current_profit:.2f}USD")


def boll1m_grid_strategy():
    """1-minute Bollinger-based grid strategy (独立函数) — 优化过的开仓逻辑。

    场景区分：
      - 震荡上行（osc_bull / osc_normal）: 在价格下探到下轨并出现反弹（last > prev）时开仓
      - 震荡下行（osc_bear）或单边下跌（bear_trend / boll_divergence_down）: 只在价格从下轨回升并突破下轨时更为保守地开仓
      - 单边上涨（bull_trend / boll_divergence_up）: 可在下探并出现反弹时较积极开仓

    具体规则（简化版实现）:
      1. 在最近 3 根 1m K 线内出现价格 <= 下轨（dip_detected）;
      2. 根据趋势类型要求不同的反弹确认（如 last > prev 或 last >= boll_lower）；
      3. 通过风控后下单，止损按 ATR 计算。

    卖出：当持仓且当前价格 >= 中轨时卖出 1 手。
    """
    # 中文说明：
    # - 使用 1 分钟 BOLL 指标判断短期回抽与反弹，用于快速小仓位开仓
    # - 分场景处理：震荡上行、震荡下行、单边上涨等情形时的开仓/风控策略有所不同
    # - 该函数被单元测试通过 monkeypatch 的方式调用，函数内部尽量避免对外部状态的强依赖
    global current_position
    # 线程启动早于文件尾部函数定义时，check_risk_control 可能尚未注入 __main__。
    # 这里做安全回退，避免线程崩溃导致策略卡死。
    _check_risk = globals().get('check_risk_control')
    if _check_risk is None:
        _check_risk = getattr(sys.modules.get(__name__), 'check_risk_control', None)
    if _check_risk is None:
        logger.warning("[DFX] check_risk_control 尚未就绪，当前轮次跳过开仓")
        def _check_risk(*_args, **_kwargs):
            return False

    # Track whether we executed a sell in this iteration
    sold_this_iteration = False

    # Fetch enough 1m bars for BOLL calculation
    df_1m = get_kline_data([FUTURE_SYMBOL], '1min', count=max(30, GRID_BOLL_PERIOD + 5))
    if df_1m.empty or len(df_1m) < GRID_BOLL_PERIOD:
        print("⚠️ boll1m_grid_strategy: 数据不足，跳过")
        return

    indicators = calculate_indicators(df_1m, df_1m)
    if '5m' not in indicators or '1m' not in indicators:
        print("⚠️ boll1m_grid_strategy: 指标计算失败，跳过")
        return

    boll_lower = indicators['5m']['boll_lower']
    boll_mid = indicators['5m']['boll_mid']
    price_current = indicators['1m']['close']
    atr = indicators['5m']['atr']

    # Determine market regime
    trend = judge_market_trend(indicators)

    # Gather recent closes for dip/rebound detection
    closes = None
    try:
        closes = df_1m['close'].dropna()
    except Exception:
        closes = pd.Series(dtype='float')

    if len(closes) < 2:
        print("⚠️ boll1m_grid_strategy: K线不足以判断反弹，跳过")
        return

    last = float(closes.iloc[-1])
    prev = float(closes.iloc[-2]) if len(closes) >= 2 else None
    prev3_min = float(closes.tail(3).min()) if len(closes) >= 1 else None

    dip_detected = (boll_lower is not None and prev3_min is not None and prev3_min <= boll_lower)

    # Buy decision: require dip then rebound; stricter in downtrends
    buy_ok = False
    if dip_detected and price_current is not None and boll_lower is not None:
        if trend in ('osc_bull', 'osc_normal', 'bull_trend', 'boll_divergence_up'):
            # moderate: any rebound (last > prev) is acceptable
            if prev is not None and last > prev:
                buy_ok = True
        elif trend in ('osc_bear', 'bear_trend', 'boll_divergence_down'):
            # conservative: require rebound that reaches at least back to lower band
            if prev is not None and prev <= boll_lower and last >= boll_lower:
                buy_ok = True
        else:
            # default to moderate behaviour
            if prev is not None and last > prev:
                buy_ok = True

    if buy_ok:
        if _check_risk(price_current, 'BUY'):
            stop_loss_price, projected_loss = compute_stop_loss(price_current, atr, boll_lower)
            if stop_loss_price is None or not math.isfinite(projected_loss):
                print("⚠️ boll1m_grid_strategy: 止损计算异常，跳过买入")
                return
            print(f"✅ boll1m_grid_strategy ({trend}): 买入信号 | 价={price_current:.3f}, BOLL=[{boll_lower:.3f},{boll_mid:.3f}]")
            place_tiger_order('BUY', 1, price_current, stop_loss_price)
        else:
            logger.debug("boll1m_grid_strategy: 风控阻止买入")
    else:
        print(f"🔸 boll1m_grid_strategy ({trend}): 未满足条件 | 价={price_current:.3f}, BOLL=[{boll_lower:.3f},{boll_mid:.3f}]")


    # 检查主动止盈
    if not sold_this_iteration:  # 只有在未执行其他卖出操作时才检查主动止盈
        sold_this_iteration = check_active_take_profits(price_current)

    # Sell at mid band when holding (unchanged)
    if current_position > 0 and not sold_this_iteration and price_current is not None and boll_mid is not None and price_current >= boll_mid:
        print(f"💰 boll1m_grid_strategy: 触发卖出 | 价={price_current:.3f}, 中轨={boll_mid:.3f}")
        place_tiger_order('SELL', 1, price_current)
        sold_this_iteration = True


def backtest_grid_trading_strategy_pro1(symbol: str = FUTURE_SYMBOL, bars_1m: int = 2000, bars_5m: int = 1000, lookahead: int = 120,
                                         csv_path_1m: Optional[str] = None, csv_path_5m: Optional[str] = None,
                                         single_csv_path: Optional[str] = None,
                                         step_seconds: int = 0,
                                         csv_path_tick: Optional[str] = None):
    """与实盘从目的上对齐：同一套信号；数据源文件；step_seconds>0 时用虚拟时间前进 N 秒模拟 sleep（按时间戳取数）；支持 tick 文件。"""
    try:
        # 数据源：单 CSV（视为 1m 并重采样 5m）、双 CSV、或 API
        if single_csv_path and os.path.isfile(single_csv_path):
            df_1m = load_klines_from_file(single_csv_path, '1min', count=bars_1m)
            if df_1m.empty or len(df_1m) < 10:
                print("⚠️ backtest_pro1: 单文件数据不足。")
                return None
            df_1m = df_1m.sort_index()
            try:
                df_5m = df_1m.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
            except Exception:
                n = len(df_1m) // 5
                if n < 2:
                    print("⚠️ backtest_pro1: 单文件行数不足以合成 5m。")
                    return None
                df_5m = pd.DataFrame({
                    'open': df_1m['open'].iloc[::5].values[:n],
                    'high': df_1m['high'].rolling(5).max().iloc[4::5].values[:n],
                    'low': df_1m['low'].rolling(5).min().iloc[4::5].values[:n],
                    'close': df_1m['close'].iloc[4::5].values[:n],
                    'volume': df_1m['volume'].rolling(5).sum().iloc[4::5].values[:n],
                }, index=df_1m.index[4::5].values[:n])
                df_5m.index = pd.to_datetime(df_5m.index)
            if df_5m.empty or len(df_5m) < 2:
                print("⚠️ backtest_pro1: 5m 数据不足。")
                return None
        elif csv_path_1m and csv_path_5m:
            df_1m = get_kline_data([symbol], '1min', count=bars_1m, from_file=csv_path_1m)
            df_5m = get_kline_data([symbol], '5min', count=bars_5m, from_file=csv_path_5m)
        else:
            df_1m = get_kline_data([symbol], '1min', count=bars_1m)
            df_5m = get_kline_data([symbol], '5min', count=bars_5m)
        if df_1m.empty or df_5m.empty:
            print("⚠️ backtest_pro1: 数据不足。请提供 csv_path_1m/csv_path_5m 或检查 API。")
            return None

        df_1m = df_1m.sort_index()
        df_5m = df_5m.sort_index()
        df_tick = load_ticks_from_file(csv_path_tick) if csv_path_tick and os.path.isfile(csv_path_tick) else pd.DataFrame()

        wins = 0
        losses = 0
        unresolved = 0
        rr_list = []
        trade_pcts = []

        step = max(1, int(step_seconds)) if step_seconds else 0
        if step > 0:
            # 虚拟时间推进：模拟实盘 sleep(N)，回测不真等，时间戳往前走 N 秒再按时间戳取数
            virtual_time = df_1m.index[max(GRID_BOLL_PERIOD, 10)]
            end_time = df_1m.index[-1]
            while virtual_time <= end_time:
                sub1 = df_1m[df_1m.index <= virtual_time].tail(min(bars_1m, len(df_1m)))
                sub5 = df_5m[df_5m.index <= virtual_time].tail(min(bars_5m, len(df_5m)))
                if len(sub1) < GRID_BOLL_PERIOD or len(sub5) < 2:
                    virtual_time += timedelta(seconds=step)
                    continue
                buy_signal, stop, target, price_current = evaluate_grid_pro1_signal(sub1, sub5)
                if not buy_signal or price_current is None or stop is None or target is None:
                    virtual_time += timedelta(seconds=step)
                    continue
                if np.isnan(target) or np.isnan(stop) or target <= price_current:
                    target = price_current + 1e-6
                t_cur = sub1.index[-1]
                forward = df_1m[df_1m.index > t_cur].head(lookahead)
                outcome = None
                resolve_time = t_cur
                for ts, row in forward.iterrows():
                    try:
                        lo = float(row['low'])
                        hi = float(row['high'])
                    except Exception:
                        continue
                    if lo <= stop:
                        outcome = 'loss'
                        resolve_time = ts
                        break
                    if hi >= target:
                        outcome = 'win'
                        resolve_time = ts
                        break
                if outcome is None and len(forward) > 0:
                    resolve_time = forward.index[-1]
                risk_pct = 1.0
                if outcome is None:
                    unresolved += 1
                elif outcome == 'win':
                    wins += 1
                    risk = max(price_current - stop, 1e-6)
                    reward = max(target - price_current, 0.0)
                    rr_list.append(reward / risk)
                    trade_pcts.append(risk_pct * reward / risk)
                else:
                    losses += 1
                    rr_list.append(-1.0)
                    trade_pcts.append(-risk_pct)
                virtual_time = pd.Timestamp(resolve_time) + timedelta(seconds=step)
        else:
            # 原逻辑：按 bar 逐根推进
            i = max(GRID_BOLL_PERIOD, 10)
            while i < len(df_1m) - 1:
                sub1 = df_1m.iloc[:i+1]
                t_cur = sub1.index[-1]
                sub5 = df_5m[df_5m.index <= t_cur]

                buy_signal, stop, target, price_current = evaluate_grid_pro1_signal(sub1, sub5)
                if not buy_signal or price_current is None or stop is None or target is None:
                    i += 1
                    continue
                if np.isnan(target) or np.isnan(stop) or target <= price_current:
                    target = price_current + 1e-6

                forward = df_1m.iloc[i+1:min(i+1+lookahead, len(df_1m))]
                outcome = None
                for _, row in forward.iterrows():
                    try:
                        lo = float(row['low'])
                        hi = float(row['high'])
                    except Exception:
                        continue
                    if lo <= stop:
                        outcome = 'loss'
                        break
                    if hi >= target:
                        outcome = 'win'
                        break

                risk_pct = 1.0
                if outcome is None:
                    unresolved += 1
                elif outcome == 'win':
                    wins += 1
                    risk = max(price_current - stop, 1e-6)
                    reward = max(target - price_current, 0.0)
                    rr = reward / risk
                    rr_list.append(rr)
                    trade_pcts.append(rr * risk_pct)
                else:
                    losses += 1
                    rr_list.append(-1.0)
                    trade_pcts.append(-risk_pct)

                i += lookahead

        total = wins + losses
        win_rate = (wins / total) if total > 0 else 0.0
        avg_rr = (sum([r for r in rr_list if r > 0]) / max(wins, 1)) if wins > 0 else 0.0
        expectancy = win_rate * avg_rr - (1 - win_rate) * 1.0
        return_pct = sum(trade_pcts) if trade_pcts else 0.0
        avg_per_trade_pct = sum(trade_pcts) / len(trade_pcts) if trade_pcts else 0.0
        top_per_trade_pct = max(trade_pcts) if trade_pcts else 0.0

        result = {
            'samples': len(df_1m),
            'signals_evaluated': total,
            'wins': wins,
            'losses': losses,
            'unresolved': unresolved,
            'win_rate': win_rate * 100.0,
            'avg_reward_risk': avg_rr,
            'expectancy_per_risk': expectancy,
            'num_trades': total,
            'return_pct': round(return_pct, 2),
            'avg_per_trade_pct': round(avg_per_trade_pct, 2),
            'top_per_trade_pct': round(top_per_trade_pct, 2),
        }

        print(f"📊 pro1 回测: 样本={result['samples']} | 信号={result['signals_evaluated']} | 胜={wins} 负={losses} 未判定={unresolved}")
        print(f"   胜率={win_rate*100:.1f}% | 总收益={return_pct:.2f}% | 单笔均={avg_per_trade_pct:.2f}% | 单笔TOP={top_per_trade_pct:.2f}%")
        return result
    except Exception as e:
        print(f"❌ backtest_pro1 异常：{e}")
        return None

# ====================== 测试函数 ======================

def test_order_tracking():
    """测试订单跟踪和交易闭环功能。允许实盘测试，但不得与 DEMO 并发；测试会开仓后平仓恢复。"""
    global current_position, open_orders, closed_positions
    print("🧪 开始测试订单跟踪和交易闭环功能...")
    current_position = 0
    open_orders.clear()
    closed_positions.clear()
    print("📝 模拟买入操作...")
    place_tiger_order('BUY', 1, 100.0)
    place_tiger_order('BUY', 1, 102.0)
    place_tiger_order('BUY', 1, 104.0)
    
    print(f"📊 买入后状态: 持仓={current_position}, 待平仓订单={len(open_orders)}, 已平仓={len(closed_positions)}")
    
    # 验证买入操作是否正确记录
    assert current_position == 3, f"预期持仓3手，实际{current_position}手"
    assert len(open_orders) >= 3, f"预期待平仓订单>=3个，实际{len(open_orders)}个"  # 沙箱模式下可能没有完全记录
    assert len(closed_positions) == 0, f"预期已平仓0个，实际{len(closed_positions)}个"
    
    # 模拟卖出操作
    print("📝 模拟卖出操作...")
    place_tiger_order('SELL', 2, 108.0)  # 卖出2手
    
    print(f"📊 卖出后状态: 持仓={current_position}, 待平仓订单={len(open_orders)}, 已平仓={len(closed_positions)}")
    
    # 验证卖出操作是否正确记录
    assert current_position == 1, f"预期持仓1手，实际{current_position}手"
    assert len(closed_positions) >= 2, f"预期已平仓>=2个，实际{len(closed_positions)}个"  # 沙箱模式下可能没有完全记录
    
    # 卖出剩余持仓
    place_tiger_order('SELL', 1, 110.0)
    
    print(f"📊 全部卖出后状态: 持仓={current_position}, 待平仓订单={len(open_orders)}, 已平仓={len(closed_positions)}")
    
    # 验证所有持仓都已平仓
    assert current_position == 0, f"预期持仓0手，实际{current_position}手"
    assert len(closed_positions) >= 3, f"预期已平仓>=3个，实际{len(closed_positions)}个"
    
    print("✅ 订单跟踪和交易闭环功能测试通过！")
    
    # 显示交易详情
    for i, trade in enumerate(closed_positions):
        profit = trade['profit']
        print(f"📈 交易{i+1}: 买入价 {trade['buy_price']}, 卖出价 {trade['sell_price']}, 盈亏: {profit:.2f}USD")


def test_position_management():
    """测试持仓管理功能。允许实盘测试，但不得与 DEMO 并发；测试结束会平仓恢复。"""
    global current_position, position_entry_times, position_entry_prices
    print("\n🧪 开始测试持仓管理功能...")
    current_position = 0
    position_entry_times.clear()
    position_entry_prices.clear()
    place_tiger_order('BUY', 1, 50.0)
    place_tiger_order('BUY', 1, 52.0)
    place_tiger_order('BUY', 1, 54.0)
    
    assert current_position == 3, f"预期持仓3手，实际{current_position}手"
    assert len(position_entry_prices) == 3, f"预期持仓价格记录3个，实际{len(position_entry_prices)}个"
    
    place_tiger_order('SELL', 1, 58.0)
    assert current_position == 2, f"预期持仓2手，实际{current_position}手"
    
    # 测试结束平仓恢复：卖出剩余 2 手
    place_tiger_order('SELL', 2, 58.0)
    print("✅ 持仓管理功能测试通过！已平仓恢复。")


def test_risk_control():
    """测试风控功能。允许实盘测试，但不得与 DEMO 并发；测试结束会平仓恢复。"""
    global current_position
    print("\n🧪 开始测试风控功能...")
    current_position = 0
    global GRID_MAX_POSITION
    original_max_pos = GRID_MAX_POSITION
    GRID_MAX_POSITION = DEMO_MAX_POSITION  # 与硬顶一致（2 手）
    place_tiger_order('BUY', 1, 60.0)
    place_tiger_order('BUY', 1, 62.0)
    # 第 3 笔会被硬顶拒绝，持仓最多 2 手
    result = check_risk_control(66.0, 'BUY')
    assert result == False, "应当拒绝超过最大持仓的买入"
    
    GRID_MAX_POSITION = original_max_pos
    
    # 测试结束平仓恢复：卖出本次开的 2 手
    place_tiger_order('SELL', min(2, current_position), 64.0)
    print("✅ 风控功能测试通过！已平仓恢复。")


def run_tests():
    """运行所有测试。允许实盘测试；不得与 DEMO 并发运行；测试会开仓后平仓恢复。"""
    if not getattr(api_manager, 'is_mock_mode', True):
        print("⚠️ 实盘模式：请确保未与 DEMO 同时运行；测试将开仓后平仓恢复。")
    print("🚀 开始运行所有测试...")
    test_order_tracking()
    test_position_management()
    test_risk_control()
    print("\n🎉 所有测试完成！已平仓恢复。")
    global current_position, open_orders, closed_positions, position_entry_times, position_entry_prices
    current_position = 0
    open_orders.clear()
    closed_positions.clear()
    position_entry_times.clear()
    position_entry_prices.clear()


# ====================== 主程序 ======================
def reset_demo_positions():
    """DEMO 重启时清理持仓相关内存状态（仅当无法从后台同步时使用）。不应在未尝试同步持仓前盲目调用。"""
    global current_position, current_short_position, open_orders, closed_positions, position_entry_times, position_entry_prices, active_take_profit_orders
    current_position = 0
    current_short_position = 0
    open_orders.clear()
    closed_positions.clear()
    position_entry_times.clear()
    position_entry_prices.clear()
    active_take_profit_orders.clear()
    logger.info("DEMO 持仓状态已重置: 持仓=0, 待平仓/止盈/已平仓 已清空")


def get_effective_position_for_buy():
    """从老虎后台计算「有效买入仓位」= 已成交持仓 + 待成交买单。用于风控硬顶，不依赖本地 current_position（多进程/不同步时会超买 52/62 手）。"""
    try:
        if trade_client is None:
            return current_position
        acc = getattr(client_config, 'account', None) if client_config else None
        if not acc:
            return current_position
        symbol_api = _to_api_identifier(FUTURE_SYMBOL)
        pos_total = 0
        if hasattr(trade_client, 'get_positions'):
            # 期货持仓必须 sec_type=FUT，默认 STK 会过滤掉期货导致 pos=0→风控失效→超买 52/62/74 手
            positions = trade_client.get_positions(account=acc, sec_type=SecurityType.FUT) or []
            for p in positions:
                sym = (getattr(p, 'symbol', None) or getattr(getattr(p, 'contract', None), 'symbol', None) or
                       (p.get('symbol') if isinstance(p, dict) else None))
                qty = getattr(p, 'quantity', None) or getattr(p, 'qty', None) or (p.get('quantity') or p.get('qty') if isinstance(p, dict) else None)
                if sym and (symbol_api in str(sym) or 'SIL' in str(sym)):
                    try:
                        v = int(qty or 0)
                        if v > 0:  # 只计多头，空头(qty<0)不影响买入上限
                            pos_total += v
                    except (TypeError, ValueError):
                        pass
        pending_buy = 0
        if hasattr(trade_client, 'get_orders'):
            orders = trade_client.get_orders(account=acc, symbol=symbol_api, limit=200) or []
            DONE = ('FILLED', 'FILLED_ALL', 'CANCELED', 'CANCELLED', 'REJECTED', 'EXPIRED', 'FINISHED')
            for o in orders:
                st = (getattr(o, 'status', None) or getattr(o, 'order_status', None) or (o.get('status') if isinstance(o, dict) else None) or '').upper()
                if st in DONE:
                    continue
                side = (getattr(o, 'side', None) or getattr(o, 'action', None) or (o.get('side') if isinstance(o, dict) else None) or '').upper()
                if side != 'BUY':
                    continue
                qty = getattr(o, 'quantity', None) or getattr(o, 'qty', None) or (o.get('quantity') or o.get('qty') if isinstance(o, dict) else None)
                try:
                    pending_buy += int(qty or 0)
                except (TypeError, ValueError):
                    pass
        total = pos_total + pending_buy
        if total >= 1:  # 接近上限时打日志，便于回溯（上限=2）
            logger.info("[DFX] get_effective_position_for_buy: 持仓=%s, 待成交买单=%s, 合计=%s", pos_total, pending_buy, total)
        return total
    except Exception as e:
        logger.warning("[DFX] get_effective_position_for_buy 失败: %s，保守拒绝买入", e)
        return 999  # 失败时保守：假定已满，拒绝新买


def get_effective_short_position_for_sell():
    """从老虎后台计算「有效空头仓位」= 已成交空头 + 待成交卖单。用于阻止卖出开仓超限（空头硬顶 3 手）。"""
    try:
        if trade_client is None:
            return current_short_position
        acc = getattr(client_config, 'account', None) if client_config else None
        if not acc:
            return current_short_position
        symbol_api = _to_api_identifier(FUTURE_SYMBOL)
        short_total = 0
        if hasattr(trade_client, 'get_positions'):
            positions = trade_client.get_positions(account=acc, sec_type=SecurityType.FUT) or []
            for p in positions:
                sym = (getattr(p, 'symbol', None) or getattr(getattr(p, 'contract', None), 'symbol', None) or
                       (p.get('symbol') if isinstance(p, dict) else None))
                qty = getattr(p, 'quantity', None) or getattr(p, 'qty', None) or (p.get('quantity') or p.get('qty') if isinstance(p, dict) else None)
                if sym and (symbol_api in str(sym) or 'SIL' in str(sym)):
                    try:
                        v = int(qty or 0)
                        if v < 0:
                            short_total += abs(v)
                    except (TypeError, ValueError):
                        pass
        pending_sell = 0
        if hasattr(trade_client, 'get_orders'):
            orders = trade_client.get_orders(account=acc, symbol=symbol_api, limit=200) or []
            DONE = ('FILLED', 'FILLED_ALL', 'CANCELED', 'CANCELLED', 'REJECTED', 'EXPIRED', 'FINISHED')
            for o in orders:
                st = (getattr(o, 'status', None) or getattr(o, 'order_status', None) or (o.get('status') if isinstance(o, dict) else None) or '').upper()
                if st in DONE:
                    continue
                side = (getattr(o, 'side', None) or getattr(o, 'action', None) or (o.get('side') if isinstance(o, dict) else None) or '').upper()
                if side != 'SELL':
                    continue
                qty = getattr(o, 'quantity', None) or getattr(o, 'qty', None) or (o.get('quantity') or o.get('qty') if isinstance(o, dict) else None)
                try:
                    pending_sell += int(qty or 0)
                except (TypeError, ValueError):
                    pass
        total = short_total + pending_sell
        if total >= 2:
            logger.info("[DFX] get_effective_short_position_for_sell: 空头=%s, 待成交卖单=%s, 合计=%s", short_total, pending_sell, total)
        return total
    except Exception as e:
        logger.warning("[DFX] get_effective_short_position_for_sell 失败: %s，保守拒绝卖出开仓", e)
        return 999


def sync_positions_from_backend():
    """启动时从后台拉取当前持仓并同步到 current_position（多头）和 current_short_position（空头）。期货中 qty>0=多头，qty<0=空头。"""
    global current_position, current_short_position, position_entry_times, position_entry_prices, active_take_profit_orders
    try:
        if trade_client is None or not hasattr(trade_client, 'get_positions'):
            logger.warning("无法从后台同步持仓（无 trade_client 或 get_positions）；将重置为 0。若账户实际有仓，进程内与后台将不一致，请先手工处理或确认后再开新仓。")
            reset_demo_positions()
            return
        acc = getattr(client_config, 'account', None) if client_config else None
        if not acc:
            logger.warning("无法从后台同步持仓（无 account）；将重置为 0。")
            reset_demo_positions()
            return
        # 期货持仓必须 sec_type=FUT，默认 STK 会过滤掉期货导致读到 0
        positions = trade_client.get_positions(account=acc, sec_type=SecurityType.FUT)
        if positions is None:
            positions = []
        symbol_api = _to_api_identifier(FUTURE_SYMBOL)
        long_total = 0
        short_total = 0
        for p in positions:
            sym = (getattr(p, 'symbol', None) or getattr(getattr(p, 'contract', None), 'symbol', None) or
                   (p.get('symbol') if isinstance(p, dict) else None))
            qty = getattr(p, 'quantity', None) or getattr(p, 'qty', None) or (p.get('quantity') or p.get('qty') if isinstance(p, dict) else None)
            if sym and (symbol_api in str(sym) or 'SIL' in str(sym)):
                try:
                    v = int(qty or 0)
                    if v > 0:
                        long_total += v
                    elif v < 0:
                        short_total += abs(v)  # 空头用正数表示手数
                except (TypeError, ValueError):
                    pass
        if long_total > 0 or short_total > 0:
            prev_long = int(current_position or 0)
            prev_short = int(current_short_position or 0)
            # 后台刚成交后的短窗口内可能回写滞后（例如本地已 2 手，后台瞬时仍报 1 手）。
            # 为避免误降再开到第 3 手，120 秒内若后台<本地，则保留本地更高值。
            lag_guard = False
            try:
                if (time.time() - float(_last_buy_success_ts or 0.0)) <= 120:
                    if long_total < prev_long:
                        long_total = prev_long
                        lag_guard = True
                    if short_total < prev_short:
                        short_total = prev_short
                        lag_guard = True
            except Exception:
                pass
            current_position = long_total
            current_short_position = short_total
            position_entry_times.clear()
            position_entry_prices.clear()
            active_take_profit_orders.clear()
            for i in range(long_total):
                position_entry_times[i] = time.time()
                position_entry_prices[i] = 0.0
            if lag_guard:
                logger.info("[DFX] 同步延迟保护生效：保留本地持仓 long=%s short=%s（后台回写滞后）", current_position, current_short_position)
            logger.info("[DFX] 已从后台同步持仓: 多头=%s 手, 空头=%s 手（账户 %s）", long_total, short_total, acc)
            print("⚠️ 已从后台同步持仓: 多头 %s 手, 空头 %s 手。若超过风控上限将拒绝新开仓。" % (long_total, short_total))
        else:
            # 多头空头均为 0：不可盲目 reset。OrderExecutor 下单后有延迟，sync 若 reset 会覆盖
            if current_position > 0 or current_short_position > 0:
                # 本地>0 但后台=0：可能 pending 买单未成交，保留本地计数，避免再买
                pass
            else:
                reset_demo_positions()
    except Exception as e:
        logger.warning("从后台同步持仓失败: %s；将重置为 0。若账户实际有仓请先手工处理。", e)
        print("⚠️ 同步持仓失败: %s；进程内已重置为 0。若实际有仓请核对。" % (e,))
        reset_demo_positions()


def refresh_period_analysis_background():
    """后台定期刷新时段分析（每天一次）"""
    if not time_period_strategy_instance:
        return
    
    import time
    while True:
        try:
            # 等待24小时（86400秒）
            time.sleep(86400)
            
            print("🔄 开始定期刷新时段分析...")
            time_period_strategy_instance.refresh_analysis(days=30)
            print("✅ 时段分析刷新完成")
        except Exception as e:
            print(f"⚠️ 时段分析刷新失败: {e}")
            # 如果失败，等待1小时后再试
            time.sleep(3600)

if __name__ == "__main__":
    # 检查是否运行测试
    if len(sys.argv) > 2 and sys.argv[2] == 'test':
        run_tests()
        exit(0)
    
    # 解析命令行参数
    count_type = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ('d', 'c') else 'd'
    strategy_type = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in ('backtest', 'llm', 'grid', 'boll', 'compare', 'large', 'huge', 'moe', 'moe_transformer', 'all') else 'all'
    
    # 验证API连接
    if not verify_api_connection():
        exit(1)
    
    # 启动前同步后台持仓，不假定从 0 开始（实盘/DEMO 重启后须与账户状态一致）
    sync_positions_from_backend()
    
    # 根据策略类型启动相应策略
    # 如果策略类型是moe或moe_transformer，使用TradingExecutor架构
    if strategy_type in ('moe', 'moe_transformer'):
        print("🚀 启动MOE策略（使用TradingExecutor架构）...")
        try:
            from src.strategies.strategy_factory import StrategyFactory
            from src.executor import MarketDataProvider, OrderExecutor, TradingExecutor
            import os
            import json
            
            # 加载策略配置
            config_path = os.path.join(_TIGER_ROOT, 'config', 'strategy_config.json')
            strategy_name = 'moe_transformer'
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    strategy_name = config.get('default_strategy', 'moe_transformer')
            
            # 从环境变量或命令行参数获取策略名称
            if len(sys.argv) > 2:
                strategy_name = sys.argv[2]
            elif os.getenv('TRADING_STRATEGY'):
                strategy_name = os.getenv('TRADING_STRATEGY')
            
            # 策略名称映射：moe -> moe_transformer
            if strategy_name == 'moe':
                strategy_name = 'moe_transformer'
            
            # 获取运行时长
            duration_hours = 20
            if len(sys.argv) > 3:
                try:
                    duration_hours = int(sys.argv[3])
                except:
                    pass
            elif os.getenv('RUN_DURATION_HOURS'):
                try:
                    duration_hours = int(os.getenv('RUN_DURATION_HOURS'))
                except:
                    pass
            
            print(f"📋 策略名称: {strategy_name}")
            print(f"⏱️  运行时长: {duration_hours} 小时")
            
            # 1. 创建策略
            strategy_config = {}
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    strategy_config = config.get('strategies', {}).get(strategy_name, {})
            
            strategy = StrategyFactory.create(
                strategy_name=strategy_name,
                model_path=strategy_config.get('model_path'),
                seq_length=strategy_config.get('seq_length', 500)
            )
            
            # 2. 创建数据提供者
            data_provider = MarketDataProvider(FUTURE_SYMBOL)
            
            # 3. 创建订单执行器。必须以 实际更新 current_position 的模块 为 risk_manager，否则 __main__ 与 src.tiger1 是不同模块→更新的 current_position 与风控读取的不是同一变量→风控永远看到 0→超买 52 手
            import sys
            from src import tiger1 as _risk_mod
            _main = sys.modules.get('__main__')
            _risk = _main if (_main is not None and callable(getattr(_main, 'check_risk_control', None))) else _risk_mod
            order_executor = OrderExecutor(_risk, state_fallback=_main if _main is not _risk else None)
            
            # 4. 创建交易执行器
            executor = TradingExecutor(
                strategy=strategy,
                data_provider=data_provider,
                order_executor=order_executor,
                config={
                    'confidence_threshold': 0.55,  # 提高阈值以提升胜率（目标>60%）：仅执行高置信度信号
                    'loop_interval': 5
                }
            )
            
            # 5. 运行交易循环
            executor.run_loop(duration_hours=duration_hours)
            
        except Exception as e:
            print(f"❌ MOE策略启动失败: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
    
    elif strategy_type == 'optimize':
        print("🚀 启动数据驱动模型优化...")
        # 初始化数据驱动优化器
        optimizer = data_driven_optimization.DataDrivenOptimizer()
        
        while True:
            try:
                # 运行分析和优化
                model_params, thresholds = optimizer.run_analysis_and_optimization()
                
                # 应用优化参数到模型
                print("🔄 应用优化参数到模型...")
                
                # 等待一段时间后再次运行分析
                print("⏰ 等待1小时后再次分析...")
                time.sleep(3600)
                
            except KeyboardInterrupt:
                print("🛑 程序被用户中断")
                break
            except Exception as e:
                print(f"❌ 数据驱动优化异常：{e}")
                import traceback
                traceback.print_exc()
                time.sleep(60)
    elif strategy_type == 'huge':
        print("🚀 启动超大Transformer交易策略...")
        # 初始化超大Transformer交易策略
        huge_strat = huge_transformer_strategy.HugeTransformerStrategy()
        
        while True:
            try:
                # 获取当前市场数据
                df_5m = get_kline_data([FUTURE_SYMBOL], '5min', count=GRID_PERIOD + 5)
                df_1m = get_kline_data([FUTURE_SYMBOL], '1min', count=GRID_PERIOD + 5)
                
                if df_5m.empty or df_1m.empty:
                    logger.debug("超大Transformer策略: 数据不足，跳过")
                    time.sleep(5)
                    continue

                # 计算技术指标
                inds = calculate_indicators(df_5m, df_1m)
                if '5m' not in inds or '1m' not in inds:
                    print("⚠️ 超大Transformer策略: 指标计算失败，跳过")
                    time.sleep(5)
                    continue

                # 获取关键指标
                price_current = inds['1m']['close']
                atr = inds['5m']['atr']
                rsi_1m = inds['1m']['rsi']
                rsi_5m = inds['5m']['rsi']

                # 使用硬编码的网格值
                grid_upper = price_current * 1.01  # 1% 上涨
                grid_lower = price_current * 0.99  # 1% 下跌

                # 计算缓冲区
                buffer = max(atr * 0.3, 0.0025)  # 用ATR的30%作为缓冲，最小值为0.0025
                threshold = grid_lower + buffer

                # 准备当前数据用于模型预测
                current_data = {
                    'price_current': price_current,
                    'grid_lower': grid_lower,
                    'grid_upper': grid_upper,
                    'atr': atr,
                    'rsi_1m': rsi_1m,
                    'rsi_5m': rsi_5m,
                    'buffer': buffer,
                    'threshold': threshold,
                    'near_lower': price_current <= threshold,
                    'rsi_ok': rsi_1m < 30 or (rsi_5m > 45 and rsi_5m < 55)  # 示例条件
                }

                # 使用超大Transformer模型预测
                action, confidence = huge_strat.predict_action(current_data)
                action_map = {0: "不操作", 1: "买入", 2: "卖出"}

                print(f"🧠 超大Transformer预测: {action_map[action]}, 置信度: {confidence:.3f}")
                print(f"📊 比较 | 价={price_current:.3f}, ATR={atr:.3f}, 网格=[{grid_lower:.3f},{grid_upper:.3f}]")
                print(f"   条件详情: BUFFER={buffer:.3f}, 近轨={current_data['near_lower']}, RSI_OK={current_data['rsi_ok']}")

                # 根据模型预测结果执行交易（这里只是示例，实际可以根据置信度调整）
                if action != 0 and confidence > 0.7:  # 有操作且置信度高
                    if action == 1:  # 买入
                        print(f"✅ 执行买入操作 at {price_current:.3f}")
                    elif action == 2:  # 卖出
                        print(f"✅ 执行卖出操作 at {price_current:.3f}")
                
                time.sleep(5)
                
            except KeyboardInterrupt:
                print("🛑 程序被用户中断")
                break
            except Exception as e:
                print(f"❌ 超大Transformer策略异常：{e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
    elif strategy_type == 'rl':
        print("🚀 启动强化学习交易策略...")
        # 初始化强化学习交易策略
        rl_strat = rl_trading_strategy.RLTradingStrategy()
        
        # 保存前一个状态用于计算奖励
        prev_data = None
        
        while True:
            try:
                # 获取当前市场数据
                df_5m = get_kline_data([FUTURE_SYMBOL], '5min', count=GRID_PERIOD + 5)
                df_1m = get_kline_data([FUTURE_SYMBOL], '1min', count=GRID_PERIOD + 5)
                
                if df_5m.empty or df_1m.empty:
                    print("⚠️ 强化学习策略: 数据不足，跳过")
                    time.sleep(5)
                    continue

                # 计算技术指标
                inds = calculate_indicators(df_5m, df_1m)
                if '5m' not in inds or '1m' not in inds:
                    logger.debug("强化学习策略: 指标计算失败，跳过")
                    time.sleep(5)
                    continue

                # 获取关键指标
                price_current = inds['1m']['close']
                atr = inds['5m']['atr']
                rsi_1m = inds['1m']['rsi']
                rsi_5m = inds['5m']['rsi']

                # 使用硬编码的网格值
                grid_upper = price_current * 1.01  # 1% 上涨
                grid_lower = price_current * 0.99  # 1% 下跌

                # 计算缓冲区
                buffer = max(atr * 0.3, 0.0025)  # 用ATR的30%作为缓冲，最小值为0.0025
                threshold = grid_lower + buffer

                # 准备当前数据用于模型预测
                current_data = {
                    'price_current': price_current,
                    'grid_lower': grid_lower,
                    'grid_upper': grid_upper,
                    'atr': atr,
                    'rsi_1m': rsi_1m,
                    'rsi_5m': rsi_5m,
                    'buffer': buffer,
                    'threshold': threshold,
                    'near_lower': price_current <= threshold,
                    'rsi_ok': rsi_1m < 30 or (rsi_5m > 45 and rsi_5m < 55)  # 示例条件
                }

                # 使用强化学习模型预测
                action, confidence = rl_strat.predict_action(current_data)
                action_map = {0: "持有", 1: "买入", 2: "卖出"}

                print(f"🧠 RL模型预测: {action_map[action]}, 置信度: {confidence:.3f}")
                print(f"📊 比较 | 价={price_current:.3f}, ATR={atr:.3f}, 网格=[{grid_lower:.3f},{grid_upper:.3f}]")
                print(f"   条件详情: BUFFER={buffer:.3f}, 近轨={current_data['near_lower']}, RSI_OK={current_data['rsi_ok']}")

                # 如果有前一个状态，计算奖励并存储经验
                if prev_data is not None:
                    reward = rl_strat.compute_reward(action, current_data, prev_data)
                    state = rl_strat.prepare_features(prev_data)
                    next_state = rl_strat.prepare_features(current_data)
                    rl_strat.remember(state, action, reward, next_state, False)
                    rl_strat.log_performance(action, action, reward)

                # 更新prev_data为当前数据
                prev_data = current_data.copy()

                # 根据模型预测结果执行交易（这里只是示例，实际可以根据置信度调整）
                if action != 0 and confidence > 0.7:  # 有操作且置信度高
                    if action == 1:  # 买入
                        print(f"✅ 执行买入操作 at {price_current:.3f}")
                    elif action == 2:  # 卖出
                        print(f"✅ 执行卖出操作 at {price_current:.3f}")
                
                time.sleep(5)
                
            except KeyboardInterrupt:
                print("🛑 程序被用户中断")
                break
            except Exception as e:
                print(f"❌ 强化学习策略异常：{e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
    elif strategy_type == 'enhanced_trans':
        print("🚀 启动增强型Transformer策略...")
        # 初始化增强型Transformer策略
        enh_trans_strat = enhanced_transformer_strategy.EnhancedTransformerStrategy()
        
        while True:
            try:
                # 获取当前市场数据
                df_5m = get_kline_data([FUTURE_SYMBOL], '5min', count=GRID_PERIOD + 5)
                df_1m = get_kline_data([FUTURE_SYMBOL], '1min', count=GRID_PERIOD + 5)
                
                if df_5m.empty or df_1m.empty:
                    print("⚠️ 增强型Transformer策略: 数据不足，跳过")
                    time.sleep(5)
                    continue

                # 计算技术指标
                inds = calculate_indicators(df_5m, df_1m)
                if '5m' not in inds or '1m' not in inds:
                    logger.debug("增强型Transformer策略: 指标计算失败，跳过")
                    time.sleep(5)
                    continue

                # 获取关键指标
                price_current = inds['1m']['close']
                atr = inds['5m']['atr']
                rsi_1m = inds['1m']['rsi']
                rsi_5m = inds['5m']['rsi']

                # 使用硬编码的网格值
                grid_upper = price_current * 1.01  # 1% 上涨
                grid_lower = price_current * 0.99  # 1% 下跌

                # 计算缓冲区
                buffer = max(atr * 0.3, 0.0025)  # 用ATR的30%作为缓冲，最小值为0.0025
                threshold = grid_lower + buffer

                # 准备当前数据用于模型预测
                current_data = {
                    'price_current': price_current,
                    'grid_lower': grid_lower,
                    'grid_upper': grid_upper,
                    'atr': atr,
                    'rsi_1m': rsi_1m,
                    'rsi_5m': rsi_5m,
                    'buffer': buffer,
                    'threshold': threshold,
                    'near_lower': price_current <= threshold,
                    'rsi_ok': rsi_1m < 30 or (rsi_5m > 45 and rsi_5m < 55)  # 示例条件
                }

                # 使用增强型Transformer模型预测
                action, confidence = enh_trans_strat.predict_action(current_data)
                action_map = {0: "不操作", 1: "买入", 2: "卖出"}

                print(f"🧠 增强型Transformer预测: {action_map[action]}, 置信度: {confidence:.3f}")
                print(f"📊 比较 | 价={price_current:.3f}, ATR={atr:.3f}, 网格=[{grid_lower:.3f},{grid_upper:.3f}]")
                print(f"   条件详情: BUFFER={buffer:.3f}, 近轨={current_data['near_lower']}, RSI_OK={current_data['rsi_ok']}")

                # 根据模型预测结果执行交易（这里只是示例，实际可以根据置信度调整）
                if action != 0 and confidence > 0.7:  # 有操作且置信度高
                    if action == 1:  # 买入
                        print(f"✅ 执行买入操作 at {price_current:.3f}")
                    elif action == 2:  # 卖出
                        print(f"✅ 执行卖出操作 at {price_current:.3f}")
                
                time.sleep(5)
                
            except KeyboardInterrupt:
                print("🛑 程序被用户中断")
                break
            except Exception as e:
                print(f"❌ 增强型Transformer策略异常：{e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
    elif strategy_type == 'llm':
        print("🚀 启动LLM交易策略（多时间尺度模型，序列长度30，集成Tick数据）...")
        # 初始化LLM交易策略（多时间尺度，序列长度30，启用收益率预测）
        llm_strat = llm_strategy.LLMTradingStrategy(mode='hybrid', predict_profit=True)
        llm_strat._seq_length = 30  # 使用30个时间步的序列长度
        print(f"✅ LLM策略已初始化，模式: hybrid, 序列长度: {llm_strat._seq_length}, 特征维度: 46维（多时间尺度）")
        
        # 历史数据缓存（用于序列预测）
        historical_data_cache = []
        
        while True:
            try:
                # 获取K线数据（用于技术指标）
                df_5m = get_kline_data([FUTURE_SYMBOL], '5min', count=GRID_PERIOD + 5)
                df_1m = get_kline_data([FUTURE_SYMBOL], '1min', count=max(GRID_PERIOD + 5, llm_strat._seq_length + 10))
                
                # 获取Tick数据（用于精确入场）
                df_tick = get_tick_data([FUTURE_SYMBOL], count=100)
                
                if df_5m.empty or df_1m.empty:
                    logger.debug("LLM策略: K线数据不足，跳过")
                    time.sleep(5)
                    continue

                # 计算技术指标
                inds = calculate_indicators(df_5m, df_1m)
                if '5m' not in inds or '1m' not in inds:
                    logger.debug("LLM策略: 指标计算失败，跳过")
                    time.sleep(5)
                    continue

                # 获取关键指标
                price_current = inds['1m']['close']
                atr = inds['5m']['atr']
                rsi_1m = inds['1m']['rsi']
                rsi_5m = inds['5m']['rsi']
                
                # 获取Tick数据的最新价格（更精确）
                tick_price = price_current
                if not df_tick.empty:
                    latest_tick = df_tick.iloc[-1]
                    tick_price = latest_tick['price'] if 'price' in latest_tick else price_current
                    print(f"📊 Tick价格: {tick_price:.3f} (K线价格: {price_current:.3f})")

                # 使用时段自适应网格参数
                trend = judge_market_trend(inds)
                adjust_grid_interval(trend, inds)
                
                # 使用调整后的网格参数
                grid_upper_val = grid_upper
                grid_lower_val = grid_lower

                # 计算缓冲区
                buffer = max(atr * 0.3, 0.0025)
                threshold = grid_lower_val + buffer

                # 准备当前数据用于模型预测（包含Tick数据）
                current_data = {
                    'price_current': tick_price,  # 使用Tick价格
                    'grid_lower': grid_lower_val,
                    'grid_upper': grid_upper_val,
                    'atr': atr,
                    'rsi_1m': rsi_1m,
                    'rsi_5m': rsi_5m,
                    'buffer': buffer,
                    'threshold': threshold,
                    'near_lower': tick_price <= threshold,
                    'rsi_ok': rsi_1m < 30 or (rsi_5m > 45 and rsi_5m < 55),
                    'tick_price': tick_price,  # Tick价格
                    'kline_price': price_current  # K线价格
                }
                
                # 更新历史数据缓存（用于序列预测）
                historical_data_cache.append(current_data)
                # 只保留最近足够的数据
                max_cache_size = llm_strat._seq_length + 20
                if len(historical_data_cache) > max_cache_size:
                    historical_data_cache = historical_data_cache[-max_cache_size:]
                
                # 设置历史数据到策略中（用于序列预测）
                if len(historical_data_cache) >= llm_strat._seq_length:
                    # 转换为DataFrame格式
                    hist_df = pd.DataFrame(historical_data_cache)
                    llm_strat._historical_data = hist_df

                # 使用LLM模型预测（会自动使用序列数据如果可用）
                prediction_result = llm_strat.predict_action(current_data)
                
                # 处理不同的返回值格式
                if isinstance(prediction_result, tuple):
                    if len(prediction_result) == 2:
                        action, confidence = prediction_result
                        grid_adjustment = 1.0
                    elif len(prediction_result) == 3:
                        action, confidence, grid_adjustment = prediction_result
                    elif len(prediction_result) == 4:
                        action, confidence, profit, grid_adjustment = prediction_result
                    else:
                        action = prediction_result[0]
                        confidence = prediction_result[1] if len(prediction_result) > 1 else 0.5
                        grid_adjustment = prediction_result[2] if len(prediction_result) > 2 else 1.0
                else:
                    # 如果返回单个值，假设是action
                    action = prediction_result
                    confidence = 0.5
                    grid_adjustment = 1.0
                
                action_map = {0: "不操作", 1: "买入", 2: "卖出"}
                
                # 应用网格调整系数
                grid_step_base = grid_upper_val - grid_lower_val  # 简化的基础网格间距
                grid_step_adjusted = grid_step_base * grid_adjustment
                grid_upper_adjusted = tick_price + grid_step_adjusted / 2
                grid_lower_adjusted = tick_price - grid_step_adjusted / 2

                print(f"🧠 LLM模型预测（序列长度{llm_strat._seq_length}）: {action_map[action]}, 置信度: {confidence:.3f}, 网格调整: {grid_adjustment:.3f}")
                print(f"📊 价格 | Tick={tick_price:.3f}, K线={price_current:.3f}, ATR={atr:.3f}")
                print(f"📊 网格 | [{grid_lower_val:.3f}, {grid_upper_val:.3f}], 阈值={threshold:.3f}")
                print(f"   条件: 近轨={current_data['near_lower']}, RSI_OK={current_data['rsi_ok']}")

                # 根据模型预测结果执行交易
                # 如果置信度低，使用规则策略作为后备
                use_llm_prediction = (action != 0 and confidence > 0.6)
                use_rule_strategy = (confidence <= 0.6)  # 置信度低时使用规则策略
                
                if use_llm_prediction:
                    # LLM模型预测（高置信度）：买入必须带止损+止盈，与 grid_trading_strategy_pro1 一致
                    if action == 1:  # 买入
                        if check_risk_control(tick_price, 'BUY'):
                            stop_loss_price, projected_loss = compute_stop_loss(tick_price, atr, grid_lower_val)
                            tp_offset = max(TAKE_PROFIT_ATR_OFFSET * (atr if atr else 0), TAKE_PROFIT_MIN_OFFSET)
                            take_profit_price = max(tick_price + getattr(sys.modules[__name__], 'MIN_TICK', 0.01),
                                                    (grid_upper_val - tp_offset) if grid_upper_val is not None else tick_price + 0.02)
                            print(f"✅ [LLM预测] 执行买入操作 | 价格={tick_price:.3f}, 止损={stop_loss_price:.3f}, 止盈={take_profit_price:.3f}")
                            place_tiger_order('BUY', 1, tick_price, stop_loss_price, take_profit_price)
                            try:
                                place_take_profit_order('BUY', 1, take_profit_price)
                            except Exception:
                                pass
                        else:
                            logger.debug("风控阻止买入")
                    elif action == 2:  # 卖出
                        if current_position > 0:
                            print(f"✅ [LLM预测] 执行卖出操作 | 价格={tick_price:.3f}")
                            place_tiger_order('SELL', 1, tick_price)
                        else:
                            logger.debug("无持仓，无法卖出")
                elif use_rule_strategy:
                    # 规则策略作为后备（当LLM置信度低时）
                    print(f"📊 [规则策略] LLM置信度低({confidence:.3f})，使用规则策略")
                    # 使用布林带策略逻辑
                    near_lower = current_data.get('near_lower', False)
                    rsi_ok = current_data.get('rsi_ok', False)
                    
                    # 买入条件：接近下轨 + RSI超卖；必须带止损+止盈
                    if current_position == 0 and near_lower and rsi_ok:
                        if check_risk_control(tick_price, 'BUY'):
                            stop_loss_price, projected_loss = compute_stop_loss(tick_price, atr, grid_lower_val)
                            tp_offset = max(TAKE_PROFIT_ATR_OFFSET * (atr if atr else 0), TAKE_PROFIT_MIN_OFFSET)
                            take_profit_price = max(tick_price + getattr(sys.modules[__name__], 'MIN_TICK', 0.01),
                                                    (grid_upper_val - tp_offset) if grid_upper_val is not None else tick_price + 0.02)
                            print(f"✅ [规则策略] 执行买入操作 | 价格={tick_price:.3f}, 止损={stop_loss_price:.3f}, 止盈={take_profit_price:.3f}")
                            place_tiger_order('BUY', 1, tick_price, stop_loss_price, take_profit_price)
                            try:
                                place_take_profit_order('BUY', 1, take_profit_price)
                            except Exception:
                                pass
                        else:
                            logger.debug("风控阻止买入")
                    
                    # 卖出条件：持有仓位 + 价格达到中轨
                    if current_position > 0 and price_current >= inds['1m'].get('boll_mid', price_current):
                        print(f"✅ [规则策略] 执行卖出操作 | 价格={tick_price:.3f}")
                        place_tiger_order('SELL', 1, tick_price)
                
                time.sleep(5)
                
            except KeyboardInterrupt:
                print("🛑 程序被用户中断")
                break
            except Exception as e:
                print(f"❌ LLM策略异常：{e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
    elif strategy_type == 'large':
        print("🚀 启动大模型交易策略...")
        # 初始化大模型交易策略
        large_strat = large_model_strategy.LargeModelStrategy()
        
        while True:
            try:
                # 获取当前市场数据
                df_5m = get_kline_data([FUTURE_SYMBOL], '5min', count=GRID_PERIOD + 5)
                df_1m = get_kline_data([FUTURE_SYMBOL], '1min', count=GRID_PERIOD + 5)
                
                if df_5m.empty or df_1m.empty:
                    print("⚠️ 大模型策略: 数据不足，跳过")
                    time.sleep(5)
                    continue

                # 计算技术指标
                inds = calculate_indicators(df_5m, df_1m)
                if '5m' not in inds or '1m' not in inds:
                    logger.debug("大模型策略: 指标计算失败，跳过")
                    time.sleep(5)
                    continue

                # 获取关键指标
                price_current = inds['1m']['close']
                atr = inds['5m']['atr']
                rsi_1m = inds['1m']['rsi']
                rsi_5m = inds['5m']['rsi']

                # 使用硬编码的网格值
                grid_upper = price_current * 1.01  # 1% 上涨
                grid_lower = price_current * 0.99  # 1% 下跌

                # 计算缓冲区
                buffer = max(atr * 0.3, 0.0025)  # 用ATR的30%作为缓冲，最小值为0.0025
                threshold = grid_lower + buffer

                # 准备当前数据用于模型预测
                current_data = {
                    'price_current': price_current,
                    'grid_lower': grid_lower,
                    'grid_upper': grid_upper,
                    'atr': atr,
                    'rsi_1m': rsi_1m,
                    'rsi_5m': rsi_5m,
                    'buffer': buffer,
                    'threshold': threshold,
                    'near_lower': price_current <= threshold,
                    'rsi_ok': rsi_1m < 30 or (rsi_5m > 45 and rsi_5m < 55)  # 示例条件
                }

                # 使用大模型预测
                action, confidence = large_strat.predict_action(current_data)
                action_map = {0: "不操作", 1: "买入", 2: "卖出"}

                print(f"🧠 大模型预测: {action_map[action]}, 置信度: {confidence:.3f}")
                print(f"📊 比较 | 价={price_current:.3f}, ATR={atr:.3f}, 网格=[{grid_lower:.3f},{grid_upper:.3f}]")
                print(f"   条件详情: BUFFER={buffer:.3f}, 近轨={current_data['near_lower']}, RSI_OK={current_data['rsi_ok']}")

                # 根据模型预测结果执行交易（这里只是示例，实际可以根据置信度调整）
                if action != 0 and confidence > 0.7:  # 有操作且置信度高
                    if action == 1:  # 买入
                        print(f"✅ 执行买入操作 at {price_current:.3f}")
                    elif action == 2:  # 卖出
                        print(f"✅ 执行卖出操作 at {price_current:.3f}")
                
                time.sleep(5)
                
            except KeyboardInterrupt:
                print("🛑 程序被用户中断")
                break
            except Exception as e:
                print(f"❌ 大模型策略异常：{e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
    # 启动时段分析后台刷新线程（如果可用）
    if time_period_strategy_instance:
        refresh_thread = threading.Thread(target=refresh_period_analysis_background, daemon=True)
        refresh_thread.start()
        print("✅ 时段分析后台刷新线程已启动（每24小时刷新一次）")
    
    if strategy_type == 'grid':
        print("🚀 启动网格策略...")
        # 可选：主循环内并发手工订单（同一线程，与自动策略共用 current_position 等变量）
        manual_monitor = None
        if os.getenv("USE_MANUAL_IN_MAIN_LOOP", "").strip().lower() in ("1", "true", "yes"):
            try:
                from src.manual_order_mode import ManualOrderMonitor, MANUAL_ORDERS_FILE, MANUAL_ORDERS_STATUS_FILE
                manual_monitor = ManualOrderMonitor(orders_file=MANUAL_ORDERS_FILE, status_file=MANUAL_ORDERS_STATUS_FILE)
                print("✅ 手工订单已接入主循环（每轮策略后轮询 manual_orders.json）")
            except Exception as e:
                print(f"⚠️ 手工订单未接入主循环: {e}")
        while True:
            try:
                grid_trading_strategy_pro1()
                # 主循环内手工订单：同一线程，共用 current_position/open_orders
                if manual_monitor is not None:
                    try:
                        df_1m = get_kline_data([FUTURE_SYMBOL], '1min', count=2)
                        if df_1m is not None and not df_1m.empty:
                            row = df_1m.iloc[-1]
                            o, h, l, c = row.get('open', row.get('Open')), row.get('high', row.get('High')), row.get('low', row.get('Low')), row.get('close', row.get('Close'))
                            manual_monitor.on_price_update(float(o), float(h), float(l), float(c), 0)
                    except Exception as e:
                        print(f"⚠️ 手工订单本轮更新跳过: {e}")
                time.sleep(5)
            except KeyboardInterrupt:
                print("🛑 程序被用户中断")
                break
            except Exception as e:
                print(f"❌ 程序异常：{e}")
                time.sleep(5)
    elif strategy_type == 'boll':
        print("🚀 启动BOLL策略...")
        while True:
            try:
                boll1m_grid_strategy()
                time.sleep(5)
            except KeyboardInterrupt:
                print("🛑 程序被用户中断")
                break
            except Exception as e:
                print(f"❌ 程序异常：{e}")
                time.sleep(5)
    else:  # 默认运行所有策略
        print("🚀 启动网格处理）...")
        while True:
            try:
                # Run all strategies concurrently
                threads = []
                
                # Start grid trading strategy in a thread
                t1 = threading.Thread(target=grid_trading_strategy_pro1)
                threads.append(t1)
                
                # Start BOLL strategy in a thread  
                t2 = threading.Thread(target=boll1m_grid_strategy)
                threads.append(t2)
                
                # Start all threads
                for t in threads:
                    t.start()
                
                # Wait for all threads to complete
                for t in threads:
                    t.join()
                    
                time.sleep(5)  # Wait 5 seconds before next iteration
                
            except KeyboardInterrupt:
                print("🛑 程序被用户中断")
                break
            except Exception as e:
                print(f"❌ 程序异常：{e}")
                time.sleep(5)
                
    print("✅ 程序结束")

def compute_stop_loss(price: float, atr_value: float, grid_lower_val: float):
    """计算止损价格和预期损失"""
    # 基于ATR的止损：使用ATR倍数，但不低于ATR下限
    atr_based_stop = max(STOP_LOSS_ATR_FLOOR, atr_value * STOP_LOSS_MULTIPLIER)  # 至少0.25的ATR保护
    
    # 结构性止损：基于网格下轨
    structural_stop = max(0.05, price - grid_lower_val)  # 网格下轨基础上的安全距离
    
    # 单笔最大亏损限制
    max_loss_per_unit = 0.1  # 最大单位亏损限制
    
    # 计算综合止损
    stop_distance = max(atr_based_stop, structural_stop, 0.05)  # 至少0.05的止损距离
    
    # 计算止损价格
    stop_loss_price = price - stop_distance
    
    # 计算预期损失
    projected_loss = stop_distance * FUTURE_MULTIPLIER
    
    logger.debug("止损计算详情: 价格=%.3f ATR=%.3f 下轨=%.3f 止损价=%.3f 预期损失=%.3f",
                 price, atr_value, grid_lower_val, stop_loss_price, projected_loss)
    
    # 返回止损价格和预期损失
    return stop_loss_price, projected_loss


def get_contract_expiry_risk(symbol=None):
    """获取当前合约到期/通知日风险，供风控使用。优先用 API 的到期日，否则用合约代码推算。

    Returns:
        dict: expire_date (date), days_to_expiry (int), is_near_expiry (bool),
              is_notice_period (bool), message (str，含展期提示)
    """
    today = date.today()
    sym = (symbol or FUTURE_SYMBOL or "").strip()
    if not sym:
        return {"expire_date": today + timedelta(days=90), "days_to_expiry": 90,
                "is_near_expiry": False, "is_notice_period": False, "message": ""}
    try:
        api_id = _to_api_identifier(sym)
    except Exception:
        api_id = sym
    expire_date = None
    try:
        brief = get_future_brief_info(api_id)
        if brief and isinstance(brief.get("expire_date"), date):
            expire_date = brief["expire_date"]
    except Exception as e:
        logger.debug("get_contract_expiry_risk: API 获取到期日失败 %s，尝试解析合约代码", e)
    if expire_date is None and len(api_id) >= 6:
        try:
            from src.futures_contract_manager import FuturesContractManager
            base = "".join(c for c in api_id if c.isalpha()) or "SIL"
            mgr = FuturesContractManager(base)
            info = mgr.parse_contract(api_id)
            expire_date = date(info["full_year"], info["month"], 25)
        except Exception as e:
            logger.debug("get_contract_expiry_risk: 解析合约失败 %s", e)
    if expire_date is None:
        expire_date = today + timedelta(days=90)
    days_to_expiry = (expire_date - today).days
    is_near_expiry = days_to_expiry <= NOTICE_DAYS
    is_notice_period = is_near_expiry
    message = ""
    if days_to_expiry <= 0:
        message = f"合约已到期（{expire_date}），请切换至下一合约并考虑展期。"
    elif is_near_expiry:
        message = f"合约即将到期（{expire_date}，剩余{days_to_expiry}天），请注意通知日风险，考虑展期至下一合约。"
    return {
        "expire_date": expire_date,
        "days_to_expiry": days_to_expiry,
        "is_near_expiry": is_near_expiry,
        "is_notice_period": is_notice_period,
        "message": message,
    }


def check_risk_control(price, side):
    """Basic risk control checks used by strategies and tests.

    Returns True if a trade of given `side` at `price` is allowed under
    simple rules (max position, daily loss, sane price).
    """
    global today, daily_loss, current_position

    # reset daily loss when date changes
    try:
        if today != datetime.now().date():
            today = datetime.now().date()
            daily_loss = 0
    except Exception:
        pass

    # basic validation of inputs（先判 None 再格式化打印，避免 TypeError）
    try:
        if price is None:
            logger.warning("风控检查失败: 价格为None")
            return False
        logger.debug("风控检查: 价格=%.3f 方向=%s 持仓=%s 当日亏损=%.2f", price, side, current_position, daily_loss)
        if not (isinstance(price, (int, float))):
            logger.warning("风控检查失败: 价格类型错误 (%s)", type(price))
            return False
        if math.isinf(price) or math.isnan(price):
            logger.warning("风控检查失败: 价格为无穷大或NaN")
            return False
        if price <= 0:
            logger.warning("风控检查失败: 价格小于等于0 (%s)", price)
            return False
    except Exception:
        logger.warning("风控检查异常: 价格验证失败")
        return False

    if side not in ('BUY', 'SELL'):
        logger.warning("风控检查失败: 交易方向错误 (%s)", side)
        return False

    # If we've already hit daily loss limit, block further buys
    if daily_loss >= DAILY_LOSS_LIMIT:
        logger.warning("风控检查失败: 当日亏损已达上限 (当前:%.2f 上限:%s)", daily_loss, DAILY_LOSS_LIMIT)
        return False

    # Prevent buys beyond max position；DEMO 一律用 DEMO_MAX_POSITION（时段自适应曾把 GRID_MAX_POSITION 改为 8/10，导致账户 8 手超标）
    effective_max = DEMO_MAX_POSITION if RUN_ENV == 'sandbox' else GRID_MAX_POSITION
    if side == 'BUY' and current_position >= effective_max:
        logger.warning("风控检查失败: 持仓已达上限 (当前:%s 上限:%s)", current_position, effective_max)
        return False

    # conservative per-trade loss check: estimate stop loss and projected loss
    try:
        stop_price, proj_loss = compute_stop_loss(price, atr_5m if atr_5m is not None else 0, grid_lower)
        if proj_loss is None:
            logger.warning("风控检查失败: 预期损失为None")
            return False
        if proj_loss > SINGLE_TRADE_LOSS or proj_loss > MAX_SINGLE_LOSS:
            logger.warning("风控检查失败: 单笔预期损失超限 (当前:%.2f 单笔上限:%s 总上限:%s)", proj_loss, SINGLE_TRADE_LOSS, MAX_SINGLE_LOSS)
            return False
        logger.debug("单笔损失检查通过: 预期损失=%.2f 阈值=%.2f", proj_loss, min(SINGLE_TRADE_LOSS, MAX_SINGLE_LOSS))
    except Exception:
        # if estimation fails, be conservative and allow None/False depending on tests
        logger.warning("损失估算失败，保守拒绝交易")
        return False

    # 合约时间限制：到期/通知日风险，API 能提供则用 API 到期日，否则按合约代码推算
    try:
        expiry_risk = get_contract_expiry_risk()
        days_to_expiry = expiry_risk.get("days_to_expiry", 999)
        is_notice = expiry_risk.get("is_notice_period", False)
        msg = expiry_risk.get("message", "")
        if days_to_expiry <= EXPIRY_BLOCK_DAYS and side == "BUY":
            logger.warning("风控检查失败: 合约即将到期（剩余 %s 天），禁止新开多单，请考虑展期。%s", days_to_expiry, msg)
            return False
        if days_to_expiry <= 1 and msg:
            logger.warning("【通知日/到期日风险】%s 请立即考虑展期或平仓。", msg)
        elif is_notice and msg:
            logger.warning("[DFX] 合约通知期风险: %s", msg)
    except Exception as e:
        logger.debug("合约到期检查跳过: %s", e)

    logger.debug("风控检查通过: 价格=%.3f 方向=%s", price, side)
    return True  # This is the actual end of the function


# FUTURE_TICK_SIZE/MIN_TICK 已移至文件顶部
FUTURE_EXPIRE_DATE = '2026-05-28'  # 合约到期日兜底（当前 SIL2605；API 能获取时以 API 为准）

# 策略参数
price_current = 0
rsi_1m = 0
rsi_5m = 0
buffer = 0
threshold = 0
active_positions = {}
pending_orders = {}

