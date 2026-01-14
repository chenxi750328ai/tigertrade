
import sys
import pandas as pd


from tigeropen.common.consts import (Language,        # 语言
                                Market,           # 市场
                                BarPeriod,        # k线周期
                                QuoteRight)       # 复权类型
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.common.util.signature_utils import read_private_key
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import TigerOpenClientConfig

import time
import os
import hmac
import hashlib
import json
import numpy as np
import pandas as pd
import talib
from datetime import datetime, timedelta
from tigeropen.common.consts import Currency
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.trade.trade_client import TradeClient
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import logging

# module logger
logger = logging.getLogger(__name__)


# Read command-line mode when running as a script, but be import-safe for tests
count_type = sys.argv[1] if len(sys.argv) > 1 else 'd'

client_config = None
quote_client = None
trade_client = None

# Only try to instantiate real client objects when running with explicit args
if len(sys.argv) > 1:
    if count_type == 'd':
        try:
            client_config = TigerOpenClientConfig(props_path='./openapicfg_dem')
            print("demo count\r\n")
        except Exception:
            client_config = None
    elif count_type == 'c':
        try:
            client_config = TigerOpenClientConfig(props_path='./openapicfg_com')
            print("combine count\r\n")
        except Exception:
            client_config = None
    else:
        print(f"错误：不支持的参数 '{count_type}'，仅支持 d 或 c")
        # When running as a script we will exit later in main; do not sys.exit on import
        client_config = None

# Try to build clients if we have a config; fail gracefully for import-time safety
if client_config is not None:
    try:
        print(client_config.account, client_config.tiger_id)
        quote_client = QuoteClient(client_config)  # 行情客户端
        trade_client = TradeClient(client_config)  # 交易客户端
    except Exception:
        quote_client = None
        trade_client = None
# anothor method 
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

# 合约配置（SIL2603：COMEX白银2026年3月期货）
# 老虎证券期货合约格式：{品种}.{交易所}.{到期月}，需确认实际合约代码
FUTURE_SYMBOL = "SIL.COMEX.202603"
FUTURE_CURRENCY = Currency.USD
FUTURE_MULTIPLIER = 1000  # 白银期货每手1000盎司

# 网格策略核心参数（匹配之前讨论的规则）
GRID_MAX_POSITION = 3          # 最大持仓手数
GRID_ATR_PERIOD = 14           # ATR计算周期
GRID_BOLL_PERIOD = 20          # BOLL带周期
GRID_BOLL_STD = 2              # BOLL标准差
GRID_RSI_PERIOD_1M = 14        # 1分钟RSI周期
GRID_RSI_PERIOD_5M = 14        # 5分钟RSI周期

# 风控参数（6万美元账户适配）
DAILY_LOSS_LIMIT = 600         # 日亏损上限（美元）
SINGLE_TRADE_LOSS = 180        # 单笔最大亏损（美元）
STOP_LOSS_MULTIPLIER = 1.0     # 止损倍数（ATR）
MIN_KLINES = 10                 # 最少K线条数阈值（用于get_kline_data）

# 行情判断阈值
BOLL_DIVERGENCE_THRESHOLD = 0.2  # BOLL发散阈值（轨道间距扩大≥20%）
ATR_AMPLIFICATION_THRESHOLD = 0.3 # ATR放大≥30%判定波动加剧

# 策略全局变量
current_position = 0           # 当前持仓手数
daily_loss = 0                 # 当日累计亏损
grid_upper = 0                 # 网格上轨
grid_lower = 0                 # 网格下轨
last_boll_width = 0            # 上一次BOLL轨道间距
atr_5m = 0                     # 5分钟ATR值
is_boll_divergence = False     # 是否BOLL发散

# 运行环境标识（用于日志/模拟下单提示），以及今日日期用于每日亏损重置
RUN_ENV = 'sandbox' if count_type == 'd' else 'production'
today = datetime.now().date()

# ====================== 核心工具函数 ======================
def get_timestamp():
    """生成API签名所需的时间戳"""
    return int(time.time() * 1000)

def verify_api_connection():
    """验证API连接（使用官方标准方法get_account_info）"""
    try:
        # 调用API查询股票行情
        stock_price = quote_client.get_stock_briefs(['00700'])

        # 查询行情函数会返回一个包含当前行情快照的pandas.DataFrame对象，见返回示例。具体字段含义参见get_stock_briefs方法说明
        print(stock_price)

        exchanges = quote_client.get_future_exchanges()
        # 打印第一个交易所的代码，名称，时区
        for exchange1 in  exchanges.iloc :
            print(f'code: {exchange1.code}, name: {exchange1.name}, zone: {exchange1.zone}')


        contracts = quote_client.get_future_contracts('COMEX')

        # 将合约代码设置为pandas DataFrame 索引，并查询字段
        contract1 = contracts.set_index('contract_code').loc['SIL2603']
        print(contract1.name)  # 合约名称
        print(contract1.multiplier)  # 合约乘数
        print(contract1.last_trading_date)  # 最后交易日

        contracts = quote_client.get_all_future_contracts('SIL')
        print(contracts)

        contract = quote_client.get_current_future_contract('SIL')
        print(contract)

        permissions = quote_client.get_quote_permission()
        print(permissions)

        klines = quote_client.get_future_brief(['SIL2603'])
            
        print(klines.head().to_string())


        klines = quote_client.get_future_bars(
            ['SIL2603'],
            BarPeriod.ONE_MINUTE,
            -1,
            -1,
            2,
            None)

        print(klines.head().to_string())

        #place_tiger_order('BUY', 1, 91.63, 90)
        #place_tiger_order('SELL', 1, 91.63, 90)

        return True
    except Exception as e:
        # 通用异常捕获，输出详细错误
        error_msg = str(e)
        print(f"❌ {count_type} 环境连接失败：{error_msg}")
        return False

def get_future_brief_info(symbol):
    """调用QuoteClient.get_future_brief获取期货合约概要信息"""
    try:
        future_brief_list = quote_client.get_future_brief(identifiers=[symbol])
        if not future_brief_list:
            raise Exception(f"未获取到 {symbol} 的概要信息，将使用默认参数（乘数1000）")
        
        # 提取合约信息
        future_brief = future_brief_list[0]
        global FUTURE_MULTIPLIER, FUTURE_TICK_SIZE, FUTURE_EXPIRY_DATE
        FUTURE_MULTIPLIER = future_brief.multiplier if future_brief.multiplier else 1000
        FUTURE_TICK_SIZE = future_brief.tick_size if future_brief.tick_size else 0.01
        FUTURE_EXPIRY_DATE = future_brief.expiry_date if future_brief.expiry_date else "2026-03-28"
        
        # 打印合约信息
        print(f"✅ 获取合约信息成功")
        print(f"   合约代码：{future_brief.symbol}")
        print(f"   交易所：{future_brief.exchange}")
        print(f"   合约乘数：{FUTURE_MULTIPLIER} 盎司/手")
        print(f"   最小变动价位：{FUTURE_TICK_SIZE} USD")
        print(f"   到期日：{FUTURE_EXPIRY_DATE}")
        
        return True
    except Exception as e:
        # 降级处理：使用默认参数
        print(f"⚠️ 获取概要信息失败：{e}")
        print(f"📌 降级使用默认参数：乘数=1000，最小变动价位=0.01，到期日=2026-03-28")
        #global FUTURE_MULTIPLIER
        #FUTURE_MULTIPLIER = 1000
        return True

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


def get_kline_data(symbol, period, count=100, start_time=None, end_time=None):
    """Fetch K-line data (candles) and normalize to a pandas.DataFrame.

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
    period_map = {
        "1min": BarPeriod.ONE_MINUTE,
        "5min": BarPeriod.FIVE_MINUTES,
        "1h": BarPeriod.ONE_HOUR,
        "1d": BarPeriod.DAY
    }
    if period not in period_map:
        print(f"❌ 不支持的周期：{period}")
        return pd.DataFrame()
    
    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=4) if period == "5min" else end_time - timedelta(hours=1)
        # keep a lightweight backward-compatible print while adding structured logs
        print(symbol)
        logger.debug("get_kline_data request: symbol=%s period=%s count=%s start_time=%s end_time=%s", symbol, period, count, start_time, end_time)
        # Accept symbol as string or list-like, and use it when calling the API
        if isinstance(symbol, str):
            symbol1 = [symbol]
        elif isinstance(symbol, (list, tuple, pd.Series)):
            symbol1 = list(symbol)
        else:
            symbol1 = [symbol]

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
        use_paging = (start_ms is not None or end_ms is not None or count > 1000) and len(symbol1) == 1 and hasattr(quote_client, 'get_future_bars_by_page')

        if use_paging:
            # fetch pages until done or we've collected `count` rows
            all_pages = []
            next_token = None
            fetched = 0
            while True:
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

                # otherwise keep looping; pass page token when possible — best-effort
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
            else:
                klines = pd.DataFrame()
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

        # Normalize returned klines: can be a pandas.DataFrame or an iterable of bar objects
        if isinstance(klines, pd.DataFrame):
            df = klines.copy()
            if 'time' not in df.columns:
                print(f"❌ 返回的K数据缺少'time'列，实际列：{df.columns.tolist()}")
                return pd.DataFrame()
            if not all(col in df.columns for col in required_cols):
                print(f"❌ K数据列缺失，必要列：{required_cols}，实际列：{df.columns.tolist()}")
                return pd.DataFrame()
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
                                        logger.warning("Parsed times appeared to be around 1970 using unit=%s; switched to unit=%s", unit, alt)
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
                print(f"❌ 时间解析失败：{e}")
                return pd.DataFrame()
        else:
            # iterable of bar-like objects (with attributes .time, .open, etc.)
            # Ensure we can measure length; if not, convert to list
            try:
                klines_len = len(klines)
            except TypeError:
                klines = list(klines)
                klines_len = len(klines)

            # print bars for debugging (now that klines is sized or converted to list)
            for bar in klines:
                print(bar)

            if (hasattr(klines, 'empty') and getattr(klines, 'empty')) or klines_len < MIN_KLINES:
                print(f"❌ K数据不足（仅获取{klines_len}条）")
                return pd.DataFrame()
            else:
                print("k数据获取\r\n")

            df = pd.DataFrame([{
                'time': getattr(bar, 'time', None),
                'open': getattr(bar, 'open', None),
                'high': getattr(bar, 'high', None),
                'low': getattr(bar, 'low', None),
                'close': getattr(bar, 'close', None),
                'volume': getattr(bar, 'volume', None)
            } for bar in klines])

            if df.empty or len(df) < MIN_KLINES:
                print(f"❌ K数据不足（仅获取{len(df)}条）")
                return pd.DataFrame()

            if not all(col in df.columns for col in required_cols):
                print(f"❌ K数据列缺失，必要列：{required_cols}，实际列：{df.columns.tolist()}")
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
                                        logger.warning("Parsed times appeared to be around 1970 using unit=%s; switched to unit=%s", unit, alt)
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
                print(f"❌ 时间解析失败：{e}")
                return pd.DataFrame()

        df.set_index('time', inplace=True)
        # sort and keep the most recent `count` rows
        df.sort_index(inplace=True)
        if len(df) > count:
            df = df.tail(count)

        print(df)
        logger.info("get_kline_data returning %s rows for %s", len(df), symbol)
        return df

    except Exception as e:
        logger.exception("get_kline_data failed")
        print(f"❌ 获取K失败11：{str(e)}")
        return pd.DataFrame()

def calculate_indicators(df_1m, df_5m):
    """计算技术指标（逻辑不变）"""
    global atr_5m, last_boll_width, is_boll_divergence
    indicators = {}
    
    if not df_5m.empty and len(df_5m) >= GRID_BOLL_PERIOD:
        # Work on a copy to avoid mutating caller's DataFrame
        df5 = df_5m.copy()
        ma = talib.MA(df5['close'], timeperiod=GRID_BOLL_PERIOD)
        df5['boll_mid'] = ma.values if hasattr(ma, 'values') else ma

        upper, mid, lower = talib.BBANDS(
            df5['close'],
            timeperiod=GRID_BOLL_PERIOD,
            nbdevup=GRID_BOLL_STD,
            nbdevdn=GRID_BOLL_STD,
            matype=0
        )
        df5['boll_upper'] = upper.values if hasattr(upper, 'values') else upper
        df5['boll_lower'] = lower.values if hasattr(lower, 'values') else lower
        
        atrv = talib.ATR(
            df5['high'],
            df5['low'],
            df5['close'],
            timeperiod=GRID_ATR_PERIOD
        )
        df5['atr'] = atrv.values if hasattr(atrv, 'values') else atrv
        
        rsiv = talib.RSI(df5['close'], timeperiod=GRID_RSI_PERIOD_5M)
        df5['rsi'] = rsiv.values if hasattr(rsiv, 'values') else rsiv

        # Helper: last valid (non-NaN) value or default
        def _last_valid(series, default=None):
            s = series.dropna()
            return s.iloc[-1] if len(s) > 0 else default

        # Update atr_5m only if the latest (last-row) ATR value is valid (non-NaN).
        # This preserves previous atr_5m when the newest bar has missing ATR.
        try:
            atr_latest = df5['atr'].iloc[-1]
        except Exception:
            atr_latest = None
        if pd.notna(atr_latest):
            atr_5m = atr_latest
        else:
            # keep previous atr_5m unchanged
            pass

        # Compute current Boll width using last valid upper/lower
        last_upper = _last_valid(df5['boll_upper'], None)
        last_lower = _last_valid(df5['boll_lower'], None)
        if last_upper is not None and last_lower is not None:
            current_boll_width = last_upper - last_lower
        else:
            current_boll_width = last_boll_width

        if last_boll_width > 0 and current_boll_width is not None:
            width_increase = (current_boll_width - last_boll_width) / last_boll_width
            # atr increase computed from last two non-NaN ATR values
            atr_nonnull = df5['atr'].dropna()
            if len(atr_nonnull) >= 1:
                atr_current = atr_nonnull.iloc[-1]
                atr_prev = atr_nonnull.iloc[-2] if len(atr_nonnull) >= 2 else atr_current
                atr_increase = (atr_current - atr_prev) / atr_prev if atr_prev > 0 else 0
            else:
                atr_current = atr_5m
                atr_prev = atr_5m
                atr_increase = 0

            is_boll_divergence = (width_increase >= BOLL_DIVERGENCE_THRESHOLD) and (atr_increase >= ATR_AMPLIFICATION_THRESHOLD)

        last_boll_width = current_boll_width

        # Use last valid (non-NaN) values for indicators to avoid NaN propagation
        indicators['5m'] = {
            'boll_mid': _last_valid(df5['boll_mid'], None),
            'boll_upper': _last_valid(df5['boll_upper'], None),
            'boll_lower': _last_valid(df5['boll_lower'], None),
            'rsi': _last_valid(df5['rsi'], None),
            'atr': atr_5m
        }
    
    if not df_1m.empty and len(df_1m) >= GRID_RSI_PERIOD_1M:
        df_1m['rsi'] = talib.RSI(df_1m['close'], timeperiod=GRID_RSI_PERIOD_1M)
        indicators['1m'] = {
            'rsi': df_1m['rsi'].iloc[-1],
            'close': df_1m['close'].iloc[-1],
            'volume': df_1m['volume'].iloc[-1]
        }
    
    return indicators

def judge_market_trend(indicators):
    """判断行情类型（逻辑不变）"""
    if '5m' not in indicators or '1m' not in indicators:
        return 'unknown'
    
    boll_mid = indicators['5m']['boll_mid']
    boll_upper = indicators['5m']['boll_upper']
    boll_lower = indicators['5m']['boll_lower']
    rsi_5m = indicators['5m']['rsi']
    price_current = indicators['1m']['close']
    
    if is_boll_divergence:
        return 'boll_divergence_up' if price_current > boll_mid else 'boll_divergence_down'
    if price_current > boll_upper and rsi_5m > 70:
        return 'bull_trend'
    elif price_current < boll_lower and rsi_5m < 30:
        return 'bear_trend'
    elif boll_mid < price_current < boll_upper and 50 < rsi_5m < 70:
        return 'osc_bull'
    elif boll_lower < price_current < boll_mid and 30 < rsi_5m < 50:
        return 'osc_bear'
    else:
        return 'osc_normal'

def adjust_grid_interval(trend, indicators):
    """动态调整网格区间（逻辑不变）"""
    global grid_upper, grid_lower
    boll_mid = indicators['5m']['boll_mid']
    boll_upper = indicators['5m']['boll_upper']
    boll_lower = indicators['5m']['boll_lower']
    atr = indicators['5m']['atr']
    
    if trend == 'boll_divergence_up':
        grid_lower = boll_mid + 0.3 * atr
        grid_upper = boll_upper - 0.5 * atr
    elif trend == 'boll_divergence_down':
        grid_lower = boll_lower + 0.5 * atr
        grid_upper = boll_mid - 0.3 * atr
    elif trend == 'bull_trend':
        grid_lower = boll_mid
        grid_upper = boll_upper
    elif trend == 'bear_trend':
        grid_lower = boll_lower
        grid_upper = boll_mid
    elif trend == 'osc_bull':
        grid_lower = boll_mid - 0.2 * atr
        grid_upper = boll_upper - 0.2 * atr
    elif trend == 'osc_bear':
        grid_lower = boll_lower + 0.2 * atr
        grid_upper = boll_mid + 0.2 * atr
    else:
        grid_lower = boll_lower
        grid_upper = boll_upper
    
    # Try to include current price (from 1m indicators) in the status printout if available
    price_current = None
    try:
        price_current = indicators.get('1m', {}).get('close')
    except Exception:
        price_current = None

    if price_current is None or (isinstance(price_current, float) and np.isnan(price_current)):
        price_str = 'N/A'
    else:
        price_str = f"{price_current:.2f}"

    print(f"📌 行情类型：{trend} | 网格区间：{grid_lower:.2f} - {grid_upper:.2f} | ATR：{atr:.2f} | 当前价：{price_str}")

def check_risk_control(price, side):
    """风控检查（适配动态乘数）"""
    global daily_loss, current_position, today
    
    # 每日重置亏损
    if datetime.now().date() != today:
        daily_loss = 0
        today = datetime.now().date()
    
    # 1. 仓位上限检查
    if side == 'BUY' and current_position >= GRID_MAX_POSITION:
        print(f"⚠️ 仓位已达上限（{GRID_MAX_POSITION}手），禁止加仓")
        return False
    
    # 2. 日亏损上限检查
    if daily_loss >= DAILY_LOSS_LIMIT:
        print(f"⚠️ 当日亏损达上限（{DAILY_LOSS_LIMIT}美元），禁止开仓")
        return False
    
    # 3. 单笔亏损检查（使用动态乘数）
    stop_loss_price = price - STOP_LOSS_MULTIPLIER * atr_5m
    single_loss = (price - stop_loss_price) * FUTURE_MULTIPLIER
    if single_loss > SINGLE_TRADE_LOSS:
        print(f"⚠️ 单笔LOSS超限（{single_loss:.2f}＞{SINGLE_TRADE_LOSS}），禁止开仓")
        return False
    
    return True

def place_tiger_order(side, quantity, price, stop_loss_price=None):
    """Place a futures order through `trade_client` (per API docs).

    This implementation tries to build a proper `contract` and `order` using
    tigeropen helper functions (constructed at runtime to keep import-time
    safety for tests). If the SDK helpers are unavailable or fail, it falls
    back to a simple namespace object and — in sandbox — simulates success.

    Production orders are refused unless ALLOW_REAL_TRADING=1 is set.
    """
    global current_position, daily_loss

    # Resolve account id
    account_id = getattr(client_config, 'account', None) or getattr(client_config, 'account_id', None)

    # refuse live orders unless explicitly allowed
    if RUN_ENV == 'production' and os.getenv('ALLOW_REAL_TRADING', '0') != '1':
        msg = "⚠️ 实盘下单受限：环境为 production，未启用 ALLOW_REAL_TRADING=1，跳过下单"
        logger.warning(msg)
        print(msg)
        return False

    order_obj = None
    try:
        # Try to construct a proper Contract + Order via tigeropen helpers
        try:
            from tigeropen.common.util.contract_utils import future_contract
            from tigeropen.common.util.order_utils import limit_order, limit_order_with_legs, order_leg

            # Best-effort contract construction: prefer compact symbol like 'SIL2603'
            try:
                contract_symbol = _to_api_identifier(FUTURE_SYMBOL)
                contract = future_contract(symbol=contract_symbol, currency=FUTURE_CURRENCY)
            except Exception:
                # Fallback to base symbol
                base = FUTURE_SYMBOL.split('.')[0]
                contract = future_contract(symbol=base, currency=FUTURE_CURRENCY)

            # Build a limit order (LMT)
            order_obj = limit_order(account=account_id, contract=contract, action=side, limit_price=price, quantity=quantity)

            # If caller provided a stop loss, attach it as an order leg when helpers allow it
            if stop_loss_price:
                try:
                    stop_leg = order_leg('LOSS', float(stop_loss_price), time_in_force='DAY', outside_rth=False)
                    order_obj = limit_order_with_legs(account_id, contract, side, quantity, limit_price=price, order_legs=[stop_leg])
                except Exception:
                    # If adding legs fails, try to set aux_price/stop_price on the order
                    try:
                        setattr(order_obj, 'aux_price', float(stop_loss_price))
                    except Exception:
                        pass
        except Exception:
            # If tigeropen helpers are unavailable, fall back to a lightweight order-like object
            from types import SimpleNamespace
            order_obj = SimpleNamespace()
            order_obj.account = account_id
            order_obj.action = side
            order_obj.quantity = quantity
            order_obj.limit_price = price
            if stop_loss_price:
                order_obj.aux_price = stop_loss_price

        # Ensure trade_client exists and attempt to place the order
        if trade_client is None:
            raise RuntimeError('trade_client not configured')

        try:
            # The SDK may return an id or populate order_obj.id; capture both
            returned = trade_client.place_order(order_obj)
            order_id = getattr(order_obj, 'id', None) or returned or getattr(returned, 'id', None) or getattr(returned, 'order_id', None)

            env_tip = "[模拟单]" if RUN_ENV == 'sandbox' else "[实盘单]"
            msg = f"✅ {env_tip} 下单成功 | {side} {quantity}手 | 价格：{price:.2f} | 订单ID：{order_id if order_id else '未知'}"
            logger.info(msg)
            print(msg)

            # Update simple in-memory state consistent with previous behavior
            if side == 'BUY':
                current_position += quantity
            else:
                current_position -= quantity
                if stop_loss_price:
                    daily_loss += (price - stop_loss_price) * FUTURE_MULTIPLIER * quantity

            return True

        except Exception as e:
            # On failure, simulate success when in sandbox (keeps previous behaviour)
            print(f"⚠️ 下单调用失败：{e} — 将在 sandbox 中模拟成功响应（如适用）")
            if RUN_ENV == 'sandbox':
                env_tip = "[模拟单]"
                msg = f"✅ {env_tip} 下单成功（模拟） | {side} {quantity}手 | 价格：{price:.2f} | 订单ID：SIMULATED"
                logger.info(msg)
                print(msg)
                if side == 'BUY':
                    current_position += quantity
                else:
                    current_position -= quantity
                    if stop_loss_price:
                        daily_loss += (price - stop_loss_price) * FUTURE_MULTIPLIER * quantity
                return True
            else:
                print(f"❌ 下单异常：{e}")
                return False

    except Exception as e:
        print(f"❌ 下单异常：{str(e)}")
        logger.exception('place_tiger_order failure')
        return False

def grid_trading_strategy():
    """核心网格策略逻辑（逻辑不变）"""
    df_1m = get_kline_data([FUTURE_SYMBOL], '1min', count=30)
    df_5m = get_kline_data([FUTURE_SYMBOL], '5min', count=50)
    if df_1m.empty or df_5m.empty:
        print("⚠️ 数据不足，跳过本STEP 22")
        return
    
    indicators = calculate_indicators(df_1m, df_5m)
    if not indicators or '5m' not in indicators or '1m' not in indicators:
        print("⚠️ 指标计算失败，跳过本次循环33")
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
            stop_loss_price = price_current - STOP_LOSS_MULTIPLIER * atr
            place_tiger_order('BUY', 1, price_current, stop_loss_price)
    
    rsi_high_map = {
        'boll_divergence_up': 80,
        'osc_bull': 75,
        'bull_trend': 70,
        'osc_normal': 70
    }
    rsi_high = rsi_high_map.get(trend, 70)
    if price_current >= grid_upper and rsi_1m >= rsi_high and current_position > 0:
        place_tiger_order('SELL', 1, price_current)
    
    if current_position > 0:
        stop_loss_price = price_current - STOP_LOSS_MULTIPLIER * atr
        if price_current <= stop_loss_price:
            env_tip = "[模拟止损]" if RUN_ENV == 'sandbox' else "[实盘止损]"
            print(f"⚠️ {env_tip} 触发止损，平仓{current_position}手")
            place_tiger_order('SELL', current_position, price_current)


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
    global current_position

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
        if check_risk_control(price_current, 'BUY'):
            stop_loss_price = price_current - STOP_LOSS_MULTIPLIER * (atr if atr else 0)
            print(f"🔧 boll1m_grid_strategy ({trend}): 发现回调+反弹，准备买入 at {price_current:.2f} (boll_lower={boll_lower:.2f})")
            place_tiger_order('BUY', 1, price_current, stop_loss_price)
        else:
            print("🔧 boll1m_grid_strategy: 风控阻止买入")
    else:
        print(f"🔧 boll1m_grid_strategy ({trend}): 未满足回调确认或未检测到dip（dip_detected={dip_detected}, last={last}, prev={prev}）")

    # Sell at mid band when holding (unchanged)
    if current_position > 0 and price_current is not None and boll_mid is not None and price_current >= boll_mid:
        print(f"🔧 boll1m_grid_strategy: 触发卖出 at {price_current:.2f} (boll_mid={boll_mid:.2f})")
        place_tiger_order('SELL', 1, price_current)


# ====================== 主程序 ======================
if __name__ == "__main__":
    # 1. 验证API连接
    if not verify_api_connection():
        exit(1)
    
    # 2. 启动网格策略
    try:
        print("🚀 启动网格处理）...")
        while True:
            #grid_trading_strategy()
            boll1m_grid_strategy()
            time.sleep(20)  # 
    except KeyboardInterrupt:
        print("🛑 用户终止程序")
    except Exception as e:
        print(f"❌ 程序异常：{e}")
    finally:
        # 平仓所有持仓（可选，实盘谨慎操作）
        '''
        if current_position > 0:
            print(f"⚠️ 程序退出，平仓{current_position}手持仓")
            latest_price = get_kline_data(FUTURE_SYMBOL, '1min', count=1)['close'].iloc[-1]
            place_tiger_order('SELL', current_position, latest_price)
        '''    
        print("✅ 程序结束")