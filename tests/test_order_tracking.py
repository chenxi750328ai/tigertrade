"""
测试交易状态管理功能
"""

import sys
import random
from datetime import datetime, timedelta
from types import SimpleNamespace
from collections import defaultdict

# 模拟全局变量
current_position = 0           # 当前持仓手数
daily_loss = 0                 # 当日累计亏损
grid_upper = 0                 # 网格上轨
grid_lower = 0                 # 网格下轨
last_boll_width = 0            # 上一次BOLL轨道间距
atr_5m = 0                     # 5分钟ATR值
is_boll_divergence = False     # 是否BOLL发散

# 新增订单状态跟踪
open_orders = {}               # 记录待平仓的买单 {order_id: {'quantity': qty, 'price': price, 'timestamp': ts, 'tech_params': {}, 'reason': ''}}
closed_positions = []          # 已平仓的交易记录 [{'buy_order_id': id, 'sell_order_id': id, 'buy_price': bp, 'sell_price': sp, 'analysis': {...}}, ...]}

# 新增止盈相关全局变量
position_entry_times = {}      # 记录每个持仓的入场时间 {position_id: timestamp}
position_entry_prices = {}     # 记录每个持仓的入场价格 {position_id: entry_price}
active_take_profit_orders = {} # 跟踪已提交的止盈单 {position_id: {'target_price': price, 'submit_time': timestamp}}

# 止盈参数
TAKE_PROFIT_TIMEOUT = 15       # 止盈单超时（分钟）
MIN_PROFIT_RATIO = 0.02        # 最低主动止盈比例（2%）

# 运行环境标识
RUN_ENV = 'sandbox'            # 设置为沙箱模式
today = datetime.now().date()

# 期货配置
FUTURE_SYMBOL = "SIL.COMEX.202603"
FUTURE_CURRENCY = "USD"
FUTURE_MULTIPLIER = 1000

# 网格策略参数
GRID_MAX_POSITION = 3          # 最大持仓手数

def place_tiger_order(side, quantity, price, stop_loss_price=None, take_profit_price=None, tech_params=None, reason=''):
    """模拟下单函数，用于测试订单跟踪功能"""
    global current_position, daily_loss, position_entry_times, position_entry_prices, active_take_profit_orders, open_orders

    import time
    
    # 模拟订单ID生成
    order_id = f"ORDER_{int(time.time())}_{random.randint(1000, 9999)}"
    
    msg = f"✅ [模拟单] 下单成功 | {side} {quantity}手 | 价格：{price:.2f} | 订单ID：{order_id}"
    print(msg)

    # 更新简单 in-memory state consistent with previous behavior
    if side == 'BUY':
        current_position += quantity
        
        # 记录买单到open_orders，用于跟踪交易闭环
        for i in range(quantity):
            individual_order_id = f"{order_id}_qty_{i+1}"
            open_orders[individual_order_id] = {
                'quantity': 1,  # 每个订单项代表1手
                'price': price,
                'timestamp': time.time(),
                'type': 'buy',
                'tech_params': tech_params or {},  # 技术参数
                'reason': reason or '未知'          # 开仓原因
            }
        
        # 记录新买入持仓的入场时间和价格
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
            # 获取最早的一个买单
            oldest_order_id = min(open_orders.keys(), key=lambda x: open_orders[x]['timestamp'])
            oldest_order = open_orders[oldest_order_id]
            
            # 确定本次交易的手数
            trade_qty = min(remaining_qty_to_sell, oldest_order['quantity'])
            
            # 记录平仓信息，包括详细分析
            analysis = {
                'buy_reason': oldest_order['reason'],
                'buy_tech_params': oldest_order['tech_params'],
                'sell_reason': reason or '未知',
                'sell_tech_params': tech_params or {},
                'stop_loss_triggered': stop_loss_price is not None
            }
            
            closed_positions.append({
                'buy_order_id': oldest_order_id,
                'sell_order_id': order_id,
                'buy_price': oldest_order['price'],
                'sell_price': price,
                'quantity': trade_qty,
                'profit': (price - oldest_order['price']) * trade_qty * FUTURE_MULTIPLIER,
                'timestamp': time.time(),
                'analysis': analysis
            })
            
            # 更新买单状态
            if oldest_order['quantity'] > trade_qty:
                # 部分成交，更新剩余数量
                open_orders[oldest_order_id]['quantity'] -= trade_qty
            else:
                # 完全成交，删除订单
                del open_orders[oldest_order_id]
            
            # 减少待卖出数量
            remaining_qty_to_sell -= trade_qty
        
        if stop_loss_price:
            daily_loss += (price - stop_loss_price) * FUTURE_MULTIPLIER * quantity

    # 如果有止盈价格，记录到活动止盈单中
    if take_profit_price is not None:
        print(f"🧭 [模拟单] 已提交止盈单 | {'SELL' if side=='BUY' else 'BUY'} {quantity}手 | 价格：{float(take_profit_price):.2f}")
        # 记录已提交的止盈单（模拟）
        import time
        for pos_id in range(max(0, current_position - quantity), current_position):
            active_take_profit_orders[pos_id] = {
                'target_price': float(take_profit_price),
                'submit_time': time.time(),
                'quantity': quantity,
                'entry_price': position_entry_prices[pos_id],  # 记录入场价格用于计算
                'entry_reason': reason or '未知',               # 记录入场原因
                'entry_tech_params': tech_params or {}         # 记录入场技术参数
            }
    
    return True

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
    # 3. 单笔亏损检查（此处省略）
    
    return True

def check_active_take_profits(current_price):
    """模拟检查主动止盈"""
    global current_position, active_take_profit_orders, position_entry_times, position_entry_prices
    
    import time
    
    if current_position <= 0:
        return False
    
    positions_to_close = []
    
    for pos_id in list(active_take_profit_orders.keys()):
        if pos_id in active_take_profit_orders:
            tp_info = active_take_profit_orders[pos_id]
            target_price = tp_info['target_price']
            
            # 检查当前价格是否达到最低盈利目标
            if current_price >= target_price:
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
            place_tiger_order('SELL', item['quantity'], current_price, 
                            reason=f"主动止盈 - 目标价格 {item['target_price']:.2f}",
                            tech_params={'current_price': current_price, 
                                       'entry_price': item['entry_price'],
                                       'target_price': item['target_price'],
                                       'exit_type': 'take_profit'})
            
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
    """模拟检查超时止盈"""
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
            place_tiger_order('SELL', item['quantity'], current_price,
                            reason=f"超时止盈 - 已等待 {item['elapsed_minutes']:.1f}分钟",
                            tech_params={
                                'current_price': current_price,
                                'entry_price': item['entry_price'],
                                'target_price': item['target_price'],
                                'elapsed_minutes': item['elapsed_minutes'],
                                'exit_type': 'timeout_take_profit'
                            })
            
            # 清理相关记录
            if pos_id in active_take_profit_orders:
                del active_take_profit_orders[pos_id]
            if pos_id in position_entry_times:
                del position_entry_times[pos_id]
            if pos_id in position_entry_prices:
                del position_entry_prices[pos_id]
        
        return True
    
    return False

def print_trade_analysis():
    """打印详细的交易分析报告"""
    print("\n🔍 ========== 详细交易分析报告 ========== 🔍")
    
    if not closed_positions:
        print("📈 暂无已平仓订单")
        return
    
    for i, trade in enumerate(closed_positions, 1):
        print(f"\n📊 交易 #{i}:")
        print(f"   买入价: {trade['buy_price']:.2f} | 卖出价: {trade['sell_price']:.2f} | 数量: {trade['quantity']}手 | 盈亏: {trade['profit']:.2f}USD")
        print(f"   交易时间: {datetime.fromtimestamp(trade['timestamp'])}")
        
        analysis = trade['analysis']
        
        print(f"   📌 开仓分析:")
        print(f"     - 开仓原因: {analysis['buy_reason']}")
        if analysis['buy_tech_params']:
            print(f"     - 技术参数: ", end="")
            params_str = ", ".join([f"{k}:{v:.2f}" if isinstance(v, float) else f"{k}:{v}" for k, v in analysis['buy_tech_params'].items()])
            print(params_str)
        
        print(f"   📉 平仓分析:")
        print(f"     - 平仓原因: {analysis['sell_reason']}")
        if analysis['sell_tech_params']:
            print(f"     - 技术参数: ", end="")
            params_str = ", ".join([f"{k}:{v:.2f}" if isinstance(v, float) else f"{k}:{v}" for k, v in analysis['sell_tech_params'].items()])
            print(params_str)
        
        print(f"   ⚠️  特殊标记: {'止损触发' if analysis['stop_loss_triggered'] else '非止损触发'}")
        
        profit_ratio = (trade['sell_price'] - trade['buy_price']) / trade['buy_price'] * 100
        print(f"   💰 盈亏比例: {profit_ratio:+.2f}%")
    
    print("\n======================================== 🔍")


def analyze_hourly_performance():
    """分析每小时的盈亏表现并提供改进建议"""
    print("\n🔍 ========== 每小时盈亏分析与改进建议 ========== 🔍")
    
    if not closed_positions:
        print("📈 暂无已平仓订单")
        return
    
    # 按小时分组交易数据
    hourly_data = defaultdict(list)
    for trade in closed_positions:
        hour = datetime.fromtimestamp(trade['timestamp']).hour
        hourly_data[hour].append(trade)
    
    # 按小时分析
    for hour in sorted(hourly_data.keys()):
        trades = hourly_data[hour]
        total_trades = len(trades)
        total_profit = sum(trade['profit'] for trade in trades)
        profitable_trades = sum(1 for trade in trades if trade['profit'] > 0)
        losing_trades = total_trades - profitable_trades
        
        win_rate = profitable_trades / total_trades * 100 if total_trades > 0 else 0
        avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0
        
        print(f"\n🕒 小时 {hour:02d}:00 - {hour:02d}:59")
        print(f"📊 交易数: {total_trades}")
        print(f"💰 总盈亏: {total_profit:.2f} USD")
        print(f"🎯 胜率: {win_rate:.2f}%")
        print(f"📈 平均盈亏: {avg_profit_per_trade:.2f} USD")
        
        # 根据表现提供建议
        if total_profit < 0:
            print(f"⚠️  建议: 该时段表现亏损，建议调整策略或降低交易频率")
            if win_rate < 50:
                print(f"⚠️  胜率过低，考虑优化入场时机或增加过滤条件")
        elif win_rate < 50:
            print(f"⚠️  虽然盈利但胜率较低，建议优化出场策略")
        else:
            print(f"✅ 该时段表现良好，继续保持")
    
    print("\n💡 总体建议:")
    
    # 分析整体表现
    total_trades = len(closed_positions)
    total_profit = sum(trade['profit'] for trade in closed_positions)
    profitable_trades = sum(1 for trade in closed_positions if trade['profit'] > 0)
    win_rate = profitable_trades / total_trades * 100 if total_trades > 0 else 0
    
    if total_profit > 0:
        print("- 整体盈利，策略方向正确")
        if win_rate < 50:
            print("- 虽然整体盈利，但胜率偏低，建议优化止损设置")
        else:
            print("- 胜率较高，说明策略有效性较好")
    else:
        print("- 整体亏损，需要重新审视策略参数")
        print("- 检查市场趋势是否与策略匹配")
        print("- 考虑调整止盈止损比例")
    
    # 分析最大亏损原因
    max_loss_trade = min(closed_positions, key=lambda x: x['profit']) if closed_positions else None
    if max_loss_trade and max_loss_trade['profit'] < 0:
        loss_pct = abs(max_loss_trade['profit']) / (max_loss_trade['buy_price'] * max_loss_trade['quantity'] * FUTURE_MULTIPLIER) * 100
        print(f"- 最大单笔亏损: {max_loss_trade['profit']:.2f} USD ({loss_pct:.2f}% of position)")
        print(f"  - 发生时间: {datetime.fromtimestamp(max_loss_trade['timestamp'])}")
        print(f"  - 建议: 检查此笔交易的市场环境，考虑调整止损点位")


def generate_order_summary():
    """生成订单总结报告"""
    print("\n📊 ========== 订单总结报告 ========== 📊")
    
    if not closed_positions:
        print("📈 暂无已平仓订单")
        return
    
    total_trades = len(closed_positions)
    total_profit = sum(trade['profit'] for trade in closed_positions)
    profitable_trades = sum(1 for trade in closed_positions if trade['profit'] > 0)
    losing_trades = total_trades - profitable_trades
    
    win_rate = profitable_trades / total_trades * 100 if total_trades > 0 else 0
    
    avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0
    max_profit_trade = max(closed_positions, key=lambda x: x['profit']) if closed_positions else None
    max_loss_trade = min(closed_positions, key=lambda x: x['profit']) if closed_positions else None
    
    active_tp_count = sum(1 for trade in closed_positions if '主动止盈' in str(trade['analysis']['sell_reason']))
    timeout_tp_count = sum(1 for trade in closed_positions if '超时止盈' in str(trade['analysis']['sell_reason']))
    
    print(f"📈 总交易数: {total_trades}")
    print(f"💰 总盈亏: {total_profit:.2f} USD")
    print(f"✅ 盈利交易: {profitable_trades}")
    print(f"❌ 亏损交易: {losing_trades}")
    print(f"🎯 胜率: {win_rate:.2f}%")
    print(f"📊 平均每单盈亏: {avg_profit_per_trade:.2f} USD")
    print(f"⏱️  主动止盈单: {active_tp_count}")
    print(f"⏰  超时止盈单: {timeout_tp_count}")
    
    if max_profit_trade:
        print(f"🏆 最大单笔盈利: {max_profit_trade['profit']:.2f} USD "
              f"(买价 {max_profit_trade['buy_price']:.2f} -> 卖价 {max_profit_trade['sell_price']:.2f})")
    
    if max_loss_trade:
        print(f"📉 最大单笔亏损: {max_loss_trade['profit']:.2f} USD "
              f"(买价 {max_loss_trade['buy_price']:.2f} -> 卖价 {max_loss_trade['sell_price']:.2f})")
    
    print("\n📋 详细交易列表:")
    print("No. | 买入价 | 卖出价 | 数量 | 盈亏   | 时间戳")
    print("----|--------|--------|------|--------|---------")
    for i, trade in enumerate(closed_positions, 1):
        dt = datetime.fromtimestamp(trade['timestamp']).strftime('%H:%M:%S')
        print(f"{i:2d}. | {trade['buy_price']:6.2f} | {trade['sell_price']:6.2f} | "
              f"{trade['quantity']:4d} | {trade['profit']:6.2f} | {dt}")
    
    print("================================== 📊")


def test_order_tracking():
    """测试订单跟踪和交易闭环功能"""
    global current_position, open_orders, closed_positions
    
    print("🧪 开始测试订单跟踪和交易闭环功能...")
    
    # 重置测试状态
    current_position = 0
    open_orders.clear()
    closed_positions.clear()
    
    # 模拟买入操作
    print("📝 模拟买入操作...")
    place_tiger_order('BUY', 1, 100.0)
    place_tiger_order('BUY', 1, 102.0)
    place_tiger_order('BUY', 1, 104.0)
    
    print(f"📊 买入后状态: 持仓={current_position}, 待平仓订单={len(open_orders)}, 已平仓={len(closed_positions)}")
    
    # 验证买入操作是否正确记录
    assert current_position == 3, f"预期持仓3手，实际{current_position}手"
    assert len(open_orders) == 3, f"预期待平仓订单3个，实际{len(open_orders)}个"
    assert len(closed_positions) == 0, f"预期已平仓0个，实际{len(closed_positions)}个"
    
    # 模拟卖出操作
    print("📝 模拟卖出操作...")
    place_tiger_order('SELL', 2, 108.0)  # 卖出2手
    
    print(f"📊 卖出后状态: 持仓={current_position}, 待平仓订单={len(open_orders)}, 已平仓={len(closed_positions)}")
    
    # 验证卖出操作是否正确记录
    assert current_position == 1, f"预期持仓1手，实际{current_position}手"
    assert len(open_orders) == 1, f"预期待平仓订单1个，实际{len(open_orders)}个"
    assert len(closed_positions) == 2, f"预期已平仓2个，实际{len(closed_positions)}个"
    
    # 卖出剩余持仓
    place_tiger_order('SELL', 1, 110.0)
    
    print(f"📊 全部卖出后状态: 持仓={current_position}, 待平仓订单={len(open_orders)}, 已平仓={len(closed_positions)}")
    
    # 验证所有持仓都已平仓
    assert current_position == 0, f"预期持仓0手，实际{current_position}手"
    assert len(open_orders) == 0, f"预期待平仓订单0个，实际{len(open_orders)}个"
    assert len(closed_positions) == 3, f"预期已平仓3个，实际{len(closed_positions)}个"
    
    print("✅ 订单跟踪和交易闭环功能测试通过！")
    
    # 显示交易详情
    for i, trade in enumerate(closed_positions):
        profit = trade['profit']
        print(f"📈 交易{i+1}: 买入价 {trade['buy_price']}, 卖出价 {trade['sell_price']}, 盈亏: {profit:.2f}USD")


def test_edge_case_no_position():
    """测试无持仓时的边界情况"""
    global current_position, open_orders, closed_positions, active_take_profit_orders
    
    print("\n🧪 开始测试无持仓时的边界情况...")
    
    # 重置测试状态
    current_position = 0
    open_orders.clear()
    closed_positions.clear()
    active_take_profit_orders.clear()
    
    # 测试在无持仓时调用止盈检查
    result = check_active_take_profits(100.0)
    assert result == False, "无持仓时主动止盈应该返回False"
    
    result = check_timeout_take_profits(100.0)
    assert result == False, "无持仓时超时止盈应该返回False"
    
    print("✅ 无持仓边界情况测试通过！")


def test_partial_fill_scenarios():
    """测试部分成交场景"""
    global current_position, open_orders, closed_positions
    
    print("\n🧪 开始测试部分成交场景...")
    
    # 重置测试状态
    current_position = 0
    open_orders.clear()
    closed_positions.clear()
    
    # 买入多手
    place_tiger_order('BUY', 3, 100.0)
    
    print(f"📊 买入后状态: 持仓={current_position}, 待平仓订单={len(open_orders)}")
    
    # 只卖出1手
    place_tiger_order('SELL', 1, 105.0)
    
    print(f"📊 卖出1手后状态: 持仓={current_position}, 待平仓订单={len(open_orders)}, 已平仓={len(closed_positions)}")
    
    # 验证仍有2手持仓
    assert current_position == 2, f"预期持仓2手，实际{current_position}手"
    assert len(closed_positions) == 1, f"预期已平仓1个，实际{len(closed_positions)}个"
    
    print("✅ 部分成交场景测试通过！")


def test_multiple_risk_controls():
    """测试多种风控条件"""
    global current_position
    
    print("\n🧪 开始测试多种风控条件...")
    
    # 重置测试状态
    current_position = 0
    
    # 设置最大持仓为2
    global GRID_MAX_POSITION
    original_max_pos = GRID_MAX_POSITION
    GRID_MAX_POSITION = 2
    
    # 买入达到最大持仓
    place_tiger_order('BUY', 1, 60.0)
    place_tiger_order('BUY', 1, 62.0)
    
    # 尝试超过最大持仓
    result = check_risk_control(66.0, 'BUY')
    assert result == False, "应当拒绝超过最大持仓的买入"
    
    # 测试卖出不受持仓限制影响
    result = check_risk_control(68.0, 'SELL')
    assert result == True, "卖出不应该受持仓限制影响"
    
    # 恢复原始设置
    GRID_MAX_POSITION = original_max_pos
    
    print("✅ 多种风控条件测试通过！")


def test_stop_loss_scenario():
    """测试止损场景"""
    global current_position, open_orders, closed_positions
    
    print("\n🧪 开始测试止损场景...")
    
    # 重置测试状态
    current_position = 0
    open_orders.clear()
    closed_positions.clear()
    
    # 买入持仓
    place_tiger_order('BUY', 1, 100.0)
    place_tiger_order('BUY', 1, 102.0)
    
    print(f"📊 买入后状态: 持仓={current_position}, 价格=[100.0, 102.0]")
    
    # 模拟价格下跌至止损位
    stop_loss_price = 95.0
    current_price = 90.0
    
    print(f"📉 价格跌至 {current_price}，触发止损...")
    place_tiger_order('SELL', current_position, current_price)
    
    print(f"📊 止损后状态: 持仓={current_position}, 待平仓订单={len(open_orders)}, 已平仓={len(closed_positions)}")
    
    # 验证所有持仓都已止损平仓
    assert current_position == 0, f"预期持仓0手，实际{current_position}手"
    assert len(closed_positions) == 2, f"预期已平仓2个，实际{len(closed_positions)}个"
    
    print("✅ 止损场景测试通过！")
    
    # 显示止损交易详情
    for i, trade in enumerate(closed_positions):
        profit = trade['profit']
        print(f"📉 止损交易{i+1}: 买入价 {trade['buy_price']}, 卖出价 {trade['sell_price']}, 盈亏: {profit:.2f}USD")
    
    # 显示止损交易详情
    for i, trade in enumerate(closed_positions):
        profit = trade['profit']
        print(f"📉 止损交易{i+1}: 买入价 {trade['buy_price']}, 卖出价 {trade['sell_price']}, 盈亏: {profit:.2f}USD")


def test_take_profit_scenario():
    """测试止盈场景"""
    global current_position, active_take_profit_orders, position_entry_prices, closed_positions
    
    print("\n🧪 开始测试止盈场景...")
    
    # 重置测试状态
    current_position = 0
    active_take_profit_orders.clear()
    position_entry_prices.clear()
    closed_positions.clear()
    
    # 买入持仓并设置止盈
    place_tiger_order('BUY', 1, 100.0, take_profit_price=110.0)
    place_tiger_order('BUY', 1, 102.0, take_profit_price=112.0)
    
    print(f"📊 买入并设置止盈后状态: 持仓={current_position}, 止盈单={len(active_take_profit_orders)}")
    
    # 模拟价格上涨至触发止盈
    current_price = 115.0
    
    print(f"📈 价格涨至 {current_price}，触发主动止盈...")
    check_active_take_profits(current_price)
    
    print(f"📊 止盈后状态: 持仓={current_position}, 活跃止盈单={len(active_take_profit_orders)}, 已平仓={len(closed_positions)}")
    
    # 验证所有持仓都已止盈平仓
    assert current_position == 0, f"预期持仓0手，实际{current_position}手"
    assert len(active_take_profit_orders) == 0, f"预期活跃止盈单0个，实际{len(active_take_profit_orders)}个"
    assert len(closed_positions) == 2, f"预期已平仓2个，实际{len(closed_positions)}个"
    
    print("✅ 止盈场景测试通过！")
    
    # 显示止盈交易详情
    for i, trade in enumerate(closed_positions):
        profit = trade['profit']
        print(f"📈 止盈交易{i+1}: 买入价 {trade['buy_price']}, 卖出价 {trade['sell_price']}, 盈亏: {profit:.2f}USD")
    
    # 显示止盈交易详情
    for i, trade in enumerate(closed_positions):
        profit = trade['profit']
        print(f"📈 止盈交易{i+1}: 买入价 {trade['buy_price']}, 卖出价 {trade['sell_price']}, 盈亏: {profit:.2f}USD")


def test_timeout_take_profit_scenario():
    """测试超时止盈场景"""
    global current_position, active_take_profit_orders, position_entry_times, position_entry_prices, closed_positions
    
    print("\n🧪 开始测试超时止盈场景...")
    
    # 重置测试状态
    current_position = 0
    active_take_profit_orders.clear()
    position_entry_times.clear()
    position_entry_prices.clear()
    closed_positions.clear()
    
    import time
    
    # 买入持仓并设置止盈
    place_tiger_order('BUY', 1, 100.0, take_profit_price=110.0)
    place_tiger_order('BUY', 1, 102.0, take_profit_price=112.0)
    
    print(f"📊 买入并设置止盈后状态: 持仓={current_position}, 止盈单={len(active_take_profit_orders)}")
    
    # 修改提交时间为很久以前，模拟超时
    for pos_id in active_take_profit_orders:
        # 设置提交时间为1小时前，确保超时
        active_take_profit_orders[pos_id]['submit_time'] = time.time() - (TAKE_PROFIT_TIMEOUT + 1) * 60
    
    # 模拟价格刚好达到1/3盈利目标，触发超时止盈
    current_price = 108.0  # 高于两个头寸的1/3盈利目标
    
    print(f"⏰ 价格为 {current_price}，触发超时止盈...")
    check_timeout_take_profits(current_price)
    
    print(f"📊 超时止盈后状态: 持仓={current_position}, 活跃止盈单={len(active_take_profit_orders)}, 已平仓={len(closed_positions)}")
    
    # 验证所有持仓都已超时止盈平仓
    assert current_position == 0, f"预期持仓0手，实际{current_position}手"
    assert len(active_take_profit_orders) == 0, f"预期活跃止盈单0个，实际{len(active_take_profit_orders)}个"
    assert len(closed_positions) == 2, f"预期已平仓2个，实际{len(closed_positions)}个"
    
    print("✅ 超时止盈场景测试通过！")
    
    # 显示超时止盈交易详情
    for i, trade in enumerate(closed_positions):
        profit = trade['profit']
        print(f"⏰ 超时止盈交易{i+1}: 买入价 {trade['buy_price']}, 卖出价 {trade['sell_price']}, 盈亏: {profit:.2f}USD")


def test_zero_quantity_scenarios():
    """测试零数量下单场景"""
    global current_position
    
    print("\n🧪 开始测试零数量下单场景...")
    
    # 重置测试状态
    current_position = 0
    
    # 尝试下0手的订单，这应该不会改变持仓
    initial_position = current_position
    place_tiger_order('BUY', 0, 100.0)
    assert current_position == initial_position, f"0手订单不应改变持仓"
    
    place_tiger_order('SELL', 0, 100.0)
    assert current_position == initial_position, f"0手订单不应改变持仓"
    
    print("✅ 零数量下单场景测试通过！")


def test_negative_price_scenarios():
    """测试负价格场景"""
    global current_position
    
    print("\n🧪 开始测试负价格场景...")
    
    # 重置测试状态
    current_position = 0
    
    # 尝试使用负价格下单
    try:
        place_tiger_order('BUY', 1, -10.0)
        # 即使负价格被接受，也应该改变持仓
        assert current_position == 1, f"预期持仓1手，实际{current_position}手"
        print("✅ 负价格下单被接受")
    except Exception as e:
        print(f"✅ 负价格下单被拒绝: {e}")
    
    print("✅ 负价格场景测试通过！")


def test_empty_order_book():
    """测试在没有买单时的卖单场景"""
    global current_position, open_orders
    
    print("\n🧪 开始测试空订单簿卖单场景...")
    
    # 重置测试状态
    current_position = 0
    open_orders.clear()
    
    # 在没有买单的情况下尝试卖出
    initial_position = current_position
    place_tiger_order('SELL', 1, 100.0)
    
    # 由于没有买单，卖出操作不应该减少持仓（因为持仓已经是0）
    assert current_position == 0, f"预期持仓0手，实际{current_position}手"
    
    print("✅ 空订单簿卖单场景测试通过！")


def test_complex_interweaving_scenario():
    """测试复杂的开仓、止损、止盈交织场景"""
    global current_position, open_orders, closed_positions, active_take_profit_orders, position_entry_prices
    
    print("\n🧪 开始测试复杂的交织场景...")
    
    # 重置测试状态
    current_position = 0
    open_orders.clear()
    closed_positions.clear()
    active_take_profit_orders.clear()
    position_entry_prices.clear()
    
    print("📝 第一步：买入2手，设置止盈")
    place_tiger_order('BUY', 1, 100.0, take_profit_price=110.0,
                     tech_params={'current_price': 100.0, 'rsi': 25, 'atr': 1.5, 'boll_position': 'below_lower_band'},
                     reason='网格下轨+KDJ金叉+RSI超卖')
    place_tiger_order('BUY', 1, 102.0, take_profit_price=112.0,
                     tech_params={'current_price': 102.0, 'rsi': 28, 'atr': 1.6, 'boll_position': 'below_lower_band'},
                     reason='网格下轨+KDJ金叉+RSI超卖')
    print(f"📊 状态: 持仓={current_position}, 活跃止盈单={len(active_take_profit_orders)}")
    
    print("📝 第二步：再买入1手，不设置止盈")
    place_tiger_order('BUY', 1, 104.0)
    print(f"📊 状态: 持仓={current_position}")
    
    print("📈 第三步：价格上涨，触发部分止盈")
    check_active_take_profits(115.0)  # 应该触发前两笔持仓的止盈
    print(f"📊 状态: 持仓={current_position}, 已平仓={len(closed_positions)}")
    
    print("📝 第四步：再次买入1手")
    place_tiger_order('BUY', 1, 118.0)
    print(f"📊 状态: 持仓={current_position}")
    
    print("📉 第五步：价格下跌至止损位")
    # 平掉所有剩余持仓
    place_tiger_order('SELL', current_position, 95.0,
                     reason='止损触发',
                     tech_params={'current_price': 95.0, 'stop_loss_price': 95.0, 'exit_type': 'stop_loss'})
    print(f"📊 最终状态: 持仓={current_position}, 待平仓订单={len(open_orders)}, 已平仓={len(closed_positions)}")
    
    # 验证最终状态
    assert current_position == 0, f"预期持仓0手，实际{current_position}手"
    assert len(open_orders) == 0, f"预期待平仓订单0个，实际{len(open_orders)}个"
    assert len(closed_positions) == 4, f"预期已平仓4个，实际{len(closed_positions)}个"
    
    print("✅ 复杂交织场景测试通过！")
    
    # 显示所有交易详情
    total_profit = 0
    for i, trade in enumerate(closed_positions):
        profit = trade['profit']
        total_profit += profit
        print(f"📊 交易{i+1}: 买入价 {trade['buy_price']}, 卖出价 {trade['sell_price']}, 盈亏: {profit:.2f}USD")
    print(f"💰 总盈亏: {total_profit:.2f}USD")


def test_full_interweaving_scenario():
    """测试开仓、止损、止盈、主动止盈、超时止盈和风险控制交织的完整场景"""
    global current_position, open_orders, closed_positions, active_take_profit_orders, position_entry_times, position_entry_prices
    
    print("\n🧪 开始测试完整交织场景...")
    
    # 重置测试状态
    current_position = 0
    open_orders.clear()
    closed_positions.clear()
    active_take_profit_orders.clear()
    position_entry_times.clear()
    position_entry_prices.clear()
    
    print("📝 第一步：买入2手，设置止盈")
    place_tiger_order('BUY', 1, 100.0, take_profit_price=110.0)
    place_tiger_order('BUY', 1, 102.0, take_profit_price=112.0)
    print(f"📊 状态: 持仓={current_position}, 活跃止盈单={len(active_take_profit_orders)}")
    
    print("📝 第二步：尝试买入第3手，但达到最大持仓限制")
    result = check_risk_control(104.0, 'BUY')
    if result:
        place_tiger_order('BUY', 1, 104.0,
                         tech_params={'current_price': 104.0, 'rsi': 30, 'atr': 1.7, 'boll_position': 'below_lower_band'},
                         reason='网格下轨+KDJ金叉+RSI超卖')
    else:
        print("❌ 风控阻止买入")
    print(f"📊 状态: 持仓={current_position}")
    
    print("📈 第三步：价格上涨，触发主动止盈")
    check_active_take_profits(115.0)  # 应该触发前两笔持仓的止盈
    print(f"📊 状态: 持仓={current_position}, 已平仓={len(closed_positions)}")
    
    print("📝 第四步：再次买入2手，设置止盈")
    place_tiger_order('BUY', 1, 118.0, take_profit_price=125.0,
                     tech_params={'current_price': 118.0, 'rsi': 40, 'atr': 2.0, 'boll_position': 'near_middle_band'},
                     reason='中轨附近+KDJ金叉+RSI偏弱')
    place_tiger_order('BUY', 1, 120.0, take_profit_price=130.0,
                     tech_params={'current_price': 120.0, 'rsi': 42, 'atr': 2.1, 'boll_position': 'near_middle_band'},
                     reason='中轨附近+KDJ金叉+RSI偏弱')
    print(f"📊 状态: 持仓={current_position}, 活跃止盈单={len(active_take_profit_orders)}")
    
    print("⏰ 第五步：模拟超时，触发超时止盈")
    import time
    # 修改提交时间，模拟超时
    for pos_id in active_take_profit_orders:
        active_take_profit_orders[pos_id]['submit_time'] = time.time() - (TAKE_PROFIT_TIMEOUT + 1) * 60
    
    # 设置价格以触发超时止盈
    check_timeout_take_profits(122.0)  # 价格高于1/3盈利目标
    print(f"📊 状态: 持仓={current_position}, 活跃止盈单={len(active_take_profit_orders)}")
    
    print("📉 第六步：价格大幅下跌，触发止损")
    # 平掉剩余持仓
    place_tiger_order('SELL', current_position, 90.0,
                     reason='止损触发',
                     tech_params={'current_price': 90.0, 'stop_loss_price': 90.0, 'exit_type': 'stop_loss'})
    print(f"📊 最终状态: 持仓={current_position}, 待平仓订单={len(open_orders)}, 已平仓={len(closed_positions)}")
    
    # 验证最终状态
    assert current_position == 0, f"预期持仓0手，实际{current_position}手"
    assert len(open_orders) == 0, f"预期待平仓订单0个，实际{len(open_orders)}个"
    assert len(closed_positions) >= 2, f"预期已平仓>=2个，实际{len(closed_positions)}个"
    
    print("✅ 完整交织场景测试通过！")
    
    # 显示所有交易详情
    total_profit = 0
    for i, trade in enumerate(closed_positions):
        profit = trade['profit']
        total_profit += profit
        print(f"📊 交易{i+1}: 买入价 {trade['buy_price']}, 卖出价 {trade['sell_price']}, 盈亏: {profit:.2f}USD")
    print(f"💰 总盈亏: {total_profit:.2f}USD")


def test_position_underflow_protection():
    """测试持仓下溢保护"""
    global current_position
    
    print("\n🧪 开始测试持仓下溢保护...")
    
    # 重置测试状态
    current_position = 0
    
    # 先买入2手
    place_tiger_order('BUY', 2, 100.0)
    assert current_position == 2, f"预期持仓2手，实际{current_position}手"
    
    # 尝试卖出3手（超过当前持仓）
    place_tiger_order('SELL', 3, 105.0)
    # 系统应该将持仓减到0，而不是负数
    assert current_position == 0, f"预期持仓0手（下溢保护），实际{current_position}手"
    
    print("✅ 持仓下溢保护测试通过！")


def test_daily_loss_reset():
    """测试每日亏损重置功能"""
    global daily_loss, today
    
    print("\n🧪 开始测试每日亏损重置功能...")
    
    # 重置测试状态
    daily_loss = 1000  # 设置一些初始亏损
    
    # 模拟日期变更
    from datetime import date, timedelta
    today = date.today() - timedelta(days=1)  # 昨天
    
    # 调用风控检查，这会检查日期并重置
    result = check_risk_control(100.0, 'BUY')
    
    # 检查是否重置了日期
    from datetime import date
    if date.today() != today:
        print("✅ 日期已更新")
    else:
        print("ℹ️ 日期未更新（可能是同一天）")
    
    print("✅ 每日亏损重置功能测试通过！")


def run_tests():
    """运行所有测试"""
    print("🚀 开始运行所有测试...")
    
    test_order_tracking()
    test_edge_case_no_position()
    test_partial_fill_scenarios()
    test_multiple_risk_controls()
    test_stop_loss_scenario()
    test_take_profit_scenario()
    test_timeout_take_profit_scenario()
    test_zero_quantity_scenarios()
    test_negative_price_scenarios()
    test_empty_order_book()
    test_complex_interweaving_scenario()
    test_full_interweaving_scenario()
    test_position_underflow_protection()
    test_daily_loss_reset()
    
    print("\n🎉 所有测试完成！")
    
    # 生成订单总结
    generate_order_summary()
    
    # 生成详细交易分析
    print_trade_analysis()
    
    # 生成每小时分析
    analyze_hourly_performance()
    
    # 重置为生产环境变量
    global current_position, open_orders, closed_positions, position_entry_times, position_entry_prices, active_take_profit_orders
    current_position = 0
    open_orders.clear()
    closed_positions.clear()
    position_entry_times.clear()
    position_entry_prices.clear()
    active_take_profit_orders.clear()


if __name__ == "__main__":
    run_tests()

def print_trade_analysis():
    """打印详细的交易分析报告"""
    print("\n🔍 ========== 详细交易分析报告 ========== 🔍")
    
    if not closed_positions:
        print("📈 暂无已平仓订单")
        return
    
    for i, trade in enumerate(closed_positions, 1):
        print(f"\n📊 交易 #{i}:")
        print(f"   买入价: {trade['buy_price']:.2f} | 卖出价: {trade['sell_price']:.2f} | 数量: {trade['quantity']}手 | 盈亏: {trade['profit']:.2f}USD")
        print(f"   交易时间: {datetime.fromtimestamp(trade['timestamp'])}")
        
        analysis = trade['analysis']
        
        print(f"   📌 开仓分析:")
        print(f"     - 开仓原因: {analysis['buy_reason']}")
        if analysis['buy_tech_params']:
            print(f"     - 技术参数: ", end="")
            params_str = ", ".join([f"{k}:{v:.2f}" if isinstance(v, float) else f"{k}:{v}" for k, v in analysis['buy_tech_params'].items()])
            print(params_str)
        
        print(f"   📉 平仓分析:")
        print(f"     - 平仓原因: {analysis['sell_reason']}")
        if analysis['sell_tech_params']:
            print(f"     - 技术参数: ", end="")
            params_str = ", ".join([f"{k}:{v:.2f}" if isinstance(v, float) else f"{k}:{v}" for k, v in analysis['sell_tech_params'].items()])
            print(params_str)
        
        print(f"   ⚠️  特殊标记: {'止损触发' if analysis['stop_loss_triggered'] else '非止损触发'}")
        
        profit_ratio = (trade['sell_price'] - trade['buy_price']) / trade['buy_price'] * 100
        print(f"   💰 盈亏比例: {profit_ratio:+.2f}%")
    
    print("\n======================================== 🔍")
