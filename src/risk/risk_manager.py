"""风险控制管理器"""

class RiskManager:
    """风险控制"""
    
    def __init__(self, config=None):
        self.config = config or {
            'max_loss_per_trade': 0.02,  # 单笔最大亏损2%
            'max_daily_loss': 0.05,       # 日最大亏损5%
            'max_position_size': 0.35     # 最大仓位35%
        }
        self.daily_loss = 0.0
    
    def check_signal(self, signal, account_value=10000):
        """检查信号是否符合风险要求"""
        if signal['action'] == 'HOLD':
            return True
        
        # 检查仓位
        if signal.get('position_size', 0) > self.config['max_position_size']:
            print(f"⚠️ 仓位超限: {signal['position_size']}")
            return False
        
        # 检查日内亏损
        if self.daily_loss >= self.config['max_daily_loss'] * account_value:
            print(f"⚠️ 达到日内止损")
            return False
        
        return True
    
    def update_daily_loss(self, loss):
        """更新日内亏损"""
        self.daily_loss += loss
    
    def reset_daily(self):
        """重置日内统计"""
        self.daily_loss = 0.0
