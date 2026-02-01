#!/usr/bin/env python3
"""
期货合约管理器
支持动态选择和切换期货合约
"""

import datetime
from typing import List, Optional
import re


class FuturesContractManager:
    """期货合约管理器"""
    
    # 支持的基础合约代码
    SUPPORTED_BASE_SYMBOLS = {
        'SIL': '白银',
        'AU': '黄金',
        'CU': '铜',
        'AL': '铝',
        'NQ': '纳指',
        'ES': '标普',
        'YM': '道指'
    }
    
    def __init__(self, base_symbol: str = 'SIL'):
        """
        初始化
        
        Args:
            base_symbol: 基础合约代码（如SIL、AU等）
        """
        if base_symbol not in self.SUPPORTED_BASE_SYMBOLS:
            raise ValueError(f"不支持的合约代码: {base_symbol}")
        
        self.base_symbol = base_symbol
    
    def get_current_contract(self, offset_months: int = 0) -> str:
        """
        获取当前活跃合约
        
        Args:
            offset_months: 月份偏移量（0=当前月，1=下月，2=后月...）
            
        Returns:
            合约代码，如 'SIL2603'
        """
        now = datetime.datetime.now()
        target_date = now + datetime.timedelta(days=30 * offset_months)
        
        year = target_date.year % 100  # 取后两位：2026 -> 26
        month = target_date.month
        
        # 如果当前月份已过半，自动切换到下月合约
        if now.day > 15 and offset_months == 0:
            month += 1
            if month > 12:
                month = 1
                year += 1
        
        return f"{self.base_symbol}{year:02d}{month:02d}"
    
    def get_next_contracts(self, count: int = 4) -> List[str]:
        """
        获取未来几个月的合约列表
        
        Args:
            count: 获取合约数量
            
        Returns:
            合约代码列表，如 ['SIL2603', 'SIL2604', 'SIL2605', 'SIL2606']
        """
        contracts = []
        for i in range(count):
            contracts.append(self.get_current_contract(offset_months=i))
        return contracts
    
    def parse_contract(self, contract_code: str) -> dict:
        """
        解析合约代码
        
        Args:
            contract_code: 合约代码，如 'SIL2603'
            
        Returns:
            解析结果 {'base': 'SIL', 'year': 26, 'month': 3, 'full_year': 2026}
        """
        # 匹配格式：字母开头 + 2位年份 + 2位月份
        pattern = r'([A-Z]+)(\d{2})(\d{2})'
        match = re.match(pattern, contract_code)
        
        if not match:
            raise ValueError(f"无效的合约代码格式: {contract_code}")
        
        base = match.group(1)
        year = int(match.group(2))
        month = int(match.group(3))
        
        # 推算完整年份（假设合约在当前年份的前后10年内）
        current_year = datetime.datetime.now().year
        century = (current_year // 100) * 100
        full_year = century + year
        
        # 如果推算的年份比当前年份早超过50年，加100年
        if full_year < current_year - 50:
            full_year += 100
        
        return {
            'base': base,
            'year': year,
            'month': month,
            'full_year': full_year,
            'name': self.SUPPORTED_BASE_SYMBOLS.get(base, '未知')
        }
    
    def is_contract_expired(self, contract_code: str, days_buffer: int = 5) -> bool:
        """
        判断合约是否已过期或即将过期
        
        Args:
            contract_code: 合约代码
            days_buffer: 缓冲天数（提前几天算过期）
            
        Returns:
            True: 已过期或即将过期，False: 仍有效
        """
        info = self.parse_contract(contract_code)
        
        # 合约到期日通常是该月的第三个周五附近
        # 简化处理：该月25日算过期
        expiry_date = datetime.datetime(info['full_year'], info['month'], 25)
        buffer_date = expiry_date - datetime.timedelta(days=days_buffer)
        
        return datetime.datetime.now() >= buffer_date
    
    def get_active_contract(self) -> str:
        """
        智能获取当前最活跃的合约
        如果当前合约即将过期，自动切换到下月合约
        
        Returns:
            活跃合约代码
        """
        current = self.get_current_contract()
        
        if self.is_contract_expired(current):
            # 当前合约即将过期，返回下月合约
            next_contract = self.get_current_contract(offset_months=1)
            print(f"⚠️  当前合约{current}即将过期，自动切换到{next_contract}")
            return next_contract
        
        return current
    
    @classmethod
    def validate_contract(cls, contract_code: str) -> bool:
        """
        验证合约代码格式是否正确
        
        Args:
            contract_code: 合约代码
            
        Returns:
            True: 有效，False: 无效
        """
        pattern = r'^[A-Z]+\d{4}$'
        return bool(re.match(pattern, contract_code))


def demo():
    """演示用法"""
    print("=" * 80)
    print("期货合约管理器演示")
    print("=" * 80)
    
    # 创建管理器
    manager = FuturesContractManager('SIL')
    
    # 获取当前合约
    current = manager.get_active_contract()
    print(f"\n当前活跃合约: {current}")
    
    # 获取未来4个月的合约
    future_contracts = manager.get_next_contracts(4)
    print(f"\n未来合约列表:")
    for i, contract in enumerate(future_contracts, 1):
        info = manager.parse_contract(contract)
        expired = manager.is_contract_expired(contract)
        status = "⚠️ 即将过期" if expired else "✅ 有效"
        print(f"  {i}. {contract} - {info['full_year']}年{info['month']}月 {status}")
    
    # 解析合约示例
    print(f"\n解析合约 'SIL2603':")
    info = manager.parse_contract('SIL2603')
    print(f"  基础代码: {info['base']}")
    print(f"  名称: {info['name']}")
    print(f"  年份: {info['full_year']}")
    print(f"  月份: {info['month']}")
    
    # 验证格式
    print(f"\n格式验证:")
    test_codes = ['SIL2603', 'AU2604', 'INVALID', 'SIL26']
    for code in test_codes:
        valid = FuturesContractManager.validate_contract(code)
        print(f"  {code}: {'✅ 有效' if valid else '❌ 无效'}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    demo()
