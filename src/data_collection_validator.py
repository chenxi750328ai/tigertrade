#!/usr/bin/env python3
"""
数据采集验证器
确保：
1. 使用真实API，拒绝Mock数据
2. 检查所有异常和错误
3. 验证数据质量
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class DataCollectionValidator:
    """数据采集验证器"""
    
    def __init__(self, strict_mode=True):
        """
        Args:
            strict_mode: 严格模式，发现问题立即终止
        """
        self.strict_mode = strict_mode
        self.errors = []
        self.warnings = []
    
    def validate_api_initialization(self, api_manager, allow_mock=False, warn_on_mock=True) -> bool:
        """
        验证API初始化状态并明确标识数据来源
        
        参数：
            api_manager: API管理器
            allow_mock: 是否允许Mock模式（默认False）
            warn_on_mock: 如果是Mock模式是否警告（默认True）
        
        返回：
            bool: 验证是否通过
        """
        print("=" * 80)
        print("🔍 步骤1: 检查API状态")
        print("=" * 80)
        
        # 检查API是否初始化
        if api_manager.quote_api is None:
            warning = "⚠️  Quote API未初始化 (None)"
            self.warnings.append(warning)
            print(warning)
            print("   → 程序可能会回退到Mock数据")
        
        if api_manager.trade_api is None:
            warning = "⚠️  Trade API未初始化 (None)"
            self.warnings.append(warning)
            print(warning)
        
        # 明确标识数据来源
        is_mock = False
        
        if api_manager.quote_api is None or api_manager.is_mock_mode:
            is_mock = True
        
        quote_api_type = type(api_manager.quote_api).__name__ if api_manager.quote_api else 'None'
        trade_api_type = type(api_manager.trade_api).__name__ if api_manager.trade_api else 'None'
        
        if 'Mock' in quote_api_type or 'Mock' in trade_api_type:
            is_mock = True
        
        # 打印数据来源
        print()
        print("📊 数据来源:")
        if is_mock:
            print("   🔶 当前使用: Mock数据（模拟数据）")
            print(f"   API状态: Mock模式={api_manager.is_mock_mode}")
            print(f"   Quote API: {quote_api_type}")
            print(f"   Trade API: {trade_api_type}")
        else:
            print("   ✅ 当前使用: 真实API数据")
            print(f"   Quote API: {quote_api_type}")
            print(f"   Trade API: {trade_api_type}")
        print()
        
        # 根据配置决定是否接受Mock
        if is_mock:
            if not allow_mock:
                error = "❌ 检测到Mock数据，但当前配置不允许使用Mock"
                self.errors.append(error)
                print(error)
                print("   提示：如果要使用Mock数据，请设置 allow_mock=True")
                if self.strict_mode:
                    raise RuntimeError(error)
                return False
            elif warn_on_mock:
                warning = "⚠️  警告：当前使用Mock数据"
                self.warnings.append(warning)
                print(warning)
                print("   → Mock数据适合开发测试")
                print("   → 训练模型建议使用真实数据")
                print()
        
        return True
    
    def validate_kline_data(self, df: pd.DataFrame, period: str, 
                           expected_min_rows: int = 100) -> bool:
        """
        验证K线数据质量
        
        检查：
        1. 数据不为空
        2. 包含必需的列
        3. 数据量足够
        4. 不是Mock数据特征
        5. 价格有合理波动
        """
        print("=" * 80)
        print(f"🔍 步骤2: 验证{period}K线数据")
        print("=" * 80)
        
        # 检查1: 数据不为空
        if df is None or df.empty:
            error = f"❌ {period}数据为空"
            self.errors.append(error)
            print(error)
            if self.strict_mode:
                raise ValueError(error)
            return False
        
        print(f"✅ 数据行数: {len(df)}")
        
        # 检查2: 必需的列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = set(required_cols) - set(df.columns)
        
        if missing_cols:
            error = f"❌ {period}数据缺少必需的列: {missing_cols}"
            self.errors.append(error)
            print(error)
            if self.strict_mode:
                raise ValueError(error)
            return False
        
        print(f"✅ 包含所有必需列: {required_cols}")
        
        # 检查3: 数据量
        if len(df) < expected_min_rows:
            error = f"❌ {period}数据量不足: {len(df)} < {expected_min_rows}"
            self.errors.append(error)
            print(error)
            if self.strict_mode:
                raise ValueError(error)
            return False
        
        print(f"✅ 数据量充足: {len(df)} >= {expected_min_rows}")
        
        # 检查4: Mock数据特征检测
        close_mean = df['close'].mean()
        close_first = df['close'].iloc[0]
        close_std = df['close'].std()
        
        # Mock数据特征：价格从90开始，均值约92.5
        is_mock_like = (
            85 < close_first < 95 and  # 首个价格在90附近
            87 < close_mean < 97 and   # 平均价格在92附近
            close_std < 2.0            # 标准差很小
        )
        
        if is_mock_like:
            warning = f"⚠️  {period}数据疑似Mock特征："
            self.warnings.append(warning)
            print(warning)
            print(f"   首个close: {close_first:.2f} (Mock应该≈90)")
            print(f"   平均close: {close_mean:.2f} (Mock应该≈92.5)")
            print(f"   标准差: {close_std:.4f}")
            print("   → 请确认这是真实市场数据！")
            print()
        
        # 检查5: 价格波动合理性
        if close_std == 0:
            error = f"❌ {period}价格无波动（标准差=0）"
            self.errors.append(error)
            print(error)
            if self.strict_mode:
                raise ValueError(error)
            return False
        
        # 检查OHLC关系
        invalid_ohlc = (
            (df['high'] < df['close']).any() or
            (df['low'] > df['close']).any() or
            (df['high'] < df['low']).any()
        )
        
        if invalid_ohlc:
            error = f"❌ {period}数据OHLC关系不合理"
            self.errors.append(error)
            print(error)
            print("   high应该 >= close >= low")
            if self.strict_mode:
                raise ValueError(error)
            return False
        
        print(f"✅ 价格统计:")
        print(f"   均值: {close_mean:.2f}")
        print(f"   标准差: {close_std:.4f}")
        print(f"   最小值: {df['close'].min():.2f}")
        print(f"   最大值: {df['close'].max():.2f}")
        print()
        
        return True
    
    def validate_features(self, df_features: pd.DataFrame) -> bool:
        """
        验证计算出的特征质量
        
        检查：
        1. 特征不为空
        2. 关键特征不是常量
        3. 没有过多的NaN
        """
        print("=" * 80)
        print("🔍 步骤3: 验证特征质量")
        print("=" * 80)
        
        if df_features is None or df_features.empty:
            error = "❌ 特征数据为空"
            self.errors.append(error)
            print(error)
            if self.strict_mode:
                raise ValueError(error)
            return False
        
        print(f"✅ 特征数据行数: {len(df_features)}")
        
        # 关键特征列表
        key_features = [
            'price_change_1', 'price_change_5', 'volatility',
            'rsi_1m', 'rsi_5m', 'atr',
            'boll_upper', 'boll_lower', 'boll_position'
        ]
        
        constant_features = []
        
        for feature in key_features:
            if feature not in df_features.columns:
                warning = f"⚠️  特征 {feature} 不存在"
                self.warnings.append(warning)
                print(warning)
                continue
            
            unique_count = df_features[feature].nunique()
            
            # 检查常量特征
            if unique_count == 1:
                constant_features.append(feature)
                error = f"❌ 特征 {feature} 是常量！"
                self.errors.append(error)
                print(error)
                print(f"   唯一值: {df_features[feature].unique()[:5]}")
                print(f"   这表明特征计算失败！")
            else:
                print(f"✅ {feature:20s}: {unique_count:5d} 个唯一值")
        
        if constant_features:
            error = f"❌ 发现 {len(constant_features)} 个常量特征: {constant_features}"
            self.errors.append(error)
            print()
            print(error)
            print("   → 这通常意味着：")
            print("      1. 数据窗口长度不足")
            print("      2. 技术指标计算返回了默认值")
            print("      3. 数据质量有问题")
            if self.strict_mode:
                raise ValueError(error)
            return False
        
        print()
        print(f"✅ 所有关键特征都有变化")
        print()
        
        return True
    
    def check_for_exceptions_in_output(self, output: str) -> List[str]:
        """
        检查输出中的异常和错误
        
        Returns:
            List[str]: 发现的异常列表
        """
        print("=" * 80)
        print("🔍 步骤4: 检查异常和错误")
        print("=" * 80)
        
        keywords = [
            'Exception', 'Error', 'Traceback', 'FAILED',
            '异常', '错误', '失败',
            'AttributeError', 'ValueError', 'KeyError',
            'NoneType'
        ]
        
        found_errors = []
        lines = output.split('\n')
        
        for i, line in enumerate(lines):
            for keyword in keywords:
                if keyword in line:
                    found_errors.append(f"第{i+1}行: {line.strip()}")
                    break
        
        if found_errors:
            print(f"❌ 发现 {len(found_errors)} 个异常/错误:")
            for err in found_errors[:10]:  # 只显示前10个
                print(f"   {err}")
            if len(found_errors) > 10:
                print(f"   ... 还有 {len(found_errors) - 10} 个")
            print()
            
            if self.strict_mode:
                raise RuntimeError(f"发现异常！请检查并修复！")
        else:
            print("✅ 未发现异常")
            print()
        
        return found_errors
    
    def get_summary(self) -> Dict[str, any]:
        """获取验证摘要"""
        return {
            'errors': self.errors,
            'warnings': self.warnings,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'passed': len(self.errors) == 0
        }
    
    def print_summary(self):
        """打印验证摘要"""
        print()
        print("=" * 80)
        print("📊 验证摘要")
        print("=" * 80)
        
        if self.errors:
            print(f"❌ 发现 {len(self.errors)} 个错误:")
            for i, err in enumerate(self.errors, 1):
                print(f"   {i}. {err}")
            print()
        
        if self.warnings:
            print(f"⚠️  发现 {len(self.warnings)} 个警告:")
            for i, warn in enumerate(self.warnings, 1):
                print(f"   {i}. {warn}")
            print()
        
        if not self.errors and not self.warnings:
            print("✅ 所有验证通过！")
            print()
        
        return len(self.errors) == 0


def force_real_api_initialization():
    """
    强制初始化真实API
    
    如果初始化失败，程序终止
    """
    print("=" * 80)
    print("🚀 强制初始化真实API")
    print("=" * 80)
    
    try:
        from api_adapter import api_manager
        
        # 检查是否已初始化
        if api_manager.quote_api is not None:
            print("ℹ️  API已经初始化")
            print(f"   Quote API: {type(api_manager.quote_api).__name__}")
            print(f"   Trade API: {type(api_manager.trade_api).__name__}")
            print(f"   Mock模式: {api_manager.is_mock_mode}")
            print()
            
            # 验证不是Mock
            if api_manager.is_mock_mode:
                raise RuntimeError("❌ API处于Mock模式，不可接受！")
            
            return api_manager
        
        # 尝试初始化生产API
        print("🔧 初始化生产API...")
        
        # 检查配置文件
        import os
        config_path = '/home/cx/openapicfg_dem'
        
        if not os.path.exists(config_path):
            raise RuntimeError(f"❌ API配置目录不存在: {config_path}")
        
        print(f"✅ 配置目录存在: {config_path}")
        
        # 初始化
        api_manager.initialize_production_apis(config_path)
        
        # 验证
        if api_manager.quote_api is None:
            raise RuntimeError("❌ API初始化后仍为None")
        
        if api_manager.is_mock_mode:
            raise RuntimeError("❌ API初始化后仍处于Mock模式")
        
        print(f"✅ API初始化成功")
        print(f"   Quote API: {type(api_manager.quote_api).__name__}")
        print(f"   Trade API: {type(api_manager.trade_api).__name__}")
        print()
        
        return api_manager
        
    except Exception as e:
        print(f"❌ API初始化失败: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("🛑 由于API初始化失败，程序终止")
        print("   → 不允许使用Mock数据训练模型")
        print("   → 请修复API配置后重试")
        sys.exit(1)


if __name__ == "__main__":
    print("数据采集验证器测试")
    print()
    
    # 测试API验证
    try:
        api_manager = force_real_api_initialization()
        
        validator = DataCollectionValidator(strict_mode=True)
        validator.validate_api_initialization(api_manager)
        
        print("✅ 测试通过")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
