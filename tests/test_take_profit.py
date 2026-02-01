import sys
import os
import time
import unittest.mock as mock
from unittest.mock import MagicMock

# Add project path
sys.path.insert(0, '/home/cx/tigertrade')

# Import the module to test
from src import tiger1

def test_active_take_profit_logic():
    """Test the active take profit feature"""
    print("Testing active take profit logic...")
    
    # Reset global state
    tiger1.current_position = 1
    tiger1.position_entry_times = {0: time.time() - 31*60}  # Position entered 31 minutes ago
    tiger1.position_entry_prices = {0: 100.0}  # Entry price was 100
    tiger1.active_take_profit_orders = {0: {
        'target_price': 105.0,  # Target was 105
        'submit_time': time.time() - 31*60,  # Order submitted 31 minutes ago
        'quantity': 1
    }}
    
    # Mock the place_tiger_order function to track calls
    with mock.patch.object(tiger1, 'place_tiger_order') as mock_place_order:
        # Current price is 102 (> 100 * 1.02 = 102 minimum profit target)
        if not hasattr(tiger1, 'check_active_take_profits'):
            raise AssertionError("check_active_take_profits 不存在，不允许 Skip，测试目的就是测出问题")
        result = tiger1.check_active_take_profits(102.0)
        
        # Verify that SELL order was placed
        assert result == True, "Active take profit should be triggered"
        mock_place_order.assert_called_once_with('SELL', 1, 102.0)
        print("✅ Test 1 PASSED: Active take profit triggered due to minimum profit target reached")
    
    # Reset for next test
    tiger1.current_position = 1
    tiger1.position_entry_times = {0: time.time() - 31*60}  # Position entered 31 minutes ago
    tiger1.position_entry_prices = {0: 100.0}  # Entry price was 100
    tiger1.active_take_profit_orders = {0: {
        'target_price': 105.0,  # Target was 105
        'submit_time': time.time() - 31*60,  # Order submitted 31 minutes ago
        'quantity': 1
    }}
    
    # Mock the place_tiger_order function to track calls
    with mock.patch.object(tiger1, 'place_tiger_order') as mock_place_order:
        # Current price is 101 (< 100 * 1.02 = 102 minimum profit target) but timeout occurred
        result = tiger1.check_active_take_profits(101.0)
        
        # Verify that SELL order was placed due to timeout
        assert result == True, "Active take profit should be triggered due to timeout"
        mock_place_order.assert_called_once_with('SELL', 1, 101.0)
        print("✅ Test 2 PASSED: Active take profit triggered due to timeout")
    
    # Test case when conditions are not met
    tiger1.current_position = 1
    tiger1.position_entry_times = {0: time.time() - 10*60}  # Position entered 10 minutes ago (not timeout yet)
    tiger1.position_entry_prices = {0: 100.0}  # Entry price was 100
    tiger1.active_take_profit_orders = {0: {
        'target_price': 105.0,  # Target was 105
        'submit_time': time.time() - 10*60,  # Order submitted 10 minutes ago
        'quantity': 1
    }}
    
    with mock.patch.object(tiger1, 'place_tiger_order') as mock_place_order:
        # Current price is 101 (< 100 * 1.02 = 102 minimum profit target) and no timeout
        result = tiger1.check_active_take_profits(101.0)
        
        # Verify that no SELL order was placed
        assert result == False, "Active take profit should NOT be triggered"
        mock_place_order.assert_not_called()
        print("✅ Test 3 PASSED: Active take profit NOT triggered when conditions not met")


def test_custom_parameters():
    """Test custom parameters for active take profit"""
    print("\nTesting custom parameters...")
    
    # Temporarily set custom values
    original_timeout = tiger1.TAKE_PROFIT_TIMEOUT
    original_ratio = tiger1.MIN_PROFIT_RATIO
    
    # Change parameters to test
    tiger1.TAKE_PROFIT_TIMEOUT = 45  # 45 minutes timeout
    tiger1.MIN_PROFIT_RATIO = 0.03   # 3% minimum profit
    
    # Reset state
    tiger1.current_position = 1
    tiger1.position_entry_times = {0: time.time() - 40*60}  # Position entered 40 minutes ago
    tiger1.position_entry_prices = {0: 100.0}  # Entry price was 100
    tiger1.active_take_profit_orders = {0: {
        'target_price': 105.0,
        'submit_time': time.time() - 40*60,  # Order submitted 40 minutes ago
        'quantity': 1
    }}
    
    with mock.patch.object(tiger1, 'place_tiger_order') as mock_place_order:
        # At 40 minutes, hasn't reached 45 minute timeout, and price 102.5 < 103 (3% profit)
        result = tiger1.check_active_take_profits(102.5)
        assert result == False, "Should not trigger with custom params not met"
        mock_place_order.assert_not_called()
        print("✅ Test 4 PASSED: Custom timeout (45 min) respected")
    
    # Now test when timeout is exceeded with custom params
    tiger1.position_entry_times = {0: time.time() - 50*60}  # Position entered 50 minutes ago
    tiger1.active_take_profit_orders = {0: {
        'target_price': 105.0,
        'submit_time': time.time() - 50*60,  # Order submitted 50 minutes ago
        'quantity': 1
    }}
    
    with mock.patch.object(tiger1, 'place_tiger_order') as mock_place_order:
        # At 50 minutes, exceeds 45 minute timeout, so should trigger even at 102.5 < 103
        result = tiger1.check_active_take_profits(102.5)
        assert result == True, "Should trigger when custom timeout exceeded"
        mock_place_order.assert_called_once_with('SELL', 1, 102.5)
        print("✅ Test 5 PASSED: Custom timeout (45 min) triggered correctly")
    
    # Restore original values
    tiger1.TAKE_PROFIT_TIMEOUT = original_timeout
    tiger1.MIN_PROFIT_RATIO = original_ratio


def run_integration_test():
    """Run an integration test with the actual strategy functions"""
    print("\nRunning integration test with strategy functions...")
    
    # Mock the external dependencies to avoid real API calls
    tiger1.quote_client = MagicMock()
    tiger1.trade_client = MagicMock()
    
    # Mock get_kline_data to return dummy data
    with mock.patch.object(tiger1, 'get_kline_data') as mock_get_kline:
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.__len__ = lambda s: 30
        mock_df.__getitem__ = lambda s, key: {
            'time': pd.date_range('2023-01-01', periods=30),
            'open': [100]*30,
            'high': [101]*30,
            'low': [99]*30,
            'close': [100]*30,
            'volume': [1000]*30
        }[key]
        mock_df.set_index = lambda s, key: mock_df
        mock_df.sort_index = lambda s: None
        mock_get_kline.return_value = mock_df
        
        # Also mock pandas for indicators
        import pandas as pd
        import numpy as np
        
        # Temporarily patch pandas and talib
        original_calc_ind = tiger1.calculate_indicators
        tiger1.calculate_indicators = lambda df1, df2: {
            '5m': {
                'boll_mid': 100.0,
                'boll_upper': 102.0,
                'boll_lower': 98.0,
                'rsi': 50.0,
                'atr': 1.0
            },
            '1m': {
                'rsi': 45.0,
                'close': 100.0,
                'volume': 1000.0
            }
        }
        
        # Setup for active take profit test
        tiger1.current_position = 1
        tiger1.position_entry_times = {0: time.time() - 35*60}  # 35 mins ago
        tiger1.position_entry_prices = {0: 100.0}
        tiger1.active_take_profit_orders = {0: {
            'target_price': 105.0,
            'submit_time': time.time() - 35*60,
            'quantity': 1
        }}
        
        # Run a strategy function to test integration
        with mock.patch.object(tiger1, 'place_tiger_order') as mock_place_order:
            tiger1.grid_trading_strategy_pro1()
            # Should trigger active take profit since 35 > 30 min timeout
            # Find if SELL order was called
            sell_calls = [call for call in mock_place_order.call_args_list if call[0][0] == 'SELL']
            assert len(sell_calls) > 0, "Active take profit should be called in strategy"
            print("✅ Integration test PASSED: Active take profit integrated with strategy")
        
        # Restore original function
        tiger1.calculate_indicators = original_calc_ind


if __name__ == "__main__":
    print("Starting tests for active take profit functionality...\n")
    
    try:
        test_active_take_profit_logic()
        test_custom_parameters()
        run_integration_test()
        
        print("\n🎉 All tests passed! Active take profit functionality is working correctly.")
        print(f"Current settings: TIMEOUT={tiger1.TAKE_PROFIT_TIMEOUT}min, MIN_PROFIT_RATIO={tiger1.MIN_PROFIT_RATIO*100}%")
        print("\nTo customize: set environment variables TAKE_PROFIT_TIMEOUT and MIN_PROFIT_RATIO before running")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()