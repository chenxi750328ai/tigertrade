#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大模型辅助时段特征分析模块
使用大模型分析时段特征，识别异常模式，提供参数建议
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import json

# 添加tigertrade目录到路径
sys.path.insert(0, '/home/cx/tigertrade')

from scripts.analysis.time_period_analyzer import TimePeriodAnalyzer


class LLMPeriodAnalyzer:
    """大模型辅助时段特征分析器"""
    
    def __init__(self, period_analyzer: TimePeriodAnalyzer):
        """
        初始化大模型分析器
        
        Args:
            period_analyzer: 时段分析器实例
        """
        self.period_analyzer = period_analyzer
        self.analysis_prompt_template = self._load_analysis_prompt()
    
    def _load_analysis_prompt(self) -> str:
        """加载分析提示词模板"""
        return """
你是一个专业的期货交易时段特征分析专家。请分析以下时段数据，识别特征模式，并提供策略建议。

## 分析任务
1. **时段波动率模式识别**: 识别不同时段的波动率特征和规律
2. **时段滑点率分析**: 分析时段滑点率与波动率的关系
3. **异常时段检测**: 识别偏离正常模式的时段
4. **参数建议**: 基于分析结果建议时段参数（网格间距、仓位上限等）

## 输入数据
{period_data}

## 参考规则（仅供参考，优先使用数据分析结果）
{reference_rules}

## 输出要求
请以JSON格式输出分析结果，包含：
1. period_patterns: 各时段的特征模式描述
2. anomalies: 异常时段列表及原因
3. recommendations: 参数建议
4. risk_warnings: 风险提示

## 分析重点
- 关注时段波动率与滑点率的关系
- 识别低波动但高滑点的时段（需要更大网格间距）
- 识别高波动但低滑点的时段（可以更小网格间距）
- 考虑时段流动性的影响
"""
    
    def analyze_with_llm(self, analysis_result: Dict, 
                        reference_rules: Dict = None,
                        use_api: bool = False) -> Dict:
        """
        使用大模型分析时段特征
        
        Args:
            analysis_result: 数据驱动的分析结果
            reference_rules: 参考规则（可选）
            use_api: 是否使用API调用大模型（False时返回模拟结果）
            
        Returns:
            大模型分析结果
        """
        if not analysis_result:
            return {}
        
        # 准备分析数据
        period_data = self._prepare_analysis_data(analysis_result)
        
        # 构建提示词
        prompt = self.analysis_prompt_template.format(
            period_data=json.dumps(period_data, indent=2, ensure_ascii=False),
            reference_rules=json.dumps(reference_rules or {}, indent=2, ensure_ascii=False)
        )
        
        if use_api:
            # TODO: 集成实际的大模型API调用
            # 这里可以调用OpenAI、Claude、或其他大模型API
            result = self._call_llm_api(prompt)
        else:
            # 模拟分析结果（用于测试）
            result = self._simulate_llm_analysis(period_data)
        
        return result
    
    def _prepare_analysis_data(self, analysis_result: Dict) -> Dict:
        """准备分析数据"""
        period_configs = analysis_result.get('period_configs', {})
        period_stats = analysis_result.get('period_stats', {})
        liquidity_stats = analysis_result.get('liquidity_stats', {})
        
        prepared_data = {}
        for period, config in period_configs.items():
            stats = period_stats.get(period, {})
            liquidity = liquidity_stats.get(period, {})
            
            prepared_data[period] = {
                'volatility': config.get('volatility', 0),
                'volatility_pct': stats.get('volatility_pct', 0),
                'slippage_rate': config.get('slippage_rate', 0),
                'balance_threshold': config.get('balance_threshold', 0),
                'max_position': config.get('max_position', 0),
                'avg_atr': stats.get('avg_atr', 0),
                'atr_pct': stats.get('atr_pct', 0),
                'price_range_pct': stats.get('price_range_pct', 0),
                'mean_volume': liquidity.get('mean_volume', 0),
                'volume_stability': liquidity.get('volume_stability', 0),
                'data_quality': config.get('data_quality', {})
            }
        
        return prepared_data
    
    def _call_llm_api(self, prompt: str) -> Dict:
        """
        调用大模型API（待实现）
        
        Args:
            prompt: 分析提示词
            
        Returns:
            大模型分析结果
        """
        # TODO: 实现实际的大模型API调用
        # 示例：
        # import openai
        # response = openai.ChatCompletion.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # result = json.loads(response.choices[0].message.content)
        # return result
        
        print("⚠️ 大模型API调用功能待实现，返回模拟结果")
        return self._simulate_llm_analysis({})
    
    def _simulate_llm_analysis(self, period_data: Dict) -> Dict:
        """模拟大模型分析结果（用于测试）"""
        anomalies = []
        recommendations = []
        
        # 检测异常：低波动但高滑点
        for period, data in period_data.items():
            volatility = data.get('volatility', 0)
            slippage_rate = data.get('slippage_rate', 0)
            
            if volatility < 0.8 and slippage_rate > 0.02:
                anomalies.append({
                    'period': period,
                    'type': '低波动高滑点',
                    'description': f'波动率{volatility*100:.1f}%较低，但滑点率{slippage_rate*100:.2f}%较高，可能导致滑点侵蚀利润',
                    'suggestion': '建议增大网格间距，降低仓位上限'
                })
            
            # 检测异常：高波动但低滑点
            if volatility > 1.8 and slippage_rate < 0.01:
                anomalies.append({
                    'period': period,
                    'type': '高波动低滑点',
                    'description': f'波动率{volatility*100:.1f}%较高，但滑点率{slippage_rate*100:.2f}%较低，适合更积极的交易策略',
                    'suggestion': '可以考虑减小网格间距，提高仓位上限'
                })
        
        # 生成建议
        for period, data in period_data.items():
            balance_threshold = data.get('balance_threshold', 0)
            current_max_position = data.get('max_position', 0)
            
            recommendations.append({
                'period': period,
                'grid_spacing': f'建议网格间距 ≥ {balance_threshold:.4f}美元',
                'max_position': f'建议最大仓位: {current_max_position}手',
                'order_type': '建议使用限价单，偏离幅度根据滑点率调整'
            })
        
        return {
            'period_patterns': {
                'summary': '基于数据分析的时段特征模式',
                'details': period_data
            },
            'anomalies': anomalies,
            'recommendations': recommendations,
            'risk_warnings': [
                '时段特征可能因市场变化而改变，建议定期重新分析',
                '重大事件（如经济数据发布、央行政策）可能影响时段特征',
                '低波动时段的滑点率可能较高，需要特别注意'
            ]
        }
    
    def detect_event_impact(self, analysis_result: Dict, 
                           event_dates: List[str] = None) -> Dict:
        """
        检测重大事件对时段特征的影响
        
        Args:
            analysis_result: 分析结果
            event_dates: 重大事件日期列表（格式：YYYY-MM-DD）
            
        Returns:
            事件影响分析结果
        """
        if not event_dates:
            return {}
        
        # TODO: 实现事件影响分析
        # 1. 提取事件前后的时段数据
        # 2. 对比事件前后的时段特征变化
        # 3. 识别受影响的时段和影响程度
        
        return {
            'event_impact_analysis': '待实现',
            'affected_periods': [],
            'impact_level': 'unknown'
        }
    
    def generate_period_report(self, data_analysis: Dict, 
                              llm_analysis: Dict) -> str:
        """生成时段分析报告"""
        report = []
        report.append("# 时段特征分析报告\n")
        report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**合约**: {data_analysis.get('symbol', 'N/A')}\n")
        report.append(f"**数据周期**: {data_analysis.get('data_period_days', 0)}天\n\n")
        
        # 时段配置
        report.append("## 时段配置建议\n\n")
        report.append("| 时段 | 波动率 | 滑点率 | 平衡阈值 | 最大仓位 | 订单偏离 |\n")
        report.append("|------|--------|--------|----------|----------|----------|\n")
        
        period_configs = data_analysis.get('period_configs', {})
        for period, config in period_configs.items():
            report.append(f"| {period} | "
                         f"{config['volatility']*100:.1f}% | "
                         f"{config['slippage_rate']*100:.2f}% | "
                         f"{config['balance_threshold']:.4f} | "
                         f"{config['max_position']} | "
                         f"{config['order_offset']:.2f} |\n")
        
        # 异常检测
        if llm_analysis.get('anomalies'):
            report.append("\n## 异常时段检测\n\n")
            for anomaly in llm_analysis['anomalies']:
                report.append(f"### {anomaly['period']} - {anomaly['type']}\n")
                report.append(f"- **描述**: {anomaly['description']}\n")
                report.append(f"- **建议**: {anomaly['suggestion']}\n\n")
        
        # 风险提示
        if llm_analysis.get('risk_warnings'):
            report.append("## 风险提示\n\n")
            for warning in llm_analysis['risk_warnings']:
                report.append(f"- {warning}\n")
        
        return "".join(report)


def main():
    """主函数"""
    # 数据驱动分析
    analyzer = TimePeriodAnalyzer(symbol="SIL2603")
    data_result = analyzer.analyze_from_klines(days=30)
    
    if not data_result:
        print("❌ 数据驱动分析失败")
        return
    
    # 大模型辅助分析
    llm_analyzer = LLMPeriodAnalyzer(analyzer)
    llm_result = llm_analyzer.analyze_with_llm(data_result, use_api=False)
    
    # 生成报告
    report = llm_analyzer.generate_period_report(data_result, llm_result)
    
    # 保存报告
    report_file = f"/home/cx/trading_data/period_analysis_report_{datetime.now().strftime('%Y%m%d')}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 分析报告已保存到: {report_file}")
    print("\n" + report)


if __name__ == "__main__":
    main()
