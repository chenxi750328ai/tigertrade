#!/usr/bin/env python3
"""
标注优化工具 - 针对小价格变化的优化标注策略
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

def analyze_price_changes(df):
    """分析价格变化分布"""
    print("=" * 80)
    print("📊 价格变化分析")
    print("=" * 80)
    
    # 计算各周期的价格变化
    for look_ahead in [3, 5, 10, 20]:
        changes = []
        for i in range(len(df) - look_ahead):
            current = df.iloc[i]['price_current']
            future = df.iloc[i + look_ahead]['price_current']
            pct_change = (future - current) / current * 100
            changes.append(pct_change)
        
        changes = np.array(changes)
        
        print(f"\n向前看 {look_ahead} 个周期:")
        print(f"  平均变化: {changes.mean():.6f}%")
        print(f"  标准差: {changes.std():.6f}%")
        print(f"  最小值: {changes.min():.6f}%")
        print(f"  最大值: {changes.max():.6f}%")
        print(f"  25分位: {np.percentile(changes, 25):.6f}%")
        print(f"  50分位: {np.percentile(changes, 50):.6f}%")
        print(f"  75分位: {np.percentile(changes, 75):.6f}%")
        
        # 建议阈值
        std = changes.std()
        suggested_buy = std * 0.5
        suggested_sell = -std * 0.5
        print(f"  建议买入阈值: {suggested_buy:.6f}% (0.5倍标准差)")
        print(f"  建议卖出阈值: {suggested_sell:.6f}% (0.5倍标准差)")


def generate_optimized_labels(df, look_ahead=5):
    """
    生成优化的标签 - 使用多种优化策略
    """
    print("\n" + "=" * 80)
    print("🏷️ 生成优化标签")
    print("=" * 80)
    
    df = df.copy()
    
    # 计算价格变化
    price_changes = []
    for i in range(len(df)):
        if i + look_ahead < len(df):
            current = df.iloc[i]['price_current']
            future = df.iloc[i + look_ahead]['price_current']
            pct_change = (future - current) / current * 100
        else:
            pct_change = 0
        price_changes.append(pct_change)
    
    df['future_price_change'] = price_changes
    
    # 策略1: 基于百分位数
    df = _label_percentile(df, look_ahead)
    
    # 策略2: 基于标准差
    df = _label_std(df, look_ahead)
    
    # 策略3: 相对强度
    df = _label_relative_strength(df, look_ahead)
    
    # 策略4: 方向性(任何正/负变化)
    df = _label_directional(df, look_ahead)
    
    # 策略5: 混合策略
    df = _label_hybrid(df)
    
    # 打印对比
    _print_label_distribution(df)
    
    return df


def _label_percentile(df, look_ahead):
    """基于百分位数的标注 - 最灵活"""
    print("\n策略1: 百分位数标注")
    
    changes = df['future_price_change'].values[:-look_ahead]
    
    # 使用33%和67%分位点
    buy_threshold = np.percentile(changes, 67)
    sell_threshold = np.percentile(changes, 33)
    
    print(f"  买入阈值: {buy_threshold:.6f}% (67分位)")
    print(f"  卖出阈值: {sell_threshold:.6f}% (33分位)")
    
    df['label_percentile'] = 0
    for i in range(len(df) - look_ahead):
        change = df.iloc[i]['future_price_change']
        if change > buy_threshold:
            df.iloc[i, df.columns.get_loc('label_percentile')] = 1
        elif change < sell_threshold:
            df.iloc[i, df.columns.get_loc('label_percentile')] = 2
    
    return df


def _label_std(df, look_ahead):
    """基于标准差的标注"""
    print("策略2: 标准差标注")
    
    changes = df['future_price_change'].values[:-look_ahead]
    mean = changes.mean()
    std = changes.std()
    
    # 使用0.25倍标准差作为阈值
    buy_threshold = mean + std * 0.25
    sell_threshold = mean - std * 0.25
    
    print(f"  均值: {mean:.6f}%")
    print(f"  标准差: {std:.6f}%")
    print(f"  买入阈值: {buy_threshold:.6f}% (均值+0.25*std)")
    print(f"  卖出阈值: {sell_threshold:.6f}% (均值-0.25*std)")
    
    df['label_std'] = 0
    for i in range(len(df) - look_ahead):
        change = df.iloc[i]['future_price_change']
        if change > buy_threshold:
            df.iloc[i, df.columns.get_loc('label_std')] = 1
        elif change < sell_threshold:
            df.iloc[i, df.columns.get_loc('label_std')] = 2
    
    return df


def _label_relative_strength(df, look_ahead):
    """基于相对强度的标注"""
    print("策略3: 相对强度标注")
    
    df['label_rel'] = 0
    
    # 使用滚动窗口比较
    window = 20
    
    for i in range(window, len(df) - look_ahead):
        current_change = df.iloc[i]['future_price_change']
        recent_changes = df.iloc[i-window:i]['future_price_change'].values
        
        # 比较当前变化与最近窗口的变化
        if current_change > np.percentile(recent_changes, 75):
            df.iloc[i, df.columns.get_loc('label_rel')] = 1
        elif current_change < np.percentile(recent_changes, 25):
            df.iloc[i, df.columns.get_loc('label_rel')] = 2
    
    print(f"  使用{window}周期滚动窗口比较")
    
    return df


def _label_directional(df, look_ahead):
    """纯方向性标注 - 任何上涨=买入，任何下跌=卖出"""
    print("策略4: 纯方向性标注")
    
    df['label_dir'] = 0
    
    threshold = 0.001  # 极小阈值，只要有方向就标记
    
    for i in range(len(df) - look_ahead):
        change = df.iloc[i]['future_price_change']
        if change > threshold:
            df.iloc[i, df.columns.get_loc('label_dir')] = 1
        elif change < -threshold:
            df.iloc[i, df.columns.get_loc('label_dir')] = 2
    
    print(f"  阈值: ±{threshold:.6f}%")
    
    return df


def _label_hybrid(df):
    """混合策略 - 综合多个策略"""
    print("策略5: 混合策略 (投票)")
    
    df['label_hybrid'] = 0
    
    label_cols = ['label_percentile', 'label_std', 'label_rel', 'label_dir']
    
    for i in range(len(df)):
        votes = []
        for col in label_cols:
            if col in df.columns:
                votes.append(df.iloc[i][col])
        
        if len(votes) > 0:
            buy_votes = sum(1 for v in votes if v == 1)
            sell_votes = sum(1 for v in votes if v == 2)
            
            # 至少2票
            if buy_votes >= 2:
                df.iloc[i, df.columns.get_loc('label_hybrid')] = 1
            elif sell_votes >= 2:
                df.iloc[i, df.columns.get_loc('label_hybrid')] = 2
    
    return df


def _print_label_distribution(df):
    """打印标签分布"""
    print("\n" + "=" * 80)
    print("📊 各策略标签分布")
    print("=" * 80)
    
    strategies = {
        'label_percentile': '百分位数',
        'label_std': '标准差',
        'label_rel': '相对强度',
        'label_dir': '纯方向性',
        'label_hybrid': '混合策略'
    }
    
    print(f"\n{'策略':<15} {'持有(0)':<20} {'买入(1)':<20} {'卖出(2)':<20}")
    print("-" * 80)
    
    for col, name in strategies.items():
        if col in df.columns:
            counts = df[col].value_counts()
            total = len(df)
            hold = counts.get(0, 0)
            buy = counts.get(1, 0)
            sell = counts.get(2, 0)
            
            print(f"{name:<15} {hold:>6} ({hold/total*100:>5.1f}%)  "
                  f"{buy:>6} ({buy/total*100:>5.1f}%)  "
                  f"{sell:>6} ({sell/total*100:>5.1f}%)")


def save_optimized_data(df, original_file):
    """保存优化后的数据"""
    print("\n" + "=" * 80)
    print("💾 保存优化数据")
    print("=" * 80)
    
    # 生成新文件名
    base_name = os.path.basename(original_file).replace('.csv', '_optimized.csv')
    output_file = os.path.join(os.path.dirname(original_file), base_name)
    
    df.to_csv(output_file, index=True, encoding='utf-8')
    print(f"✅ 优化数据已保存: {output_file}")
    
    return output_file


def generate_comparison_report(df, output_dir):
    """生成标注策略对比报告"""
    report_file = os.path.join(output_dir, f'label_optimization_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 标注策略优化报告\n\n")
        f.write(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## 📊 价格变化统计\n\n")
        changes = df['future_price_change'].values
        f.write(f"- **平均变化:** {changes.mean():.6f}%\n")
        f.write(f"- **标准差:** {changes.std():.6f}%\n")
        f.write(f"- **最小值:** {changes.min():.6f}%\n")
        f.write(f"- **最大值:** {changes.max():.6f}%\n\n")
        
        f.write("## 🏷️ 标注策略对比\n\n")
        
        strategies = {
            'label_percentile': '百分位数',
            'label_std': '标准差',
            'label_rel': '相对强度',
            'label_dir': '纯方向性',
            'label_hybrid': '混合策略'
        }
        
        f.write("| 策略 | 持有(0) | 买入(1) | 卖出(2) | 平衡度 |\n")
        f.write("|------|---------|---------|---------|--------|\n")
        
        for col, name in strategies.items():
            if col in df.columns:
                counts = df[col].value_counts()
                hold = counts.get(0, 0)
                buy = counts.get(1, 0)
                sell = counts.get(2, 0)
                total = len(df)
                
                # 计算平衡度 (买入和卖出的比例差异)
                if buy + sell > 0:
                    balance = min(buy, sell) / max(buy, sell)
                else:
                    balance = 0
                
                f.write(f"| {name} | {hold} ({hold/total*100:.1f}%) | "
                       f"{buy} ({buy/total*100:.1f}%) | "
                       f"{sell} ({sell/total*100:.1f}%) | {balance:.3f} |\n")
        
        f.write("\n## 💡 推荐\n\n")
        f.write("### 各策略特点\n\n")
        f.write("1. **百分位数策略**: 自动适应数据分布，最灵活\n")
        f.write("2. **标准差策略**: 基于统计显著性，较保守\n")
        f.write("3. **相对强度策略**: 考虑短期趋势，动态调整\n")
        f.write("4. **纯方向性策略**: 最激进，所有有方向的变化都标记\n")
        f.write("5. **混合策略**: 综合多个策略，最稳健\n\n")
        
        f.write("### 使用建议\n\n")
        f.write("- **Demo环境训练**: 推荐使用 **百分位数** 或 **混合策略**\n")
        f.write("- **真实环境**: 可以使用更保守的 **标准差策略**\n")
        f.write("- **研究探索**: 可以尝试 **纯方向性策略**\n")
    
    print(f"✅ 对比报告已保存: {report_file}")
    return report_file


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python optimize_labels.py <数据文件路径>")
        print("示例: python optimize_labels.py /home/cx/trading_data/enhanced/full_*.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("🔧 标注优化工具")
    print("=" * 80)
    print(f"\n输入文件: {input_file}")
    
    # 读取数据
    print("\n读取数据...")
    df = pd.read_csv(input_file, index_col=0)
    print(f"✅ 数据加载成功: {len(df)} 条记录")
    
    # 分析价格变化
    analyze_price_changes(df)
    
    # 生成优化标签
    df_optimized = generate_optimized_labels(df)
    
    # 保存优化数据
    output_file = save_optimized_data(df_optimized, input_file)
    
    # 生成对比报告
    output_dir = os.path.dirname(input_file)
    report_file = generate_comparison_report(df_optimized, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ 标注优化完成！")
    print("=" * 80)
    print(f"\n优化数据: {output_file}")
    print(f"对比报告: {report_file}")


if __name__ == "__main__":
    main()
