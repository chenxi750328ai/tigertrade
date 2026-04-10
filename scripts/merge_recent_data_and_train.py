#!/usr/bin/env python3
"""合并近期白银数据（含暴跌行情）并运行预处理+训练准备"""
import sys
import os
import glob
import pandas as pd
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def main():
    base = Path('/home/cx/trading_data')
    large_dir = base / 'large_dataset'
    out_merged = large_dir / f'full_merged_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

    # 1. 收集近期每日数据 (1月26日至今)
    daily_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith('2026-')])
    recent_dirs = [d for d in daily_dirs if d.name >= '2026-01-26']

    dfs = []
    for d in recent_dirs:
        for f in d.glob('trading_data_*.csv'):
            try:
                df = pd.read_csv(f)
                if 'timestamp' in df.columns and 'price_current' in df.columns and len(df) > 0:
                    dfs.append(df)
                    print(f"  ✅ {f.relative_to(base)}: {len(df)} 条")
            except Exception as e:
                print(f"  ⚠️ {f.name}: {e}")

    if not dfs:
        print("❌ 未找到近期每日数据，尝试使用 large_dataset 最新 full_*.csv")
        full_files = sorted(large_dir.glob('full_*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
        if full_files:
            latest = full_files[0]
            print(f"  使用: {latest.name}")
            input_file = str(latest)
        else:
            print("❌ 无可用数据")
            return 1
    else:
        merged = pd.concat(dfs, ignore_index=True)
        if 'timestamp' in merged.columns:
            merged['timestamp'] = pd.to_datetime(merged['timestamp'], errors='coerce')
            merged = merged.dropna(subset=['timestamp'])
            merged = merged.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
        merged.to_csv(out_merged, index=False)
        print(f"\n✅ 合并完成: {out_merged} ({len(merged)} 条)")
        input_file = str(out_merged)

    # 2. 与历史 full_ 合并（去重）
    hist_full = large_dir / 'full_20260121_100827.csv'
    if hist_full.exists() and input_file != str(hist_full):
        hist = pd.read_csv(hist_full)
        new_df = pd.read_csv(input_file)
        # 统一列：取交集 + 补全
        common = [c for c in hist.columns if c in new_df.columns]
        for c in hist.columns:
            if c not in new_df.columns and c != 'Unnamed: 0':
                new_df[c] = None
        hist_sel = hist[[c for c in common if c in hist.columns]]
        new_sel = new_df[[c for c in common if c in new_df.columns]]
        combined = pd.concat([hist_sel, new_sel], ignore_index=True)
        if 'timestamp' in combined.columns:
            combined['timestamp'] = pd.to_datetime(combined['timestamp'], errors='coerce')
            combined = combined.dropna(subset=['timestamp'])
            combined = combined.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
        final_path = large_dir / f'full_with_recent_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        combined.to_csv(final_path, index=False)
        print(f"✅ 与历史合并: {final_path} ({len(combined)} 条)")
        input_file = str(final_path)

    # 3. 运行数据预处理
    print("\n" + "="*60)
    print("🔄 运行数据预处理...")
    print("="*60)
    from scripts.data_preprocessing import TigerTradeDataProcessor, _ensure_sample_data
    # 使用合并后的文件
    output_dir = str(_REPO_ROOT / 'data' / 'processed')
    processor = TigerTradeDataProcessor(input_file, output_dir)
    (processor
     .load_data()
     .clean_data()
     .add_technical_indicators()
     .add_custom_features()
     .create_target()
     .split_data(train_ratio=0.7, val_ratio=0.15)
     .save_data()
     .generate_report())

    print("\n" + "="*60)
    print("🎉 数据已更新，可用于训练")
    print("="*60)
    return 0

if __name__ == '__main__':
    sys.exit(main())
