#!/usr/bin/env python3
"""
CSV Metrics Collection Script

This script aggregates CSV metrics from all experiments in a comprehensive benchmark
to create unified analysis files for easy visualization and comparison.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import json
from datetime import datetime


def collect_epoch_metrics(benchmark_dir: Path) -> pd.DataFrame:
    """Collect all epoch-level metrics from experiments"""
    all_epoch_metrics = []
    
    for exp_dir in benchmark_dir.glob("*_run*"):
        for exp_subdir in exp_dir.glob("ctc_seg_*"):
            epoch_files = list(exp_subdir.glob("*_epoch_metrics.csv"))
            
            for epoch_file in epoch_files:
                try:
                    df = pd.read_csv(epoch_file)
                    
                    # Add experiment metadata
                    strategy_name = exp_dir.name.split('_run')[0]
                    run_id = exp_dir.name.split('_run')[1].split('_')[0]
                    df['experiment_name'] = exp_dir.name
                    df['strategy_name'] = strategy_name
                    df['run_id'] = int(run_id)
                    
                    all_epoch_metrics.append(df)
                except Exception as e:
                    print(f"Warning: Could not read {epoch_file}: {e}")
    
    if all_epoch_metrics:
        return pd.concat(all_epoch_metrics, ignore_index=True)
    else:
        return pd.DataFrame()


def collect_iteration_metrics(benchmark_dir: Path) -> pd.DataFrame:
    """Collect all iteration-level metrics from experiments"""
    all_iteration_metrics = []
    
    for exp_dir in benchmark_dir.glob("*_run*"):
        for exp_subdir in exp_dir.glob("ctc_seg_*"):
            iteration_files = list(exp_subdir.glob("*_iteration_metrics.csv"))
            
            for iteration_file in iteration_files:
                try:
                    df = pd.read_csv(iteration_file)
                    
                    # Add experiment metadata
                    strategy_name = exp_dir.name.split('_run')[0]
                    run_id = exp_dir.name.split('_run')[1].split('_')[0]
                    df['experiment_name'] = exp_dir.name
                    df['strategy_name'] = strategy_name
                    df['run_id'] = int(run_id)
                    
                    all_iteration_metrics.append(df)
                except Exception as e:
                    print(f"Warning: Could not read {iteration_file}: {e}")
    
    if all_iteration_metrics:
        return pd.concat(all_iteration_metrics, ignore_index=True)
    else:
        return pd.DataFrame()


def collect_sample_selection_logs(benchmark_dir: Path) -> pd.DataFrame:
    """Collect all sample selection logs from experiments"""
    all_selection_logs = []
    
    for exp_dir in benchmark_dir.glob("*_run*"):
        for exp_subdir in exp_dir.glob("ctc_seg_*"):
            selection_files = list(exp_subdir.glob("*_sample_selection.csv"))
            
            for selection_file in selection_files:
                try:
                    df = pd.read_csv(selection_file)
                    
                    # Add experiment metadata
                    strategy_name = exp_dir.name.split('_run')[0]
                    run_id = exp_dir.name.split('_run')[1].split('_')[0]
                    df['experiment_name'] = exp_dir.name
                    df['strategy_name'] = strategy_name
                    df['run_id'] = int(run_id)
                    
                    all_selection_logs.append(df)
                except Exception as e:
                    print(f"Warning: Could not read {selection_file}: {e}")
    
    if all_selection_logs:
        return pd.concat(all_selection_logs, ignore_index=True)
    else:
        return pd.DataFrame()


def generate_summary_statistics(iteration_df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics across strategies and runs"""
    if iteration_df.empty:
        return pd.DataFrame()
    
    # Group by strategy and compute statistics
    summary_stats = []
    
    for strategy in iteration_df['strategy_name'].unique():
        strategy_data = iteration_df[iteration_df['strategy_name'] == strategy]
        
        # Get final performance for each run (last iteration)
        final_performances = []
        for run_id in strategy_data['run_id'].unique():
            run_data = strategy_data[strategy_data['run_id'] == run_id]
            if not run_data.empty:
                final_performance = run_data.iloc[-1]  # Last iteration
                final_performances.append({
                    'run_id': run_id,
                    'final_test_iou': final_performance['test_iou'],
                    'final_labeled_samples': final_performance['labeled_samples'],
                    'total_training_time': run_data['training_duration_seconds'].sum()
                })
        
        if final_performances:
            final_df = pd.DataFrame(final_performances)
            
            summary_stats.append({
                'strategy_name': strategy,
                'n_runs': len(final_performances),
                'mean_final_iou': final_df['final_test_iou'].mean(),
                'std_final_iou': final_df['final_test_iou'].std(),
                'min_final_iou': final_df['final_test_iou'].min(),
                'max_final_iou': final_df['final_test_iou'].max(),
                'mean_final_samples': final_df['final_labeled_samples'].mean(),
                'mean_training_time_sec': final_df['total_training_time'].mean(),
                'mean_training_time_min': final_df['total_training_time'].mean() / 60,
                'ci_95_lower': final_df['final_test_iou'].mean() - 1.96 * final_df['final_test_iou'].std() / np.sqrt(len(final_performances)),
                'ci_95_upper': final_df['final_test_iou'].mean() + 1.96 * final_df['final_test_iou'].std() / np.sqrt(len(final_performances))
            })
    
    return pd.DataFrame(summary_stats)


def generate_learning_curves(iteration_df: pd.DataFrame) -> pd.DataFrame:
    """Generate learning curves data for visualization"""
    if iteration_df.empty:
        return pd.DataFrame()
    
    # Create learning curves by averaging across runs
    learning_curves = []
    
    for strategy in iteration_df['strategy_name'].unique():
        strategy_data = iteration_df[iteration_df['strategy_name'] == strategy]
        
        for iteration in sorted(strategy_data['iteration'].unique()):
            iteration_data = strategy_data[strategy_data['iteration'] == iteration]
            
            if not iteration_data.empty:
                learning_curves.append({
                    'strategy_name': strategy,
                    'iteration': iteration,
                    'mean_test_iou': iteration_data['test_iou'].mean(),
                    'std_test_iou': iteration_data['test_iou'].std(),
                    'mean_labeled_samples': iteration_data['labeled_samples'].mean(),
                    'n_runs': len(iteration_data)
                })
    
    return pd.DataFrame(learning_curves)


def main():
    parser = argparse.ArgumentParser(description='Collect CSV metrics from comprehensive benchmark')
    parser.add_argument('benchmark_dir', type=str, 
                       help='Directory containing benchmark results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: benchmark_dir/csv_metrics_analysis)')
    
    args = parser.parse_args()
    
    benchmark_dir = Path(args.benchmark_dir)
    if not benchmark_dir.exists():
        print(f"Error: Benchmark directory {benchmark_dir} does not exist")
        sys.exit(1)
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = benchmark_dir / "csv_metrics_analysis"
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"🔍 Collecting CSV metrics from: {benchmark_dir}")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Collect all metrics
    print("📊 Collecting epoch-level metrics...")
    epoch_df = collect_epoch_metrics(benchmark_dir)
    print(f"   Found {len(epoch_df)} epoch records")
    
    print("📊 Collecting iteration-level metrics...")
    iteration_df = collect_iteration_metrics(benchmark_dir)
    print(f"   Found {len(iteration_df)} iteration records")
    
    print("📊 Collecting sample selection logs...")
    selection_df = collect_sample_selection_logs(benchmark_dir)
    print(f"   Found {len(selection_df)} selection records")
    
    # Save collected data
    if not epoch_df.empty:
        epoch_file = output_dir / "all_epoch_metrics.csv"
        epoch_df.to_csv(epoch_file, index=False)
        print(f"✅ Saved: {epoch_file}")
    
    if not iteration_df.empty:
        iteration_file = output_dir / "all_iteration_metrics.csv"
        iteration_df.to_csv(iteration_file, index=False)
        print(f"✅ Saved: {iteration_file}")
    
    if not selection_df.empty:
        selection_file = output_dir / "all_sample_selection.csv"
        selection_df.to_csv(selection_file, index=False)
        print(f"✅ Saved: {selection_file}")
    
    # Generate summary statistics
    if not iteration_df.empty:
        print("📈 Generating summary statistics...")
        summary_df = generate_summary_statistics(iteration_df)
        if not summary_df.empty:
            summary_file = output_dir / "csv_summary_statistics.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"✅ Saved: {summary_file}")
            
            # Display summary
            print("\n📊 Summary Statistics:")
            print("=" * 80)
            for _, row in summary_df.iterrows():
                print(f"{row['strategy_name']:20s}: {row['mean_final_iou']:.4f} ± {row['std_final_iou']:.4f} (n={int(row['n_runs'])})")
            print("=" * 80)
    
    # Generate learning curves
    if not iteration_df.empty:
        print("📈 Generating learning curves data...")
        curves_df = generate_learning_curves(iteration_df)
        if not curves_df.empty:
            curves_file = output_dir / "learning_curves.csv"
            curves_df.to_csv(curves_file, index=False)
            print(f"✅ Saved: {curves_file}")
    
    # Create metadata
    metadata = {
        'collection_timestamp': datetime.now().isoformat(),
        'source_directory': str(benchmark_dir),
        'output_directory': str(output_dir),
        'collected_files': {
            'epoch_records': len(epoch_df) if not epoch_df.empty else 0,
            'iteration_records': len(iteration_df) if not iteration_df.empty else 0,
            'selection_records': len(selection_df) if not selection_df.empty else 0
        }
    }
    
    if not iteration_df.empty:
        metadata['strategies'] = list(iteration_df['strategy_name'].unique())
        metadata['experiments'] = list(iteration_df['experiment_name'].unique())
    
    metadata_file = output_dir / "collection_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved: {metadata_file}")
    
    print()
    print("🎉 CSV metrics collection completed!")
    print(f"📁 All results saved to: {output_dir}")
    print()
    print("📋 Generated files:")
    print("   - all_epoch_metrics.csv: Training metrics per epoch")
    print("   - all_iteration_metrics.csv: Performance metrics per AL iteration")
    print("   - all_sample_selection.csv: Sample selection details")
    print("   - csv_summary_statistics.csv: Statistical summary by strategy")
    print("   - learning_curves.csv: Learning curves data for plotting")
    print("   - collection_metadata.json: Collection metadata")


if __name__ == "__main__":
    main()