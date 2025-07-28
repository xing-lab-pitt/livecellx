#!/usr/bin/env python3
"""
Comprehensive Active Learning Benchmark Runner

This module provides functionality to run comprehensive benchmarks comparing
NAIVE baseline strategy with sophisticated Active Learning strategies.

Features:
- Multiple runs per strategy for statistical significance
- Comprehensive metric collection
- Automated results analysis and visualization
- Performance comparison with confidence intervals
- Publication-ready results
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class BenchmarkRunner:
    """Main class for running comprehensive AL benchmarks"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize benchmark runner with configuration"""
        self.config = config
        self.base_output_dir = config['base_output_dir']
        self.strategies = config['strategies']
        self.strategy_names = config['strategy_names']
        self.n_runs = config['n_runs']
        self.data_dir = config['data_dir']
        
        # Create base output directory
        Path(self.base_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Track experiment statistics
        self.total_experiments = len(self.strategies) * self.n_runs
        self.completed_experiments = 0
        self.failed_experiments = 0
        self.start_time = time.time()
        
        # Create master log
        self.create_master_log()
    
    def setup_logging(self):
        """Setup logging for the benchmark runner"""
        log_file = Path(self.base_output_dir) / "benchmark_runner.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_master_log(self):
        """Create master experiment log"""
        self.master_log_path = Path(self.base_output_dir) / "master_experiment_log.txt"
        with open(self.master_log_path, 'w') as f:
            f.write("Comprehensive AL Benchmark - Master Log\n")
            f.write(f"Start Time: {datetime.now().strftime('%c')}\n")
            f.write("Configuration:\n")
            f.write(f"- Strategies: {' '.join(self.strategies)}\n")
            f.write(f"- Runs per Strategy: {self.n_runs}\n")
            f.write(f"- AL Iterations: {self.config['al_iterations']}\n")
            f.write(f"- Total Experiments: {self.total_experiments}\n")
            f.write("\n")
    
    def update_master_log(self, message: str):
        """Update master log with experiment status"""
        with open(self.master_log_path, 'a') as f:
            f.write(f"{message}\n")
    
    def run_single_experiment(self, strategy: str, run_id: int, strategy_name: str) -> bool:
        """Run a single experiment"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{strategy_name}_run{run_id}_{timestamp}"
        exp_output = Path(self.base_output_dir) / exp_name
        
        self.logger.info("=" * 50)
        self.logger.info(f"🎯 Strategy: {strategy_name}")
        self.logger.info(f"🔄 Run: {run_id}/{self.n_runs}")
        self.logger.info(f"📁 Output: {exp_output}")
        self.logger.info(f"⏰ Start: {datetime.now().strftime('%c')}")
        self.logger.info("=" * 50)
        
        # Create experiment-specific output directory
        exp_output.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"📁 Created directory: {exp_output}")
        
        # Build command
        seed = 42 + run_id * 1000
        cmd = self.build_training_command(strategy, exp_output, seed)
        
        # Run training
        try:
            self.logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.config.get('experiment_timeout', 3600)  # 1 hour default
            )
            
            # Save output
            log_file = exp_output / "training.log"
            with open(log_file, 'w') as f:
                f.write(result.stdout)
            
            if result.returncode == 0:
                self.logger.info(f"✅ {strategy_name} Run {run_id} completed successfully")
                self.logger.info(f"⏰ End: {datetime.now().strftime('%c')}")
                
                # Extract and log results
                self.extract_and_log_results(exp_output, strategy_name, run_id, seed)
                return True
            else:
                self.logger.error(f"❌ {strategy_name} Run {run_id} failed with return code {result.returncode}")
                self.logger.error(f"📝 Check log: {log_file}")
                self.create_failure_summary(exp_output, strategy_name, run_id, seed, result.returncode)
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"❌ {strategy_name} Run {run_id} timed out")
            return False
        except Exception as e:
            self.logger.error(f"❌ {strategy_name} Run {run_id} failed with exception: {e}")
            return False
    
    def build_training_command(self, strategy: str, exp_output: Path, seed: int) -> List[str]:
        """Build the training command"""
        cmd = [
            "python", "train_ctc_segmentation.py",
            "--data_dir", str(self.data_dir),
            "--output_dir", str(exp_output),
            "--al_strategy", strategy,
            "--al_iterations", str(self.config['al_iterations']),
            "--initial_samples", str(self.config['initial_samples']),
            "--samples_per_iteration", str(self.config['samples_per_iteration']),
            "--max_epochs", str(self.config['max_epochs']),
            "--batch_size", str(self.config['batch_size']),
            "--num_workers", str(self.config['num_workers']),
            "--learning_rate", str(self.config['learning_rate']),
            "--loss_function", self.config['loss_function'],
            "--base_channels", str(self.config['base_channels']),
            "--image_size", str(self.config['image_size']),
            "--seed", str(seed)
        ]
        
        # Add optional parameters
        if self.config.get('debug', False):
            cmd.append("--debug")
        if self.config.get('subset_ratio'):
            cmd.extend(["--subset_ratio", str(self.config['subset_ratio'])])
        
        return cmd
    
    def extract_and_log_results(self, exp_output: Path, strategy_name: str, run_id: int, seed: int):
        """Extract key results and create summary"""
        results_file = exp_output / "experiment_results.json"
        final_iou = "N/A"
        
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                if data.get('test_performances'):
                    final_perf = data['test_performances'][-1]
                    final_iou = f"{final_perf.get('test_iou', 0):.4f}"
                    self.logger.info(f"📊 Final Test IoU: {final_iou}")
            except Exception as e:
                self.logger.warning(f"Could not extract results: {e}")
        
        # Create run summary
        summary_file = exp_output / "run_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Experiment Summary\n")
            f.write("==================\n")
            f.write(f"Strategy: {strategy_name}\n")
            f.write(f"Run ID: {run_id}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Start Time: {datetime.now().strftime('%c')}\n")
            f.write("Status: SUCCESS\n")
            f.write(f"Final IoU: {final_iou}\n")
            f.write(f"Output Directory: {exp_output}\n")
            f.write(f"Training Log: {exp_output}/training.log\n")
            f.write(f"Results File: {exp_output}/experiment_results.json\n")
    
    def create_failure_summary(self, exp_output: Path, strategy_name: str, run_id: int, seed: int, exit_code: int):
        """Create summary for failed experiment"""
        summary_file = exp_output / "run_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Experiment Summary\n")
            f.write("==================\n")
            f.write(f"Strategy: {strategy_name}\n")
            f.write(f"Run ID: {run_id}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Start Time: {datetime.now().strftime('%c')}\n")
            f.write(f"Status: FAILED (exit code: {exit_code})\n")
            f.write(f"Error Log: {exp_output}/training.log\n")
    
    def run_all_experiments(self, interactive: bool = True) -> bool:
        """Run all experiments for all strategies"""
        self.logger.info("🚀 Starting comprehensive benchmark experiments...")
        self.logger.info(f"📊 Total experiments: {self.total_experiments}")
        self.logger.info("")
        
        # Run experiments for each strategy
        for i, (strategy, strategy_name) in enumerate(zip(self.strategies, self.strategy_names)):
            self.logger.info("=" * 50)
            self.logger.info(f"📈 STRATEGY: {strategy_name} ({strategy})")
            self.logger.info("=" * 50)
            
            # Run multiple independent experiments for this strategy
            for run_id in range(1, self.n_runs + 1):
                self.logger.info(f"Experiment {self.completed_experiments + 1}/{self.total_experiments}")
                
                if self.run_single_experiment(strategy, run_id, strategy_name):
                    self.completed_experiments += 1
                    self.update_master_log(f"✅ Success: {strategy_name} Run {run_id}")
                else:
                    self.failed_experiments += 1
                    self.update_master_log(f"❌ Failed: {strategy_name} Run {run_id}")
                    
                    # Ask whether to continue or abort
                    if interactive:
                        self.logger.warning("⚠️  Experiment failed. Continue with remaining experiments?")
                        response = input("Continue? (y/n): ").strip().lower()
                        if response not in ['y', 'yes']:
                            self.logger.info("Aborting benchmark due to failure.")
                            return False
                
                # Brief pause between experiments
                if run_id < self.n_runs:
                    self.logger.info("⏳ Waiting 30 seconds before next run...")
                    time.sleep(30)
            
            self.logger.info(f"✅ Completed all runs for {strategy_name}")
            self.logger.info("")
        
        return True
    
    def analyze_results(self):
        """Analyze results across all experiments"""
        end_time = time.time()
        total_time = end_time - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        
        self.logger.info("=" * 50)
        self.logger.info("📊 COMPREHENSIVE RESULTS ANALYSIS")
        self.logger.info("=" * 50)
        self.logger.info(f"✅ Completed: {self.completed_experiments}/{self.total_experiments} experiments")
        self.logger.info(f"❌ Failed: {self.failed_experiments} experiments")
        self.logger.info(f"⏱️  Total Time: {hours}h {minutes}m")
        self.logger.info("")
        
        # Update master log
        self.update_master_log("")
        self.update_master_log("Completion Summary:")
        self.update_master_log(f"- End Time: {datetime.now().strftime('%c')}")
        self.update_master_log(f"- Total Time: {hours}h {minutes}m")
        self.update_master_log(f"- Completed: {self.completed_experiments}/{self.total_experiments}")
        self.update_master_log(f"- Failed: {self.failed_experiments}")
        
        if self.completed_experiments == 0:
            self.logger.error("❌ No experiments completed successfully. Cannot generate analysis.")
            return False
        
        self.logger.info("🔬 Analyzing results across all experiments...")
        
        # Generate comprehensive analysis
        try:
            self.generate_comprehensive_analysis()
            return True
        except Exception as e:
            self.logger.error(f"Failed to generate analysis: {e}")
            return False
    
    def generate_comprehensive_analysis(self):
        """Generate comprehensive statistical analysis and visualizations"""
        self.logger.info("🔍 Collecting results from all experiments...")
        
        # Collect all results
        all_results = []
        strategy_stats = {}
        
        base_dir = Path(self.base_output_dir)
        
        for strategy, strategy_name in zip(self.strategies, self.strategy_names):
            strategy_results = []
            
            # Find all experiment directories for this strategy
            exp_dirs = list(base_dir.glob(f"{strategy_name}_run*"))
            self.logger.info(f"📂 Found {len(exp_dirs)} experiments for {strategy_name}")
            
            for exp_dir in exp_dirs:
                results_file = exp_dir / "experiment_results.json"
                if results_file.exists():
                    try:
                        with open(results_file, 'r') as f:
                            data = json.load(f)
                        
                        # Extract final performance
                        if data.get('test_performances'):
                            final_perf = data['test_performances'][-1]
                            result = {
                                'strategy': strategy,
                                'strategy_name': strategy_name,
                                'run_id': exp_dir.name,
                                'final_iou': final_perf.get('test_iou', 0),
                                'final_dice': final_perf.get('test_dice', 0),
                                'final_acc': final_perf.get('test_acc', 0),
                                'final_samples': data['labeled_counts'][-1] if data.get('labeled_counts') else 0,
                                'iterations': len(data['test_performances'])
                            }
                            
                            strategy_results.append(result['final_iou'])
                            all_results.append(result)
                    except Exception as e:
                        self.logger.warning(f"⚠️  Error reading {results_file}: {e}")
            
            # Calculate statistics for this strategy
            if strategy_results:
                strategy_stats[strategy_name] = {
                    'mean_iou': np.mean(strategy_results),
                    'std_iou': np.std(strategy_results),
                    'min_iou': np.min(strategy_results),
                    'max_iou': np.max(strategy_results),
                    'n_runs': len(strategy_results)
                }
        
        # Create comprehensive results DataFrame
        results_df = pd.DataFrame(all_results)
        self.logger.info(f"📊 Collected {len(results_df)} successful experiments")
        
        # Create analysis directory
        analysis_dir = base_dir / "comprehensive_analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        # Save results and generate analysis
        self.save_analysis_results(results_df, strategy_stats, analysis_dir)
        self.generate_visualizations(results_df, strategy_stats, analysis_dir)
        self.generate_final_report(strategy_stats, analysis_dir)
        
        self.logger.info("✅ Comprehensive analysis completed successfully!")
        self.logger.info(f"📋 Analysis location: {analysis_dir}")
    
    def save_analysis_results(self, results_df: pd.DataFrame, strategy_stats: Dict, analysis_dir: Path):
        """Save analysis results to files"""
        # Save raw results
        results_df.to_csv(analysis_dir / "all_experiment_results.csv", index=False)
        
        # Generate summary statistics
        summary_stats = []
        for strategy_name, stats in strategy_stats.items():
            summary_stats.append({
                'Strategy': strategy_name,
                'Mean_IoU': f"{stats['mean_iou']:.4f}",
                'Std_IoU': f"{stats['std_iou']:.4f}",
                'Min_IoU': f"{stats['min_iou']:.4f}",
                'Max_IoU': f"{stats['max_iou']:.4f}",
                'N_Runs': stats['n_runs'],
                'CI_95_Lower': f"{stats['mean_iou'] - 1.96*stats['std_iou']/np.sqrt(stats['n_runs']):.4f}",
                'CI_95_Upper': f"{stats['mean_iou'] + 1.96*stats['std_iou']/np.sqrt(stats['n_runs']):.4f}"
            })
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(analysis_dir / "strategy_summary_statistics.csv", index=False)
        
        self.logger.info("📊 Summary Statistics:")
        self.logger.info(summary_df.to_string(index=False))
    
    def generate_visualizations(self, results_df: pd.DataFrame, strategy_stats: Dict, analysis_dir: Path):
        """Generate comprehensive visualizations"""
        self.logger.info("📊 Creating comprehensive visualizations...")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create comprehensive figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comprehensive Active Learning Benchmark Results', fontsize=16, fontweight='bold')
        
        # 1. Box plot comparison
        if len(results_df) > 0:
            sns.boxplot(data=results_df, x='strategy_name', y='final_iou', ax=axes[0, 0])
            axes[0, 0].set_title('Final IoU Distribution by Strategy')
            axes[0, 0].set_xlabel('Active Learning Strategy')
            axes[0, 0].set_ylabel('Final Test IoU')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Bar plot with error bars
        strategies_plot = list(strategy_stats.keys())
        means = [strategy_stats[s]['mean_iou'] for s in strategies_plot]
        stds = [strategy_stats[s]['std_iou'] for s in strategies_plot]
        
        bars = axes[0, 1].bar(strategies_plot, means, yerr=stds, capsize=5, alpha=0.7)
        axes[0, 1].set_title('Mean Performance with Standard Deviation')
        axes[0, 1].set_xlabel('Active Learning Strategy')
        axes[0, 1].set_ylabel('Mean Test IoU')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                           f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Performance improvement analysis
        if 'Naive_Baseline' in strategy_stats and len(strategy_stats) > 1:
            naive_mean = strategy_stats['Naive_Baseline']['mean_iou']
            improvements = {}
            
            for strategy_name, stats in strategy_stats.items():
                if strategy_name != 'Naive_Baseline':
                    improvement = ((stats['mean_iou'] - naive_mean) / naive_mean) * 100
                    improvements[strategy_name] = improvement
            
            if improvements:
                strategy_names_imp = list(improvements.keys())
                improvement_values = list(improvements.values())
                
                colors = ['green' if x > 0 else 'red' for x in improvement_values]
                bars = axes[1, 0].bar(strategy_names_imp, improvement_values, color=colors, alpha=0.7)
                
                axes[1, 0].set_title('Performance Improvement over Naive Baseline')
                axes[1, 0].set_xlabel('Active Learning Strategy')
                axes[1, 0].set_ylabel('Improvement (%)')
                axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                axes[1, 0].tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, improvement_values):
                    height = bar.get_height()
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2.,
                                   height + (1 if height > 0 else -1),
                                   f'{value:.1f}%', ha='center',
                                   va='bottom' if height > 0 else 'top', fontweight='bold')
        
        # 4. Sample efficiency comparison
        if len(results_df) > 0:
            sns.scatterplot(data=results_df, x='final_samples', y='final_iou',
                           hue='strategy_name', s=100, alpha=0.7, ax=axes[1, 1])
            axes[1, 1].set_title('Sample Efficiency: Performance vs Labeled Samples')
            axes[1, 1].set_xlabel('Final Number of Labeled Samples')
            axes[1, 1].set_ylabel('Final Test IoU')
        
        plt.tight_layout()
        plt.savefig(analysis_dir / 'comprehensive_benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_final_report(self, strategy_stats: Dict, analysis_dir: Path):
        """Generate final comprehensive report"""
        self.logger.info("📝 Generating comprehensive report...")
        
        report_path = analysis_dir / "comprehensive_benchmark_report.md"
        with open(report_path, 'w') as f:
            f.write("# Comprehensive Active Learning Benchmark Report\n\n")
            
            f.write("## Experiment Overview\n\n")
            f.write(f"- **Date**: {datetime.now().strftime('%c')}\n")
            f.write(f"- **Total Experiments**: {self.completed_experiments}\n")
            f.write(f"- **Strategies Tested**: {len(strategy_stats)}\n")
            f.write(f"- **Runs per Strategy**: {self.n_runs} (for statistical significance)\n")
            f.write(f"- **Dataset**: CTC Single-Cell Segmentation (~32k samples)\n\n")
            
            f.write("## Key Findings\n\n")
            f.write("### Performance Summary\n\n")
            
            f.write("| Strategy | Mean IoU | Std Dev | 95% CI | N Runs |\n")
            f.write("|----------|----------|---------|--------|--------|\n")
            
            for strategy_name, stats in strategy_stats.items():
                ci_lower = stats['mean_iou'] - 1.96*stats['std_iou']/np.sqrt(stats['n_runs'])
                ci_upper = stats['mean_iou'] + 1.96*stats['std_iou']/np.sqrt(stats['n_runs'])
                f.write(f"| {strategy_name} | {stats['mean_iou']:.4f} | {stats['std_iou']:.4f} | [{ci_lower:.4f}, {ci_upper:.4f}] | {stats['n_runs']} |\n")
            
            if 'Naive_Baseline' in strategy_stats:
                f.write(f"\n### Performance vs Naive Baseline\n\n")
                naive_mean = strategy_stats['Naive_Baseline']['mean_iou']
                
                for strategy_name, stats in strategy_stats.items():
                    if strategy_name != 'Naive_Baseline':
                        improvement = ((stats['mean_iou'] - naive_mean) / naive_mean) * 100
                        f.write(f"- **{strategy_name}**: {improvement:+.1f}% improvement\n")
            
            f.write(f"\n## Experimental Details\n\n")
            f.write(f"- **AL Iterations**: {self.config['al_iterations']}\n")
            f.write(f"- **Initial Samples**: {self.config['initial_samples']}\n")
            f.write(f"- **Samples per Iteration**: {self.config['samples_per_iteration']}\n")
            f.write(f"- **Final Sample Count**: {self.config['initial_samples'] + self.config['al_iterations'] * self.config['samples_per_iteration']}\n")
            f.write(f"- **Training Epochs**: {self.config['max_epochs']}\n")
            f.write(f"- **Model**: U-Net ({self.config['base_channels']} base channels, {self.config['image_size']}×{self.config['image_size']} input)\n\n")
            
            f.write("## Statistical Significance\n\n")
            f.write(f"Multiple independent runs (N={self.n_runs}) were conducted for each strategy to ensure statistical rigor. 95% confidence intervals are provided for all metrics.\n\n")
            
            f.write("## Files Generated\n\n")
            f.write("- `all_experiment_results.csv`: Raw results from all experiments\n")
            f.write("- `strategy_summary_statistics.csv`: Statistical summary by strategy\n")
            f.write("- `comprehensive_benchmark_results.png`: Visualization of all results\n")
            f.write("- Individual experiment logs in respective directories\n\n")
            
            f.write("## Conclusion\n\n")
            
            if len(strategy_stats) > 1 and 'Naive_Baseline' in strategy_stats:
                best_strategy = max([(name, stats['mean_iou']) for name, stats in strategy_stats.items() 
                                   if name != 'Naive_Baseline'], key=lambda x: x[1])
                f.write(f"The best performing strategy was **{best_strategy[0]}** with a mean IoU of {best_strategy[1]:.4f}. ")
                
                naive_mean = strategy_stats['Naive_Baseline']['mean_iou']
                if best_strategy[1] > naive_mean:
                    improvement = ((best_strategy[1] - naive_mean) / naive_mean) * 100
                    f.write(f"This represents a {improvement:.1f}% improvement over the naive baseline strategy.")
                else:
                    f.write("Interestingly, sophisticated AL strategies did not significantly outperform the naive baseline, suggesting the importance of temporal information in this domain.")
        
        self.logger.info(f"📋 Report location: {report_path}")


def create_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Create configuration dictionary from command line arguments"""
    config = {
        'base_output_dir': args.output_dir,
        'data_dir': args.data_dir,
        'strategies': args.strategies,
        'strategy_names': args.strategy_names,
        'n_runs': args.n_runs,
        'al_iterations': args.al_iterations,
        'initial_samples': args.initial_samples,
        'samples_per_iteration': args.samples_per_iteration,
        'max_epochs': args.max_epochs,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'learning_rate': args.learning_rate,
        'loss_function': args.loss_function,
        'base_channels': args.base_channels,
        'image_size': args.image_size,
        'debug': args.debug,
        'subset_ratio': args.subset_ratio,
        'experiment_timeout': args.timeout
    }
    return config


def main():
    """Main entry point for benchmark runner"""
    parser = argparse.ArgumentParser(description='Comprehensive Active Learning Benchmark Runner')
    
    # Required arguments
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Base output directory for benchmark results')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing the CTC dataset')
    
    # Strategy configuration
    parser.add_argument('--strategies', nargs='+', 
                       default=['time_interval', 'random', 'uncertainty'],
                       help='List of AL strategies to compare')
    parser.add_argument('--strategy_names', nargs='+',
                       default=['Naive_Baseline', 'Random_Sampling', 'Uncertainty_Sampling'],
                       help='Human-readable names for strategies')
    parser.add_argument('--n_runs', type=int, default=5,
                       help='Number of independent runs per strategy')
    
    # Training configuration
    parser.add_argument('--al_iterations', type=int, default=8,
                       help='Number of active learning iterations')
    parser.add_argument('--initial_samples', type=int, default=500,
                       help='Initial number of labeled samples')
    parser.add_argument('--samples_per_iteration', type=int, default=250,
                       help='Number of samples to add per AL iteration')
    parser.add_argument('--max_epochs', type=int, default=30,
                       help='Maximum training epochs per iteration')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for training')
    parser.add_argument('--loss_function', type=str, default='combined',
                       help='Loss function to use')
    parser.add_argument('--base_channels', type=int, default=64,
                       help='Base number of channels in U-Net')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size')
    
    # Optional configuration
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--subset_ratio', type=float, default=None,
                       help='Use subset of data for testing')
    parser.add_argument('--timeout', type=int, default=7200,
                       help='Timeout per experiment in seconds')
    parser.add_argument('--non_interactive', action='store_true',
                       help='Run in non-interactive mode (continue on failures)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if len(args.strategies) != len(args.strategy_names):
        parser.error("Number of strategies must match number of strategy names")
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Run benchmark
    runner = BenchmarkRunner(config)
    
    print("=" * 60)
    print("🚀 COMPREHENSIVE AL BENCHMARK EXPERIMENT")
    print("=" * 60)
    print(f"Base Output: {config['base_output_dir']}")
    print(f"Strategies: {' '.join(config['strategies'])}")
    print(f"Runs per Strategy: {config['n_runs']}")
    print(f"AL Iterations: {config['al_iterations']}")
    print(f"Total Experiments: {len(config['strategies']) * config['n_runs']}")
    print("")
    
    # Run all experiments
    success = runner.run_all_experiments(interactive=not args.non_interactive)
    
    if success:
        # Analyze results
        runner.analyze_results()
        
        print("=" * 60)
        print("🎉 COMPREHENSIVE BENCHMARK COMPLETED!")
        print("=" * 60)
        print(f"📁 Results: {config['base_output_dir']}")
        print(f"📊 Analysis: {config['base_output_dir']}/comprehensive_analysis/")
        print("=" * 60)
    else:
        print("❌ Benchmark failed or was aborted")
        sys.exit(1)


if __name__ == "__main__":
    main()