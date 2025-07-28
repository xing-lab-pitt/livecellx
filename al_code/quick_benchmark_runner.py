#!/usr/bin/env python3
"""
Quick Active Learning Benchmark Runner

This module provides functionality to run quick benchmarks for testing
and validation of the Active Learning framework.

Features:
- Quick testing with reduced parameters
- All strategy validation
- Fast feedback for development
- Comprehensive verification that all strategies work
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


class QuickBenchmarkRunner:
    """Main class for running quick AL benchmarks"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize quick benchmark runner with configuration"""
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
    
    def setup_logging(self):
        """Setup logging for the benchmark runner"""
        log_file = Path(self.base_output_dir) / "quick_benchmark.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_single_experiment(self, strategy: str, run_id: int, strategy_name: str) -> bool:
        """Run a single experiment"""
        exp_name = f"{strategy_name}_run{run_id}"
        exp_output = Path(self.base_output_dir) / exp_name
        
        self.logger.info("=" * 50)
        self.logger.info(f"🧪 Testing Strategy: {strategy_name}")
        self.logger.info(f"🔄 Run: {run_id}/{self.n_runs}")
        self.logger.info(f"📁 Output: {exp_output}")
        self.logger.info("=" * 50)
        
        # Create experiment-specific output directory
        exp_output.mkdir(parents=True, exist_ok=True)
        
        # Build command
        seed = 42 + run_id * 100
        cmd = self.build_training_command(strategy, exp_output, seed)
        
        # Run training
        try:
            self.logger.info(f"Running: {' '.join(cmd[:5])}...")  # Show abbreviated command
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.config.get('experiment_timeout', 600)  # 10 minutes default
            )
            
            # Save output
            log_file = exp_output / "training.log"
            with open(log_file, 'w') as f:
                f.write(result.stdout)
            
            if result.returncode == 0:
                self.logger.info(f"  ✅ Success")
                
                # Extract final IoU for quick feedback
                final_iou = self.extract_final_iou(exp_output)
                if final_iou != "N/A":
                    self.logger.info(f"  📊 Final IoU: {final_iou}")
                
                return True
            else:
                self.logger.error(f"  ❌ Failed (check {log_file})")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"  ❌ Timed out")
            return False
        except Exception as e:
            self.logger.error(f"  ❌ Exception: {e}")
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
    
    def extract_final_iou(self, exp_output: Path) -> str:
        """Extract final IoU from results"""
        results_file = exp_output / "experiment_results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                if data.get('test_performances'):
                    final_perf = data['test_performances'][-1]
                    return f"{final_perf.get('test_iou', 0):.4f}"
            except Exception:
                pass
        return "N/A"
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """Run all experiments for all strategies"""
        self.logger.info("⚡ Starting quick benchmark experiments...")
        self.logger.info(f"📊 Total experiments: {self.total_experiments}")
        self.logger.info("")
        
        results = {}
        
        # Run experiments for each strategy
        for strategy, strategy_name in zip(self.strategies, self.strategy_names):
            self.logger.info(f"🧪 Testing Strategy: {strategy_name}")
            
            strategy_results = []
            for run_id in range(1, self.n_runs + 1):
                if self.run_single_experiment(strategy, run_id, strategy_name):
                    self.completed_experiments += 1
                    strategy_results.append(True)
                else:
                    self.failed_experiments += 1
                    strategy_results.append(False)
                
                self.logger.info("")
            
            results[strategy_name] = {
                'success_count': sum(strategy_results),
                'total_count': len(strategy_results),
                'success_rate': sum(strategy_results) / len(strategy_results)
            }
        
        return results
    
    def analyze_results(self, results: Dict[str, Any]):
        """Analyze and display results"""
        end_time = time.time()
        total_time = end_time - self.start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        
        self.logger.info("=" * 50)
        self.logger.info("📈 QUICK BENCHMARK RESULTS")
        self.logger.info("=" * 50)
        self.logger.info(f"✅ Completed: {self.completed_experiments} experiments")
        self.logger.info(f"❌ Failed: {self.failed_experiments} experiments")
        self.logger.info(f"⏱️  Total Time: {minutes}m {seconds}s")
        self.logger.info("")
        
        if self.completed_experiments > 0:
            self.logger.info("🔬 Strategy Results:")
            self.logger.info("=" * 55)
            
            for strategy_name, result in results.items():
                success_rate = result['success_rate'] * 100
                status = "✅" if result['success_count'] == result['total_count'] else "⚠️"
                self.logger.info(f"{status} {strategy_name:20s}: {result['success_count']}/{result['total_count']} ({success_rate:.0f}%)")
            
            # Quick performance analysis if results exist
            self.perform_quick_analysis()
            
            self.logger.info("")
            self.logger.info(f"📁 Results saved in: {self.base_output_dir}")
            self.logger.info(f"🔍 Individual logs: {self.base_output_dir}/*/training.log")
            self.logger.info("")
            
            if self.failed_experiments == 0:
                self.logger.info("🎉 ALL STRATEGIES WORKING! Ready for comprehensive benchmark.")
                self.logger.info("💡 Run: bash run_comprehensive_benchmark_v2.sh")
            else:
                self.logger.info("⚠️  Some experiments failed. Check logs before running full benchmark.")
        else:
            self.logger.info("❌ No experiments completed successfully.")
    
    def perform_quick_analysis(self):
        """Perform quick analysis of results"""
        try:
            base_dir = Path(self.base_output_dir)
            results = []
            
            for exp_dir in base_dir.glob("*_run*"):
                results_file = exp_dir / "experiment_results.json"
                if results_file.exists():
                    try:
                        with open(results_file, 'r') as f:
                            data = json.load(f)
                        if data.get('test_performances'):
                            final_perf = data['test_performances'][-1]
                            strategy = exp_dir.name.split('_run')[0]
                            results.append({
                                'strategy': strategy,
                                'final_iou': final_perf.get('test_iou', 0),
                                'final_samples': data['labeled_counts'][-1] if data.get('labeled_counts') else 0
                            })
                    except Exception:
                        continue
            
            if results:
                df = pd.DataFrame(results)
                summary = df.groupby('strategy')['final_iou'].agg(['mean', 'std', 'count']).round(4)
                self.logger.info("")
                self.logger.info("📊 Quick Performance Summary:")
                self.logger.info("=" * 55)
                for strategy, row in summary.iterrows():
                    self.logger.info(f"{strategy:20s}: {row['mean']:.4f} ± {row['std']:.4f} (n={int(row['count'])})")
                
                if len(summary) > 0:
                    best_strategy = df.loc[df['final_iou'].idxmax(), 'strategy']
                    best_iou = df['final_iou'].max()
                    self.logger.info(f"")
                    self.logger.info(f"🏆 Best Strategy: {best_strategy} (IoU: {best_iou:.4f})")
        
        except Exception as e:
            self.logger.warning(f"Could not perform quick analysis: {e}")


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
    """Main entry point for quick benchmark runner"""
    parser = argparse.ArgumentParser(description='Quick Active Learning Benchmark Runner')
    
    # Required arguments
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Base output directory for benchmark results')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing the CTC dataset')
    
    # Strategy configuration
    parser.add_argument('--strategies', nargs='+', 
                       default=['time_interval', 'random', 'uncertainty'],
                       help='List of AL strategies to test')
    parser.add_argument('--strategy_names', nargs='+',
                       default=['Naive_Baseline', 'Random_Sampling', 'Uncertainty_Sampling'],
                       help='Human-readable names for strategies')
    parser.add_argument('--n_runs', type=int, default=2,
                       help='Number of runs per strategy for quick testing')
    
    # Training configuration (quick/reduced parameters)
    parser.add_argument('--al_iterations', type=int, default=3,
                       help='Number of active learning iterations (reduced for speed)')
    parser.add_argument('--initial_samples', type=int, default=100,
                       help='Initial number of labeled samples (reduced for speed)')
    parser.add_argument('--samples_per_iteration', type=int, default=50,
                       help='Number of samples to add per AL iteration (reduced for speed)')
    parser.add_argument('--max_epochs', type=int, default=5,
                       help='Maximum training epochs per iteration (reduced for speed)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size (reduced for speed)')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loading workers (reduced for speed)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for training')
    parser.add_argument('--loss_function', type=str, default='combined',
                       help='Loss function to use')
    parser.add_argument('--base_channels', type=int, default=32,
                       help='Base number of channels in U-Net (reduced for speed)')
    parser.add_argument('--image_size', type=int, default=128,
                       help='Input image size (reduced for speed)')
    
    # Optional configuration
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--subset_ratio', type=float, default=0.05,
                       help='Use subset of data for quick testing (default: 5%)')
    parser.add_argument('--timeout', type=int, default=600,
                       help='Timeout per experiment in seconds (default: 10 minutes)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if len(args.strategies) != len(args.strategy_names):
        parser.error("Number of strategies must match number of strategy names")
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Run benchmark
    runner = QuickBenchmarkRunner(config)
    
    print("=" * 60)
    print("⚡ QUICK AL BENCHMARK (Testing/Demo Mode)")
    print("=" * 60)
    print(f"Data Subset: {config['subset_ratio']} ({config['subset_ratio']*100}% of full dataset)")
    print(f"Strategies: {' '.join(config['strategies'])}")
    print(f"Runs per Strategy: {config['n_runs']}")
    print(f"AL Iterations: {config['al_iterations']}")
    print(f"Max Epochs: {config['max_epochs']} (quick training)")
    print(f"Total Experiments: {len(config['strategies']) * config['n_runs']}")
    print("")
    
    # Validate data
    if not Path(config['data_dir']).exists() or not (Path(config['data_dir']) / "train_data_aux.csv").exists():
        print(f"❌ Data not found. Please ensure {config['data_dir']} exists with CSV files.")
        sys.exit(1)
    
    print("✅ Data validated. Starting quick benchmark...")
    print("")
    
    # Run all experiments
    results = runner.run_all_experiments()
    
    # Analyze results
    runner.analyze_results(results)
    
    print(f"Quick benchmark completed at: {datetime.now().strftime('%c')}")


if __name__ == "__main__":
    main()