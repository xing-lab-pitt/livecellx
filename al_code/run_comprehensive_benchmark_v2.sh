#!/bin/bash
# =============================================================================
# Comprehensive CTC Segmentation AL Benchmark Script (Version 2)
# =============================================================================
# 
# This script runs a comprehensive comparison between NAIVE baseline strategy 
# and sophisticated Active Learning strategies with statistical rigor.
#
# Features:
# - Uses modular Python benchmark runner
# - Multiple runs per strategy for statistical significance
# - Comprehensive metric collection
# - Automated results analysis and visualization
# - Performance comparison with confidence intervals
# - Final publication-ready results
# =============================================================================

set -e  # Exit on error

# =============================================================================
# CONFIGURATION
# =============================================================================

echo "=============================================="
echo "🚀 COMPREHENSIVE AL BENCHMARK EXPERIMENT v2"
echo "=============================================="

# Comprehensive benchmark parameters - paths relative to parent directory (livecellx_al)
DATA_DIR="./al_code/data/comprehensive_ctc_single_cell_data_maximized"
BASE_OUTPUT_DIR="./al_code/results/comprehensive_benchmark_v2_$(date +%Y%m%d_%H%M%S)"
N_RUNS=5  # Number of independent runs per strategy for statistical significance
AL_ITERATIONS=8
INITIAL_SAMPLES=500
SAMPLES_PER_ITERATION=250
MAX_EPOCHS=30
BATCH_SIZE=16
NUM_WORKERS=8

# Strategies to compare (NAIVE baseline + sophisticated AL)
STRATEGIES="time_interval random uncertainty"
STRATEGY_NAMES="Naive_Baseline Random_Sampling Uncertainty_Sampling"

# Hardware optimization - Use single GPU to avoid distributed training issues
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

echo "Comprehensive Benchmark Configuration:"
echo "- Base Output: $BASE_OUTPUT_DIR"
echo "- Strategies: $STRATEGIES"
echo "- Strategy Names: $STRATEGY_NAMES"
echo "- Runs per Strategy: $N_RUNS"
echo "- AL Iterations: $AL_ITERATIONS"
echo "- Initial Samples: $INITIAL_SAMPLES"
echo "- Samples per Iteration: $SAMPLES_PER_ITERATION"
echo "- Final Sample Count: $((INITIAL_SAMPLES + AL_ITERATIONS * SAMPLES_PER_ITERATION))"
echo "- Max Epochs: $MAX_EPOCHS"
echo "- Total Experiments: $(($(echo $STRATEGIES | wc -w) * N_RUNS))"
echo ""

# =============================================================================
# VALIDATION CHECKS
# =============================================================================

echo "🔍 Running pre-flight checks..."

# Check data directory
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory $DATA_DIR not found!"
    exit 1
fi

if [ ! -f "$DATA_DIR/train_data_aux.csv" ]; then
    echo "❌ Error: Required CSV file not found!"
    exit 1
fi

# Check benchmark runner
if [ ! -f "./al_code/benchmark_runner.py" ]; then
    echo "❌ Error: benchmark_runner.py not found!"
    exit 1
fi

# Check training script
if [ ! -f "./al_code/train_ctc_segmentation.py" ]; then
    echo "❌ Error: train_ctc_segmentation.py not found!"
    exit 1
fi

# Check available resources
available_space=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
required_space=20
if [ "$available_space" -lt "$required_space" ]; then
    echo "⚠️  Warning: Low disk space ($available_space GB available, recommend ${required_space}GB)"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check CUDA availability
if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
    echo "⚠️  Warning: CUDA not available, training will be slow"
    read -p "Continue with CPU training? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python dependencies
echo "🔍 Checking Python dependencies..."
python -c "
import torch
import pytorch_lightning as pl
import torchmetrics
import tensorboard
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
print('✅ All dependencies available')
" || {
    echo "❌ Error: Missing required Python dependencies"
    echo "Please install: torch pytorch-lightning torchmetrics tensorboard pandas numpy matplotlib seaborn opencv-python"
    exit 1
}

echo "✅ Pre-flight checks passed"
echo ""

# =============================================================================
# RUN COMPREHENSIVE BENCHMARK
# =============================================================================

echo "🚀 Starting comprehensive benchmark using Python runner..."
echo ""

# Build command - execute Python script from al_code directory
cmd="cd al_code && python benchmark_runner.py"
cmd="$cmd --output_dir \"./results/comprehensive_benchmark_v2_$(date +%Y%m%d_%H%M%S)\""
cmd="$cmd --data_dir \"./data/comprehensive_ctc_single_cell_data_maximized\""
cmd="$cmd --strategies $STRATEGIES"
cmd="$cmd --strategy_names $STRATEGY_NAMES"
cmd="$cmd --n_runs $N_RUNS"
cmd="$cmd --al_iterations $AL_ITERATIONS"
cmd="$cmd --initial_samples $INITIAL_SAMPLES"
cmd="$cmd --samples_per_iteration $SAMPLES_PER_ITERATION"
cmd="$cmd --max_epochs $MAX_EPOCHS"
cmd="$cmd --batch_size $BATCH_SIZE"
cmd="$cmd --num_workers $NUM_WORKERS"
cmd="$cmd --learning_rate 0.001"
cmd="$cmd --loss_function combined"
cmd="$cmd --base_channels 64"
cmd="$cmd --image_size 256"
cmd="$cmd --timeout 7200"  # 2 hours per experiment

echo "Running command:"
echo "$cmd"
echo ""

# Execute benchmark
if eval "$cmd"; then
    echo ""
    echo "=============================================="
    echo "🎉 COMPREHENSIVE BENCHMARK COMPLETED!"
    echo "=============================================="
    echo ""
    echo "📁 **Output Locations:**"
    echo "   - Base Directory: $BASE_OUTPUT_DIR"
    echo "   - Analysis: $BASE_OUTPUT_DIR/comprehensive_analysis/"
    echo "   - Report: $BASE_OUTPUT_DIR/comprehensive_analysis/comprehensive_benchmark_report.md"
    echo "   - Plots: $BASE_OUTPUT_DIR/comprehensive_analysis/comprehensive_benchmark_results.png"
    echo ""
    echo "📈 **Key Files:**"
    echo "   - Master Log: $BASE_OUTPUT_DIR/master_experiment_log.txt"
    echo "   - Raw Results: $BASE_OUTPUT_DIR/comprehensive_analysis/all_experiment_results.csv"
    echo "   - Statistics: $BASE_OUTPUT_DIR/comprehensive_analysis/strategy_summary_statistics.csv"
    echo ""
    
    # Collect CSV metrics from all experiments
    echo "📊 Collecting CSV metrics from all experiments..."
    if cd al_code && python collect_csv_metrics.py "./results/comprehensive_benchmark_v2_$(date +%Y%m%d_%H%M%S)"; then
        echo "✅ CSV metrics collected successfully"
        echo ""
        echo "📋 **CSV Metrics Files:**"
        echo "   - Epoch Metrics: $BASE_OUTPUT_DIR/csv_metrics_analysis/all_epoch_metrics.csv"
        echo "   - Iteration Metrics: $BASE_OUTPUT_DIR/csv_metrics_analysis/all_iteration_metrics.csv"
        echo "   - Sample Selection: $BASE_OUTPUT_DIR/csv_metrics_analysis/all_sample_selection.csv"
        echo "   - Summary Stats: $BASE_OUTPUT_DIR/csv_metrics_analysis/csv_summary_statistics.csv"
        echo "   - Learning Curves: $BASE_OUTPUT_DIR/csv_metrics_analysis/learning_curves.csv"
    else
        echo "⚠️  Warning: CSV metrics collection failed, but benchmark completed successfully"
    fi
    echo ""
    echo "🔍 **Next Steps:**"
    echo "   1. Review the comprehensive report"
    echo "   2. Examine the statistical analysis"
    echo "   3. Use CSV files for custom visualization and analysis"
    echo "   4. Check individual experiment logs if needed"
    echo "   5. Use results for publication/presentation"
    echo ""
    echo "Benchmark completed at: $(date)"
else
    echo ""
    echo "❌ BENCHMARK FAILED!"
    echo "Check the logs for details: $BASE_OUTPUT_DIR/"
    exit 1
fi