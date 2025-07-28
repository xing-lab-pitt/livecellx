#!/bin/bash
# =============================================================================
# Quick AL Benchmark Script v2 (for testing/demo)
# =============================================================================

set -e

echo "=============================================="
echo "⚡ QUICK AL BENCHMARK v2 (Testing/Demo Mode)"
echo "=============================================="

# Quick test parameters - paths relative to parent directory (livecellx_al)
DATA_DIR="./al_code/data/comprehensive_ctc_single_cell_data_maximized"
OUTPUT_DIR="./al_code/results/quick_benchmark_v2_$(date +%Y%m%d_%H%M%S)"
N_RUNS=2  # Fewer runs for quick testing
SUBSET_RATIO=0.05  # Use 5% of data for speed
AL_ITERATIONS=3
INITIAL_SAMPLES=100
SAMPLES_PER_ITERATION=50
MAX_EPOCHS=5
BATCH_SIZE=8

# ALL AL strategies to verify they work correctly
STRATEGIES="time_interval random uncertainty"
STRATEGY_NAMES="Naive_Baseline Random_Sampling Uncertainty_Sampling"

# Hardware optimization - Use single GPU to avoid distributed training issues
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2

echo "Quick Benchmark Configuration:"
echo "- Data Subset: ${SUBSET_RATIO} ($(echo "$SUBSET_RATIO * 100" | bc)% of full dataset)"
echo "- Strategies: $STRATEGIES"
echo "- Strategy Names: $STRATEGY_NAMES"
echo "- Runs per Strategy: $N_RUNS"
echo "- AL Iterations: $AL_ITERATIONS"
echo "- Max Epochs: $MAX_EPOCHS (quick training)"
echo "- Total Experiments: $(echo "$STRATEGIES" | wc -w) × $N_RUNS = $(($(echo $STRATEGIES | wc -w) * N_RUNS))"
echo ""

# =============================================================================
# VALIDATION CHECKS
# =============================================================================

echo "🔍 Running validation checks..."

# Check data directory
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory $DATA_DIR not found!"
    exit 1
fi

if [ ! -f "$DATA_DIR/train_data_aux.csv" ]; then
    echo "❌ Error: Required CSV file not found!"
    exit 1
fi

# Check quick benchmark runner
if [ ! -f "./al_code/quick_benchmark_runner.py" ]; then
    echo "❌ Error: quick_benchmark_runner.py not found!"
    exit 1
fi

# Check training script
if [ ! -f "./al_code/train_ctc_segmentation.py" ]; then
    echo "❌ Error: train_ctc_segmentation.py not found!"
    exit 1
fi

# Check Python dependencies
echo "🔍 Checking Python dependencies..."
python -c "
import torch
import pytorch_lightning as pl
import torchmetrics
import pandas as pd
import numpy as np
print('✅ Core dependencies available')
" || {
    echo "❌ Error: Missing required Python dependencies"
    exit 1
}

echo "✅ Validation checks passed"
echo ""

# =============================================================================
# RUN QUICK BENCHMARK
# =============================================================================

echo "🚀 Starting quick benchmark using Python runner..."
echo ""

# Build command - execute from al_code directory
cmd="cd al_code && python quick_benchmark_runner.py"
cmd="$cmd --output_dir \"./results/quick_benchmark_v2_$(date +%Y%m%d_%H%M%S)\""
cmd="$cmd --data_dir \"./data/comprehensive_ctc_single_cell_data_maximized\""
cmd="$cmd --strategies $STRATEGIES"
cmd="$cmd --strategy_names $STRATEGY_NAMES"
cmd="$cmd --n_runs $N_RUNS"
cmd="$cmd --al_iterations $AL_ITERATIONS"
cmd="$cmd --initial_samples $INITIAL_SAMPLES"
cmd="$cmd --samples_per_iteration $SAMPLES_PER_ITERATION"
cmd="$cmd --max_epochs $MAX_EPOCHS"
cmd="$cmd --batch_size $BATCH_SIZE"
cmd="$cmd --num_workers 2"
cmd="$cmd --learning_rate 0.001"
cmd="$cmd --loss_function combined"
cmd="$cmd --base_channels 32"
cmd="$cmd --image_size 128"
cmd="$cmd --subset_ratio $SUBSET_RATIO"
cmd="$cmd --timeout 600"  # 10 minutes per experiment
cmd="$cmd --debug"

echo "Running command:"
echo "$cmd"
echo ""

# Execute benchmark
if eval "$cmd"; then
    echo ""
    echo "=============================================="
    echo "🎉 QUICK BENCHMARK COMPLETED!"
    echo "=============================================="
    echo ""
    echo "📁 Results Location: $OUTPUT_DIR"
    echo "📝 Log File: $OUTPUT_DIR/quick_benchmark.log"
    echo ""
    echo "🔍 **Verification Results:**"
    echo "   - Check that ALL strategies completed successfully"
    echo "   - If all strategies work, the comprehensive benchmark is ready"
    echo ""
    # Collect CSV metrics
    echo "📊 Collecting CSV metrics..."
    if cd al_code && python collect_csv_metrics.py "./results/quick_benchmark_v2_$(date +%Y%m%d_%H%M%S)"; then
        echo "✅ CSV metrics collected: $OUTPUT_DIR/csv_metrics_analysis/"
        echo ""
        echo "📋 **CSV Metrics Available:**"
        echo "   - All epoch metrics: $OUTPUT_DIR/csv_metrics_analysis/all_epoch_metrics.csv"
        echo "   - All iteration metrics: $OUTPUT_DIR/csv_metrics_analysis/all_iteration_metrics.csv"
        echo "   - Sample selection logs: $OUTPUT_DIR/csv_metrics_analysis/all_sample_selection.csv"
        echo "   - Summary statistics: $OUTPUT_DIR/csv_metrics_analysis/csv_summary_statistics.csv"
    else
        echo "⚠️  Warning: CSV metrics collection failed"
    fi
    echo ""
    echo "💡 **Next Steps:**"
    echo "   1. Review the strategy results above"
    echo "   2. Check CSV metrics for detailed analysis"
    echo "   3. If all strategies passed, run: bash run_comprehensive_benchmark_v2.sh"
    echo "   4. If any failed, check individual logs in $OUTPUT_DIR/*/training.log"
    echo ""
    echo "Quick benchmark completed at: $(date)"
else
    echo ""
    echo "❌ QUICK BENCHMARK FAILED!"
    echo "Check the logs for details: $OUTPUT_DIR/"
    exit 1
fi