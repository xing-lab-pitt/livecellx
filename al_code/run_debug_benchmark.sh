#!/bin/bash
# =============================================================================
# Debug CTC Segmentation AL Benchmark Script 
# =============================================================================
# 
# This script runs a minimal benchmark for debugging with:
# - 1 AL iteration only
# - 1 epoch per iteration
# - Small sample sizes
# - All strategies tested
# - Quick validation that the pipeline works
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "🐛 DEBUG AL BENCHMARK (1 iteration, 1 epoch)"
echo "=============================================="

# Debug parameters - minimal for fast testing, paths relative to parent directory (livecellx_al)
DATA_DIR="./al_code/data/comprehensive_ctc_single_cell_data_maximized"
OUTPUT_DIR="./al_code/results/debug_benchmark_$(date +%Y%m%d_%H%M%S)"
N_RUNS=1  # Single run for debugging
AL_ITERATIONS=1  # Just 1 iteration to test the pipeline
INITIAL_SAMPLES=50  # Very small sample size
SAMPLES_PER_ITERATION=25  # Small increments
MAX_EPOCHS=1  # Single epoch for speed
BATCH_SIZE=8  # Small batch for speed
NUM_WORKERS=2  # Limited workers

# Test all strategies to ensure they work
STRATEGIES="time_interval random uncertainty"
STRATEGY_NAMES="Naive_Baseline Random_Sampling Uncertainty_Sampling"

# Hardware optimization
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2

echo "Debug Configuration:"
echo "- Data Directory: $DATA_DIR"
echo "- Output Directory: $OUTPUT_DIR"
echo "- Strategies: $STRATEGIES"
echo "- AL Iterations: $AL_ITERATIONS (debug mode)"
echo "- Epochs per iteration: $MAX_EPOCHS (debug mode)"
echo "- Initial samples: $INITIAL_SAMPLES"
echo "- Batch size: $BATCH_SIZE"
echo "- Total experiments: $(echo $STRATEGIES | wc -w) strategies × $N_RUNS runs = $(($(echo $STRATEGIES | wc -w) * N_RUNS))"
echo ""

# =============================================================================
# VALIDATION CHECKS
# =============================================================================

echo "🔍 Running validation checks..."

# Check maximized data directory
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Maximized data directory $DATA_DIR not found!"
    echo "   Please run create_real_single_cell_data_maximized.py first"
    exit 1
fi

if [ ! -f "$DATA_DIR/train_data_aux.csv" ]; then
    echo "❌ Error: Required CSV file not found in maximized dataset!"
    exit 1
fi

# Check CSV file content
sample_count=$(wc -l < "$DATA_DIR/train_data_aux.csv")
echo "📊 Found $((sample_count - 1)) samples in maximized dataset"

if [ "$sample_count" -lt 1000 ]; then
    echo "⚠️  Warning: Very few samples ($sample_count) in dataset"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check benchmark runner
if [ ! -f "./al_code/quick_benchmark_runner.py" ]; then
    echo "❌ Error: quick_benchmark_runner.py not found!"
    exit 1
fi

# Check training script
if [ ! -f "./al_code/train_ctc_segmentation.py" ]; then
    echo "❌ Error: train_ctc_segmentation.py not found!"
    exit 1
fi

# Check column mapping for maximized dataset
echo "🔍 Checking dataset column compatibility..."
python -c "
import pandas as pd
df = pd.read_csv('$DATA_DIR/train_data_aux.csv')
print(f'Dataset columns: {list(df.columns)}')

# Check required columns exist (maximized dataset format)
required_cols = ['raw_img_path', 'mask_path', 'dataset', 'cell_id', 'padding']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f'Missing required columns: {missing_cols}')

print('✅ All required columns present')
print(f'Sample data shape: {df.shape}')
print(f'Unique datasets: {df[\"dataset\"].nunique()}')
print(f'Sample paths check:')
print(f'  Raw: {df[\"raw_img_path\"].iloc[0][:50]}...')
print(f'  Mask: {df[\"mask_path\"].iloc[0][:50]}...')
" || {
    echo "❌ Error: Dataset format validation failed"
    exit 1
}

echo "✅ Validation checks passed"
echo ""

# =============================================================================
# RUN DEBUG BENCHMARK
# =============================================================================

echo "🚀 Starting debug benchmark..."
echo ""

# Build command for debug run - execute from al_code directory
cmd="cd al_code && python quick_benchmark_runner.py"
cmd="$cmd --output_dir \"./results/debug_benchmark_$(date +%Y%m%d_%H%M%S)\""
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
cmd="$cmd --base_channels 32"  # Smaller model for speed
cmd="$cmd --image_size 128"    # Smaller images for speed
cmd="$cmd --subset_ratio 0.1"  # Use 10% of data for debugging
cmd="$cmd --timeout 300"       # 5 minutes per experiment
cmd="$cmd --debug"

echo "Running debug command:"
echo "$cmd"
echo ""

# Execute debug benchmark
start_time=$(date +%s)

if eval "$cmd"; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo ""
    echo "=============================================="
    echo "🎉 DEBUG BENCHMARK COMPLETED!"
    echo "=============================================="
    echo ""
    echo "⏱️  **Execution Time:** ${duration} seconds"
    echo "📁 **Results Location:** $OUTPUT_DIR"
    echo ""
    echo "🔍 **Debug Results:**"
    
    # Check if all strategies completed (correct directory naming)
    success_count=0
    total_strategies=$(echo $STRATEGIES | wc -w)
    strategy_names_array=($STRATEGY_NAMES)
    
    i=0
    for strategy in $STRATEGIES; do
        strategy_name=${strategy_names_array[$i]}
        if [ -d "$OUTPUT_DIR/${strategy_name}_run1" ]; then
            echo "   ✅ $strategy ($strategy_name): COMPLETED"
            success_count=$((success_count + 1))
        else
            echo "   ❌ $strategy ($strategy_name): FAILED"
        fi
        i=$((i + 1))
    done
    
    echo ""
    echo "📊 **Success Rate:** $success_count/$total_strategies strategies completed"
    
    if [ "$success_count" -eq "$total_strategies" ]; then
        echo ""
        echo "🎯 **PIPELINE VALIDATION SUCCESSFUL!**"
        echo "   - All AL strategies work correctly"
        echo "   - Maximized dataset loads properly"
        echo "   - Training pipeline is functional"
        echo "   - Model training completes without errors"
        echo ""
        echo "💡 **Next Steps:**"
        echo "   1. The debug test passed - comprehensive benchmark is ready"
        echo "   2. To run quick test: bash run_quick_benchmark.sh"
        echo "   3. To run full benchmark: bash run_comprehensive_benchmark_v2.sh"
        echo ""
        echo "✅ Ready for production experiments!"
    else
        echo ""
        echo "⚠️  **PARTIAL SUCCESS**"
        echo "   Some strategies failed. Check individual logs:"
        i=0
        for strategy in $STRATEGIES; do
            strategy_name=${strategy_names_array[$i]}
            if [ ! -d "$OUTPUT_DIR/${strategy_name}_run1" ]; then
                echo "   - $OUTPUT_DIR/${strategy_name}_run1/training.log"
            fi
            i=$((i + 1))
        done
        echo ""
        echo "🔧 Fix the failing strategies before running comprehensive benchmark"
    fi
    
else
    echo ""
    echo "❌ DEBUG BENCHMARK FAILED!"
    echo "Check the logs for details: $OUTPUT_DIR/"
    echo ""
    echo "🔧 **Troubleshooting:**"
    echo "   1. Check $OUTPUT_DIR/debug_benchmark.log"
    echo "   2. Verify CUDA availability: python -c 'import torch; print(torch.cuda.is_available())'"
    echo "   3. Check data format: head -5 $DATA_DIR/train_data_aux.csv"
    echo "   4. Test individual components manually"
    exit 1
fi