# Active Learning for Cell Segmentation

This directory contains the core active learning pipeline for single-cell segmentation using CTC datasets.

## 🚀 Quick Start

### Run Comprehensive Benchmark (Production)
```bash
# From the root directory
bash run_comprehensive_benchmark.sh

# Or directly from al_code/
cd al_code
bash run_comprehensive_benchmark_v2.sh
```

### Quick Testing
```bash
cd al_code
bash run_quick_benchmark.sh      # Quick test with 2 runs, 3 iterations
bash run_debug_benchmark.sh      # Minimal test with 1 iteration, 1 epoch
```

## 📁 Directory Structure

```
al_code/
├── data/                              # Dataset storage (ignored by git)
│   └── comprehensive_ctc_single_cell_data_maximized/
├── results/                           # Experiment outputs (ignored by git)
│   ├── comprehensive_benchmark_v2_*/
│   ├── quick_benchmark_v2_*/
│   └── debug_benchmark_*/
├── benchmark_runner.py               # Main benchmark orchestration
├── collect_csv_metrics.py            # CSV metrics aggregation
├── ctc_segmentation_dataset.py       # Dataset loading classes
├── ctc_segmentation_model.py         # U-Net model definitions
├── create_real_single_cell_data_maximized.py  # Data generation
├── quick_benchmark_runner.py         # Quick testing runner
├── run_comprehensive_benchmark_v2.sh # Production benchmark script
├── run_debug_benchmark.sh            # Debug testing script
├── run_quick_benchmark.sh            # Quick testing script
└── train_ctc_segmentation.py         # Core training script
```

## 🎯 Features

- **3 Active Learning Strategies**: Naive baseline, Random sampling, Uncertainty sampling
- **Statistical Rigor**: Multiple independent runs for significance testing
- **CSV Metrics**: Portable metrics output for easy analysis
- **Comprehensive Testing**: Debug → Quick → Full benchmark pipeline
- **GPU Optimization**: Single GPU training to avoid distributed issues

## 📊 Outputs

Each experiment generates:
- **CSV Metrics**: Epoch, iteration, and sample selection data
- **Model Checkpoints**: Best models saved automatically  
- **Logs**: Detailed training and benchmark logs
- **Analysis**: Statistical summaries and visualizations

## 🔧 Configuration

Key parameters in the shell scripts:
- `AL_ITERATIONS`: Number of active learning rounds (default: 8)
- `MAX_EPOCHS`: Training epochs per iteration (default: 30)  
- `N_RUNS`: Independent runs per strategy (default: 5)
- `INITIAL_SAMPLES`: Starting labeled samples (default: 500)
- `SAMPLES_PER_ITERATION`: Samples added per round (default: 250)

## 🧪 Testing Pipeline

1. **Debug Test**: `bash run_debug_benchmark.sh` (~1 minute)
2. **Quick Test**: `bash run_quick_benchmark.sh` (~10 minutes)  
3. **Full Benchmark**: `bash run_comprehensive_benchmark_v2.sh` (~8 hours)

## 📈 Results Analysis

Automated CSV collection and analysis:
```bash
python collect_csv_metrics.py ./results/comprehensive_benchmark_v2_YYYYMMDD_HHMMSS/
```

This generates unified CSV files for easy visualization and statistical analysis.