#!/usr/bin/env python3
"""
Comprehensive CTC Single-Cell Segmentation Training Script

Active learning framework for single-cell segmentation on CTC datasets using U-Net.
Supports multiple AL strategies, comprehensive metrics tracking, and robust experimentation.
"""

import argparse
import os
import sys
import json
import logging
import random
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split

# Import our custom modules
from ctc_segmentation_model import CTCSegmentationModel, create_ctc_segmentation_model
from ctc_segmentation_dataset import CTCSegmentationDataModule, create_ctc_dataloader

warnings.filterwarnings('ignore')


class MetricsTracker:
    """CSV-based metrics tracking for active learning experiments"""
    
    def __init__(self, output_dir: Path, experiment_name: str = "al_experiment"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # Initialize metric DataFrames
        self.epoch_metrics = pd.DataFrame()
        self.iteration_metrics = pd.DataFrame()
        self.sample_selection_log = pd.DataFrame()
        
        # File paths
        self.epoch_metrics_file = self.output_dir / f"{experiment_name}_epoch_metrics.csv"
        self.iteration_metrics_file = self.output_dir / f"{experiment_name}_iteration_metrics.csv"
        self.selection_log_file = self.output_dir / f"{experiment_name}_sample_selection.csv"
    
    def log_epoch_metrics(self, iteration: int, epoch: int, metrics: Dict[str, float]):
        """Log per-epoch training metrics"""
        row = {
            'timestamp': datetime.now().isoformat(),
            'iteration': iteration,
            'epoch': epoch,
            **metrics
        }
        self.epoch_metrics = pd.concat([self.epoch_metrics, pd.DataFrame([row])], ignore_index=True)
        self.epoch_metrics.to_csv(self.epoch_metrics_file, index=False)
    
    def log_iteration_metrics(self, iteration: int, metrics: Dict[str, Any]):
        """Log per-iteration AL metrics"""
        row = {
            'timestamp': datetime.now().isoformat(),
            'iteration': iteration,
            **metrics
        }
        self.iteration_metrics = pd.concat([self.iteration_metrics, pd.DataFrame([row])], ignore_index=True)
        self.iteration_metrics.to_csv(self.iteration_metrics_file, index=False)
    
    def log_sample_selection(self, iteration: int, strategy: str, selected_indices: List[int], 
                           selection_metrics: Dict[str, Any] = None):
        """Log sample selection details"""
        rows = []
        selection_metrics = selection_metrics or {}
        
        for idx in selected_indices:
            row = {
                'timestamp': datetime.now().isoformat(),
                'iteration': iteration,
                'strategy': strategy,
                'selected_index': idx,
                **selection_metrics
            }
            rows.append(row)
        
        if rows:
            new_df = pd.DataFrame(rows)
            self.sample_selection_log = pd.concat([self.sample_selection_log, new_df], ignore_index=True)
            self.sample_selection_log.to_csv(self.selection_log_file, index=False)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from collected metrics"""
        summary = {}
        
        if not self.iteration_metrics.empty:
            # Best performance per iteration
            summary['best_test_iou'] = self.iteration_metrics['test_iou'].max() if 'test_iou' in self.iteration_metrics.columns else None
            summary['final_test_iou'] = self.iteration_metrics['test_iou'].iloc[-1] if 'test_iou' in self.iteration_metrics.columns else None
            summary['total_iterations'] = len(self.iteration_metrics)
            summary['final_labeled_samples'] = self.iteration_metrics['labeled_samples'].iloc[-1] if 'labeled_samples' in self.iteration_metrics.columns else None
        
        if not self.epoch_metrics.empty:
            summary['total_epochs'] = len(self.epoch_metrics)
            summary['best_val_loss'] = self.epoch_metrics['val_loss'].min() if 'val_loss' in self.epoch_metrics.columns else None
        
        return summary


class CSVMetricsCallback(Callback):
    """Custom callback to extract and log metrics to CSV"""
    
    def __init__(self, metrics_tracker: MetricsTracker, al_iteration: int):
        super().__init__()
        self.metrics_tracker = metrics_tracker
        self.al_iteration = al_iteration
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Log training metrics at end of each epoch"""
        if trainer.logged_metrics:
            metrics = {}
            for key, value in trainer.logged_metrics.items():
                if isinstance(value, torch.Tensor):
                    metrics[key] = float(value.item())
                else:
                    metrics[key] = float(value)
            
            self.metrics_tracker.log_epoch_metrics(
                iteration=self.al_iteration,
                epoch=trainer.current_epoch,
                metrics=metrics
            )
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation metrics at end of each epoch"""
        if trainer.logged_metrics:
            metrics = {}
            for key, value in trainer.logged_metrics.items():
                if isinstance(value, torch.Tensor):
                    metrics[key] = float(value.item())
                else:
                    metrics[key] = float(value)
            
            # Update the latest epoch entry with validation metrics
            # This ensures train and val metrics are in the same row
            if not self.metrics_tracker.epoch_metrics.empty:
                latest_idx = self.metrics_tracker.epoch_metrics.index[-1]
                for key, value in metrics.items():
                    if key.startswith('val_'):
                        self.metrics_tracker.epoch_metrics.loc[latest_idx, key] = value
                
                # Save updated metrics
                self.metrics_tracker.epoch_metrics.to_csv(
                    self.metrics_tracker.epoch_metrics_file, index=False
                )


def setup_logging(log_dir: Path, debug: bool = False) -> logging.Logger:
    """Setup comprehensive logging"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized: {log_file}")
    return logger


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)


def load_ctc_data(data_dir: str, subset_ratio: Optional[float] = None, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Load CTC single-cell data from CSV or create from directory structure"""
    data_path = Path(data_dir)
    
    # Look for existing CSV files in the directory
    csv_files = []
    for pattern in ["**/train_data_aux.csv", "**/test_data_aux.csv", "**/val_data_aux.csv", "**/single_cell_data*.csv"]:
        csv_files.extend(list(data_path.glob(pattern)))
    
    if csv_files:
        # Use train_data_aux.csv if available (most comprehensive)
        train_csv = [f for f in csv_files if "train_data_aux.csv" in str(f)]
        if train_csv:
            if logger:
                logger.info(f"Loading data from {train_csv[0]}")
            df = pd.read_csv(train_csv[0])
        else:
            if logger:
                logger.info(f"Loading data from {csv_files[0]}")
            df = pd.read_csv(csv_files[0])
    else:
        raise FileNotFoundError(f"No CTC data CSV files found in {data_dir}")
    
    # Apply subset for debugging
    if subset_ratio and subset_ratio < 1.0:
        n_samples = int(len(df) * subset_ratio)
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
        if logger:
            logger.info(f"Using subset: {len(df)} samples ({subset_ratio:.1%})")
    
    # Map column names to expected format
    if 'raw' in df.columns and 'raw_img_path' not in df.columns:
        df['raw_img_path'] = df['raw']
    if 'seg' in df.columns and 'mask_path' not in df.columns:
        df['mask_path'] = df['seg']
    if 'source_origin' in df.columns and 'source' not in df.columns:
        df['source'] = df['source_origin']
    
    # Check required columns after mapping
    required_cols = ['raw_img_path', 'mask_path']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        if logger:
            logger.error(f"Missing required columns: {missing_cols}")
            logger.info(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if logger:
        logger.info(f"Loaded {len(df)} samples")
        if 'source' in df.columns:
            logger.info(f"Sources: {df['source'].unique()}")
    
    return df


def create_data_splits(
    df: pd.DataFrame, 
    test_size: float = 0.2, 
    val_size: float = 0.1,
    stratify_by: str = "source"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val/test splits with stratification"""
    
    # Check if stratification is possible
    stratify_col = None
    if stratify_by in df.columns and len(df[stratify_by].unique()) > 1:
        # Check if each class has at least 2 samples for stratification
        class_counts = df[stratify_by].value_counts()
        if class_counts.min() >= 2:
            stratify_col = df[stratify_by]
    
    # First split: train+val vs test
    try:
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=stratify_col
        )
    except ValueError:
        # Fallback to non-stratified split
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=None
        )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)  # Adjust for remaining data
    train_stratify = None
    if stratify_col is not None:
        # Check if stratification is still possible for the remaining data
        remaining_class_counts = train_val_df[stratify_by].value_counts()
        if remaining_class_counts.min() >= 2:
            train_stratify = train_val_df[stratify_by]
    
    try:
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_ratio, random_state=42, stratify=train_stratify
        )
    except ValueError:
        # Fallback to non-stratified split
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_ratio, random_state=42, stratify=None
        )
    
    print(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def select_samples_for_al(
    train_df: pd.DataFrame,
    labeled_indices: np.ndarray,
    unlabeled_indices: np.ndarray,
    model: Optional[CTCSegmentationModel],
    strategy: str,
    quota: int,
    **kwargs
) -> np.ndarray:
    """Select samples for active learning"""
    
    unlabeled_df = train_df.iloc[unlabeled_indices]
    
    if strategy == "random":
        selected_idx = np.random.choice(len(unlabeled_df), size=min(quota, len(unlabeled_df)), replace=False)
        
    elif strategy == "time_interval":
        # Naive baseline: select by time interval
        if 'timepoint' in unlabeled_df.columns:
            # Sort by timepoint and select evenly spaced samples
            sorted_df = unlabeled_df.sort_values('timepoint')
            step = max(1, len(sorted_df) // quota)
            selected_idx = np.arange(0, len(sorted_df), step)[:quota]
            # Map back to original indices
            selected_idx = sorted_df.iloc[selected_idx].index.values
            selected_idx = np.array([np.where(unlabeled_indices == idx)[0][0] for idx in selected_idx])
        else:
            # Fallback to random if no timepoint info
            selected_idx = np.random.choice(len(unlabeled_df), size=min(quota, len(unlabeled_df)), replace=False)
    
    elif strategy == "uncertainty":
        if model is None:
            raise ValueError("Model required for uncertainty sampling")
        
        # Get uncertainty estimates
        uncertainties = get_model_uncertainties(model, unlabeled_df, **kwargs)
        selected_idx = np.argsort(uncertainties)[-quota:]  # Most uncertain
    
    else:
        raise ValueError(f"Unknown AL strategy: {strategy}")
    
    return selected_idx


def get_model_uncertainties(
    model: CTCSegmentationModel, 
    unlabeled_df: pd.DataFrame,
    batch_size: int = 16,
    mc_samples: int = 10
) -> np.ndarray:
    """Get uncertainty estimates from model"""
    
    # Create temporary dataloader for unlabeled data
    temp_loader = create_ctc_dataloader(
        unlabeled_df, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        include_square_mask=True,
        square_size_range=(5, 50)
    )
    
    model.eval()
    uncertainties = []
    
    with torch.no_grad():
        for batch in temp_loader:
            if isinstance(batch, dict):
                images = batch['image']
            else:
                images = batch[0]
            
            # Get uncertainty using Monte Carlo dropout
            _, uncertainty = model.get_uncertainty(images, n_samples=mc_samples)
            batch_uncertainties = uncertainty.mean(dim=[1, 2, 3]).cpu().numpy()  # Average over spatial dims
            uncertainties.extend(batch_uncertainties)
    
    return np.array(uncertainties)


def validate_data_paths(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Validate that all data paths exist"""
    valid_mask = df['raw_img_path'].apply(lambda x: Path(x).exists()) & \
                 df['mask_path'].apply(lambda x: Path(x).exists())
    
    if not valid_mask.all():
        if logger:
            logger.warning(f"Removing {(~valid_mask).sum()} samples with missing files")
        df = df[valid_mask].reset_index(drop=True)
    
    return df


def train_model(
    model: CTCSegmentationModel,
    data_module: CTCSegmentationDataModule,
    max_epochs: int,
    checkpoint_dir: Path,
    logger_pl: TensorBoardLogger,
    metrics_tracker: Optional[MetricsTracker] = None,
    iteration: int = 0,
    debug: bool = False
) -> Dict[str, Any]:
    """Train model with comprehensive callbacks"""
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True
    )
    
    callbacks = [checkpoint_callback, early_stopping]
    
    # Add CSV metrics callback if tracker provided
    if metrics_tracker is not None:
        csv_callback = CSVMetricsCallback(metrics_tracker, iteration)
        callbacks.append(csv_callback)
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger_pl,
        callbacks=callbacks,
        accelerator="auto",
        devices=1,  # Force single GPU to avoid distributed training issues
        precision=16 if torch.cuda.is_available() else 32,
        log_every_n_steps=10,
        limit_train_batches=1.0,  # Always use full training set
        limit_val_batches=1.0,  # Always use full validation set
        enable_progress_bar=True,
        enable_model_summary=debug
    )
    
    # Train model
    trainer.fit(model, 
                train_dataloaders=data_module.get_train_dataloader(),
                val_dataloaders=data_module.get_val_dataloader())
    
    # Test model
    test_results = trainer.test(model, 
                               dataloaders=data_module.get_test_dataloader(),
                               ckpt_path="best")
    
    return {
        'test_results': test_results[0] if test_results else {},
        'best_model_path': checkpoint_callback.best_model_path,
        'last_model_path': checkpoint_callback.last_model_path
    }


def run_active_learning_experiment(
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Run complete active learning experiment"""
    
    # Initialize experiment tracking
    experiment_results = {
        'strategy': args.al_strategy,
        'iterations': [],
        'model_paths': [],
        'test_performances': [],
        'labeled_counts': [],
        'selection_info': []
    }
    
    # Setup experiment directory
    exp_dir = Path(args.output_dir) / f"ctc_seg_{args.al_strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize CSV metrics tracker
    metrics_tracker = MetricsTracker(
        output_dir=exp_dir,
        experiment_name=f"{args.al_strategy}_experiment"
    )
    logger.info(f"CSV metrics will be saved to: {exp_dir}")
    
    # Initialize labeled/unlabeled splits
    labeled_mask = np.zeros(len(train_df), dtype=bool)
    init_indices = np.random.choice(len(train_df), size=args.initial_samples, replace=False)
    labeled_mask[init_indices] = True
    
    logger.info(f"Starting AL experiment with {args.initial_samples} initial samples")
    
    # Active learning loop
    for iteration in range(args.al_iterations):
        iter_start_time = time.time()  # Track iteration timing
        iter_dir = exp_dir / f"iter_{iteration}"
        iter_dir.mkdir(exist_ok=True)
        
        logger.info(f"=== AL Iteration {iteration + 1}/{args.al_iterations} ===")
        logger.info(f"Labeled samples: {labeled_mask.sum()}")
        
        # Create data module for current labeled set
        current_train_df = train_df[labeled_mask].reset_index(drop=True)
        
        data_module = CTCSegmentationDataModule(
            train_df=current_train_df,
            val_df=val_df,
            test_df=test_df,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_type="binary",
            image_size=(args.image_size, args.image_size),
            include_square_mask=True,
            square_size_range=(5, 50)
        )
        
        # Create model with appropriate number of input channels
        n_input_channels = 2  # Image + square mask channel
        model = create_ctc_segmentation_model(
            n_channels=n_input_channels,
            n_classes=1,
            learning_rate=args.learning_rate,
            loss_function=args.loss_function,
            base_channels=args.base_channels
        )
        
        # Setup logger
        logger_pl = TensorBoardLogger(
            save_dir=iter_dir,
            name="lightning_logs",
            version=f"iter_{iteration}"
        )
        
        # Train model
        train_results = train_model(
            model=model,
            data_module=data_module,
            max_epochs=args.max_epochs,
            checkpoint_dir=iter_dir / "checkpoints",
            logger_pl=logger_pl,
            metrics_tracker=metrics_tracker,
            iteration=iteration,
            debug=args.debug
        )
        
        # Record results
        experiment_results['iterations'].append(iteration)
        experiment_results['model_paths'].append(train_results['best_model_path'])
        experiment_results['test_performances'].append(train_results['test_results'])
        experiment_results['labeled_counts'].append(labeled_mask.sum())
        
        # Log iteration-level metrics to CSV
        iteration_metrics = {
            'strategy': args.al_strategy,
            'labeled_samples': int(labeled_mask.sum()),
            'unlabeled_samples': int((~labeled_mask).sum()),
            'test_iou': train_results['test_results'].get('test_iou', 0.0),
            'test_loss': train_results['test_results'].get('test_loss', 0.0),
            'best_model_path': str(train_results['best_model_path']),
            'training_duration_seconds': time.time() - iter_start_time
        }
        
        # Add any additional test metrics
        for key, value in train_results['test_results'].items():
            if key not in iteration_metrics and isinstance(value, (int, float)):
                iteration_metrics[key] = float(value)
        
        metrics_tracker.log_iteration_metrics(iteration, iteration_metrics)
        
        logger.info(f"Iteration {iteration} - Test IoU: {train_results['test_results'].get('test_iou', 'N/A'):.4f}")
        logger.info(f"CSV metrics saved: {exp_dir}/{args.al_strategy}_experiment_iteration_metrics.csv")
        
        # Select new samples (except for last iteration)
        if iteration < args.al_iterations - 1:
            unlabeled_indices = np.where(~labeled_mask)[0]
            
            if len(unlabeled_indices) == 0:
                logger.warning("No more unlabeled samples available")
                break
            
            # Load best model for uncertainty estimation
            if args.al_strategy == "uncertainty":
                best_model = CTCSegmentationModel.load_from_checkpoint(train_results['best_model_path'])
                # Move model to GPU and set to eval mode
                if torch.cuda.is_available():
                    best_model = best_model.cuda()
                best_model.eval()
            else:
                best_model = None
            
            selected_local_idx = select_samples_for_al(
                train_df=train_df,
                labeled_indices=np.where(labeled_mask)[0],
                unlabeled_indices=unlabeled_indices,
                model=best_model,
                strategy=args.al_strategy,
                quota=args.samples_per_iteration
            )
            
            # Convert to global indices and update labeled mask
            selected_global_idx = unlabeled_indices[selected_local_idx]
            labeled_mask[selected_global_idx] = True
            
            # Record selection info
            selection_info = {
                'iteration': iteration,
                'selected_indices': selected_global_idx.tolist(),
                'strategy': args.al_strategy
            }
            experiment_results['selection_info'].append(selection_info)
            
            # Log sample selection to CSV
            selection_metrics = {
                'selection_method': args.al_strategy,
                'samples_added': len(selected_global_idx),
                'total_unlabeled_before': len(unlabeled_indices),
                'total_labeled_after': int(labeled_mask.sum())
            }
            
            metrics_tracker.log_sample_selection(
                iteration=iteration,
                strategy=args.al_strategy,
                selected_indices=selected_global_idx.tolist(),
                selection_metrics=selection_metrics
            )
            
            logger.info(f"Selected {len(selected_global_idx)} new samples")
    
    # Save experiment results
    results_file = exp_dir / "experiment_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in experiment_results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value
        json.dump(serializable_results, f, indent=2, default=str)
    
    # Generate and save experiment summary
    summary_stats = metrics_tracker.get_summary_stats()
    summary_file = exp_dir / f"{args.al_strategy}_experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    logger.info(f"Experiment completed. Results saved to {results_file}")
    logger.info(f"CSV metrics summary: {summary_file}")
    logger.info(f"Detailed CSV files:")
    logger.info(f"  - Epoch metrics: {metrics_tracker.epoch_metrics_file}")
    logger.info(f"  - Iteration metrics: {metrics_tracker.iteration_metrics_file}")
    logger.info(f"  - Sample selection: {metrics_tracker.selection_log_file}")
    
    return experiment_results


def main():
    parser = argparse.ArgumentParser(description="CTC Single-Cell Segmentation with Active Learning")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing CTC single-cell data")
    parser.add_argument("--output_dir", type=str, default="./ctc_seg_experiments",
                       help="Output directory for experiments")
    
    # Model arguments
    parser.add_argument("--base_channels", type=int, default=64,
                       help="Base number of channels in U-Net")
    parser.add_argument("--image_size", type=int, default=256,
                       help="Input image size (square)")
    parser.add_argument("--loss_function", type=str, default="combined",
                       choices=["bce", "dice", "combined", "mse"],
                       help="Loss function to use")
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=50,
                       help="Maximum training epochs")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    
    # Active learning arguments
    parser.add_argument("--al_strategy", type=str, default="time_interval",
                       choices=["random", "time_interval", "uncertainty"],
                       help="Active learning strategy")
    parser.add_argument("--al_iterations", type=int, default=5,
                       help="Number of AL iterations")
    parser.add_argument("--initial_samples", type=int, default=100,
                       help="Initial number of labeled samples")
    parser.add_argument("--samples_per_iteration", type=int, default=50,
                       help="Samples to add per AL iteration")
    
    # Debug arguments
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with reduced data")
    parser.add_argument("--subset_ratio", type=float, default=None,
                       help="Use subset of data for testing (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    logger = setup_logging(Path(args.output_dir), args.debug)
    
    logger.info("Starting CTC Segmentation Active Learning Experiment")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Load and prepare data
        logger.info("Loading CTC data...")
        df = load_ctc_data(args.data_dir, args.subset_ratio, logger)
        df = validate_data_paths(df, logger)
        
        # Create train/val/test splits
        train_df, val_df, test_df = create_data_splits(df)
        
        # Run active learning experiment
        results = run_active_learning_experiment(args, train_df, val_df, test_df, logger)
        
        logger.info("Experiment completed successfully!")
        logger.info(f"Final labeled samples: {results['labeled_counts'][-1]}")
        logger.info(f"Final test performance: {results['test_performances'][-1]}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()