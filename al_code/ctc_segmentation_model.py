#!/usr/bin/env python3
"""
CTC Single-Cell Segmentation Model

A comprehensive PyTorch implementation using U-Net architecture for single-cell segmentation
on Cell Tracking Challenge (CTC) datasets. Designed for active learning experiments.

Features:
- U-Net architecture with skip connections
- PyTorch Lightning integration for easy training
- Support for both binary and distance transform targets
- Comprehensive metrics tracking (IoU, Dice, pixel accuracy)
- Active learning compatible with uncertainty estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torchmetrics
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """Double Convolution block: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downscaling block: MaxPool -> DoubleConv"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upscaling block: Upsample -> DoubleConv"""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution layer"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for single-cell segmentation
    
    Args:
        n_channels: Number of input channels (typically 1 for grayscale)
        n_classes: Number of output classes (1 for binary segmentation, 2 for background/foreground)
        bilinear: Whether to use bilinear upsampling (True) or transposed convolutions (False)
        base_channels: Base number of channels (default: 64)
    """
    
    def __init__(self, n_channels: int = 1, n_classes: int = 1, bilinear: bool = True, base_channels: int = 64):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(base_channels * 8, base_channels * 16 // factor)
        
        self.up1 = UpBlock(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = UpBlock(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = UpBlock(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = UpBlock(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, n_classes)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits


class CTCSegmentationModel(pl.LightningModule):
    """
    PyTorch Lightning wrapper for CTC single-cell segmentation using U-Net
    
    Features:
    - Flexible loss functions (BCE, Dice, Combined)
    - Comprehensive metrics tracking
    - Uncertainty estimation for active learning
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 1,
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
        loss_function: str = "bce",
        weight_decay: float = 1e-4,
        scheduler: str = "step",
        scheduler_params: Optional[Dict[str, Any]] = None,
        base_channels: int = 64,
        bilinear: bool = True,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model architecture
        self.model = UNet(
            n_channels=n_channels,
            n_classes=n_classes,
            bilinear=bilinear,
            base_channels=base_channels
        )
        
        # Loss function
        self.loss_function = self._get_loss_function(loss_function)
        
        # Metrics
        self._setup_metrics()
        
        # Training parameters
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler
        self.scheduler_params = scheduler_params or {}
        
        logger.info(f"Initialized CTC Segmentation Model with {self._count_parameters()} parameters")
    
    def _get_loss_function(self, loss_name: str):
        """Get loss function by name"""
        if loss_name == "bce":
            return nn.BCEWithLogitsLoss()
        elif loss_name == "dice":
            return self._dice_loss
        elif loss_name == "combined":
            return self._combined_loss
        elif loss_name == "mse":
            return nn.MSELoss()  # For distance transform targets
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
    
    def _setup_metrics(self):
        """Setup torchmetrics for comprehensive evaluation"""
        # Binary segmentation metrics
        self.train_iou = torchmetrics.JaccardIndex(task="binary", num_classes=2)
        self.val_iou = torchmetrics.JaccardIndex(task="binary", num_classes=2)
        self.test_iou = torchmetrics.JaccardIndex(task="binary", num_classes=2)
        
        self.train_dice = torchmetrics.F1Score(task="binary")
        self.val_dice = torchmetrics.F1Score(task="binary")
        self.test_dice = torchmetrics.F1Score(task="binary")
        
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")
    
    def _count_parameters(self) -> int:
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """Dice loss for segmentation"""
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return 1 - dice
    
    def _combined_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Combined BCE + Dice loss"""
        bce = F.binary_cross_entropy_with_logits(pred, target)
        dice = self._dice_loss(pred, target)
        return 0.5 * bce + 0.5 * dice
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.model(x)
    
    def _shared_step(self, batch, stage: str):
        """Shared step for training/validation/test"""
        if isinstance(batch, dict):
            x = batch['image']
            y = batch['target']
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
        else:
            raise ValueError(f"Expected batch to be dict or (x, y), got {type(batch)}")
        
        # Forward pass
        pred = self(x)
        
        # Ensure target has same shape as prediction
        if y.dim() == 3:  # Add channel dimension if missing
            y = y.unsqueeze(1)
        
        # Calculate loss
        loss = self.loss_function(pred, y.float())
        
        # Convert predictions to probabilities for metrics
        pred_probs = torch.sigmoid(pred)
        pred_binary = (pred_probs > 0.5).float()
        
        # Update metrics
        if stage == "train":
            self.train_iou(pred_binary, y.int())
            self.train_dice(pred_binary, y.int())
            self.train_acc(pred_binary, y.int())
        elif stage == "val":
            self.val_iou(pred_binary, y.int())
            self.val_dice(pred_binary, y.int())
            self.val_acc(pred_binary, y.int())
        elif stage == "test":
            self.test_iou(pred_binary, y.int())
            self.test_dice(pred_binary, y.int())
            self.test_acc(pred_binary, y.int())
        
        return {
            'loss': loss,
            'pred': pred_probs,
            'target': y,
            'pred_binary': pred_binary
        }
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        result = self._shared_step(batch, "train")
        
        # Log metrics
        self.log('train_loss', result['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_iou', self.train_iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_dice', self.train_dice, on_step=False, on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        
        return result['loss']
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        result = self._shared_step(batch, "val")
        
        # Log metrics
        self.log('val_loss', result['loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', self.val_iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_dice', self.val_dice, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)
        
        return result['loss']
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        result = self._shared_step(batch, "test")
        
        # Log metrics
        self.log('test_loss', result['loss'], on_step=False, on_epoch=True)
        self.log('test_iou', self.test_iou, on_step=False, on_epoch=True)
        self.log('test_dice', self.test_dice, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        
        return result['loss']
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step for inference"""
        if isinstance(batch, dict):
            x = batch['image']
        elif isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        
        pred = self(x)
        pred_probs = torch.sigmoid(pred)
        pred_binary = (pred_probs > 0.5).float()
        
        return {
            'probabilities': pred_probs,
            'predictions': pred_binary,
            'logits': pred
        }
    
    def get_uncertainty(self, x: torch.Tensor, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictive uncertainty using Monte Carlo dropout
        
        Args:
            x: Input tensor
            n_samples: Number of MC samples
            
        Returns:
            mean_pred: Mean prediction
            uncertainty: Predictive uncertainty (variance)
        """
        # Ensure input is on the same device as the model
        device = next(self.parameters()).device
        x = x.to(device)
        
        self.train()  # Enable dropout
        
        with torch.no_grad():
            predictions = []
            for _ in range(n_samples):
                pred = torch.sigmoid(self(x))
                predictions.append(pred)
            
            predictions = torch.stack(predictions)
            mean_pred = predictions.mean(dim=0)
            uncertainty = predictions.var(dim=0)
        
        self.eval()  # Back to eval mode
        return mean_pred, uncertainty
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        # Optimizer
        if self.optimizer_name.lower() == "adam":
            optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name.lower() == "sgd":
            optimizer = SGD(self.parameters(), lr=self.learning_rate, 
                          momentum=0.9, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
        
        # Learning rate scheduler
        if self.scheduler_name.lower() == "step":
            scheduler = StepLR(optimizer, 
                             step_size=self.scheduler_params.get('step_size', 30),
                             gamma=self.scheduler_params.get('gamma', 0.1))
        elif self.scheduler_name.lower() == "cosine":
            scheduler = CosineAnnealingLR(optimizer,
                                        T_max=self.scheduler_params.get('T_max', 100),
                                        eta_min=self.scheduler_params.get('eta_min', 1e-6))
        else:
            return optimizer
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }


# Convenience function for model creation
def create_ctc_segmentation_model(
    model_type: str = "unet",
    n_channels: int = 1,
    n_classes: int = 1,
    **kwargs
) -> CTCSegmentationModel:
    """
    Create a CTC segmentation model
    
    Args:
        model_type: Type of model architecture ("unet")
        n_channels: Number of input channels
        n_classes: Number of output classes
        **kwargs: Additional arguments for the model
    
    Returns:
        CTCSegmentationModel instance
    """
    if model_type.lower() == "unet":
        return CTCSegmentationModel(
            n_channels=n_channels,
            n_classes=n_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test model creation
    model = create_ctc_segmentation_model(
        model_type="unet",
        n_channels=1,
        n_classes=1,
        learning_rate=1e-3,
        loss_function="combined"
    )
    
    # Test forward pass
    x = torch.randn(2, 1, 256, 256)  # Batch of 2 images
    y = torch.randint(0, 2, (2, 1, 256, 256)).float()  # Binary masks
    
    print(f"Model created with {model._count_parameters()} parameters")
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    pred = model(x)
    print(f"Output shape: {pred.shape}")
    
    # Test uncertainty estimation
    mean_pred, uncertainty = model.get_uncertainty(x, n_samples=5)
    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"Mean uncertainty: {uncertainty.mean().item():.4f}")