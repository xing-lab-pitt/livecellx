#!/usr/bin/env python3
"""
CTC Segmentation Dataset

Dataset classes for loading and preprocessing Cell Tracking Challenge (CTC) data
for single-cell segmentation tasks. Supports various data formats and augmentations.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Callable, Tuple, Union, List, Dict, Any
import logging
from scipy.ndimage import distance_transform_edt
from skimage import measure, morphology
import cv2
import random

logger = logging.getLogger(__name__)


class CTCSegmentationDataset(Dataset):
    """
    Dataset for CTC single-cell segmentation
    
    Loads single-cell crops and their corresponding segmentation masks
    from CTC datasets for active learning segmentation tasks.
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        target_type: str = "binary",  # "binary", "distance", "both"
        padding: int = 10,
        image_size: Tuple[int, int] = (256, 256),
        normalize: bool = True,
        cache_data: bool = False,
        include_square_mask: bool = True,
        square_size_range: Tuple[int, int] = (5, 50)
    ):
        """
        Args:
            dataframe: DataFrame with columns ['raw_img_path', 'mask_path', 'source', ...]
            transform: Optional transform for input images
            target_transform: Optional transform for target masks
            target_type: Type of target ("binary", "distance", "both")
            padding: Padding around single cells
            image_size: Target image size (H, W)
            normalize: Whether to normalize images to [0, 1]
            cache_data: Whether to cache loaded data in memory
            include_square_mask: Whether to include square mask as additional input channel
            square_size_range: Range of square sizes (min, max) in pixels
        """
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.target_transform = target_transform
        self.target_type = target_type
        self.padding = padding
        self.image_size = image_size
        self.normalize = normalize
        self.cache_data = cache_data
        self.include_square_mask = include_square_mask
        self.square_size_range = square_size_range
        
        if cache_data:
            self.cache = {}
        
        logger.info(f"Created CTCSegmentationDataset with {len(self.df)} samples")
        logger.info(f"Target type: {target_type}, Image size: {image_size}")
        if include_square_mask:
            logger.info(f"Including square mask channel with size range: {square_size_range}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _generate_square_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Generate a square mask at a random point inside the cell
        
        Args:
            mask: Binary mask of the cell (0 for background, >0 for cell)
            
        Returns:
            square_mask: Binary mask with a square (0 for background, 1 for square)
        """
        # Find all foreground pixels (inside the cell)
        foreground_pixels = np.where(mask > 0)
        
        if len(foreground_pixels[0]) == 0:
            # No foreground pixels, return empty mask
            return np.zeros_like(mask, dtype=np.uint8)
        
        # Randomly choose a point inside the cell
        pixel_idx = random.randint(0, len(foreground_pixels[0]) - 1)
        center_y = foreground_pixels[0][pixel_idx]
        center_x = foreground_pixels[1][pixel_idx]
        
        # Randomly choose square size
        square_size = random.randint(self.square_size_range[0], self.square_size_range[1])
        
        # Create square mask
        square_mask = np.zeros_like(mask, dtype=np.uint8)
        
        # Calculate square boundaries
        half_size = square_size // 2
        y_start = max(0, center_y - half_size)
        y_end = min(mask.shape[0], center_y + half_size + 1)
        x_start = max(0, center_x - half_size)
        x_end = min(mask.shape[1], center_x + half_size + 1)
        
        # Fill the square
        square_mask[y_start:y_end, x_start:x_end] = 1
        
        return square_mask
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample"""
        if self.cache_data and idx in self.cache:
            return self.cache[idx]
        
        # Get sample info
        sample_info = self.df.iloc[idx]
        
        try:
            # Load raw image
            raw_img_path = sample_info['raw_img_path']
            raw_img = self._load_image(raw_img_path)
            
            # Load mask
            mask_path = sample_info['mask_path']
            mask = self._load_mask(mask_path)
            
            # Ensure images are the same size
            if raw_img.shape != mask.shape:
                mask = cv2.resize(mask, (raw_img.shape[1], raw_img.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            
            # Resize to target size
            if raw_img.shape[:2] != self.image_size:
                raw_img = cv2.resize(raw_img, self.image_size, interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
            
            # Normalize image
            if self.normalize:
                raw_img = self._normalize_image(raw_img)
            
            # Generate square mask if requested
            if self.include_square_mask:
                square_mask = self._generate_square_mask(mask)
                # Resize square mask to match image size
                if square_mask.shape[:2] != self.image_size:
                    square_mask = cv2.resize(square_mask, self.image_size, interpolation=cv2.INTER_NEAREST)
                
                # Stack original image and square mask as separate channels
                if len(raw_img.shape) == 2:
                    # Both are 2D, stack them
                    combined_image = np.stack([raw_img, square_mask.astype(np.float32)], axis=0)
                else:
                    # Raw image might already have a channel dimension
                    if len(raw_img.shape) == 3 and raw_img.shape[0] == 1:
                        raw_img = raw_img[0]  # Remove channel dimension
                    combined_image = np.stack([raw_img, square_mask.astype(np.float32)], axis=0)
                
                image_tensor = torch.from_numpy(combined_image).float()
            else:
                # Convert to tensors (original behavior)
                image_tensor = torch.from_numpy(raw_img).float()
                if len(image_tensor.shape) == 2:
                    image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension
            
            # Process target based on target_type
            target = self._process_target(mask)
            
            # Apply transforms
            if self.transform:
                image_tensor = self.transform(image_tensor)
            
            if self.target_transform:
                target = self.target_transform(target)
            
            sample = {
                'image': image_tensor,
                'target': target,
                'image_path': str(raw_img_path),
                'mask_path': str(mask_path),
                'source': sample_info.get('source', 'unknown'),
                'idx': idx
            }
            
            # Add additional metadata if available
            for col in ['timepoint', 'cell_id', 'dataset_name']:
                if col in sample_info:
                    sample[col] = sample_info[col]
            
            if self.cache_data:
                self.cache[idx] = sample
            
            return sample
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return a dummy sample to avoid breaking the training
            return self._get_dummy_sample(idx)
    
    def _load_image(self, path: Union[str, Path]) -> np.ndarray:
        """Load and preprocess raw image"""
        path = Path(path)
        
        if path.suffix.lower() in ['.tif', '.tiff']:
            # Use PIL for TIFF files
            with Image.open(path) as img:
                image = np.array(img)
        else:
            # Use PIL for other formats
            with Image.open(path) as img:
                image = np.array(img.convert('L'))  # Convert to grayscale
        
        return image.astype(np.float32)
    
    def _load_mask(self, path: Union[str, Path]) -> np.ndarray:
        """Load segmentation mask"""
        path = Path(path)
        
        if path.suffix.lower() in ['.tif', '.tiff']:
            with Image.open(path) as img:
                mask = np.array(img)
        else:
            with Image.open(path) as img:
                mask = np.array(img.convert('L'))
        
        # Ensure binary mask
        mask = (mask > 0).astype(np.uint8)
        return mask
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range"""
        if image.max() > image.min():
            return (image - image.min()) / (image.max() - image.min())
        else:
            return np.zeros_like(image)
    
    def _process_target(self, mask: np.ndarray) -> torch.Tensor:
        """Process target mask based on target_type"""
        if self.target_type == "binary":
            target = torch.from_numpy(mask.astype(np.float32))
            if len(target.shape) == 2:
                target = target.unsqueeze(0)  # Add channel dimension
            return target
        
        elif self.target_type == "distance":
            # Create distance transform
            dist_transform = distance_transform_edt(mask > 0)
            # Normalize distance transform
            if dist_transform.max() > 0:
                dist_transform = dist_transform / dist_transform.max()
            
            target = torch.from_numpy(dist_transform.astype(np.float32))
            if len(target.shape) == 2:
                target = target.unsqueeze(0)
            return target
        
        elif self.target_type == "both":
            # Return both binary and distance
            binary_mask = torch.from_numpy(mask.astype(np.float32))
            dist_transform = distance_transform_edt(mask > 0)
            if dist_transform.max() > 0:
                dist_transform = dist_transform / dist_transform.max()
            dist_mask = torch.from_numpy(dist_transform.astype(np.float32))
            
            # Stack along channel dimension
            target = torch.stack([binary_mask, dist_mask], dim=0)
            return target
        
        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")
    
    def _get_dummy_sample(self, idx: int) -> Dict[str, Any]:
        """Create a dummy sample for error cases"""
        # Determine number of input channels
        n_channels = 2 if self.include_square_mask else 1
        dummy_image = torch.zeros(n_channels, *self.image_size, dtype=torch.float32)
        
        if self.target_type == "both":
            dummy_target = torch.zeros(2, *self.image_size, dtype=torch.float32)
        else:
            dummy_target = torch.zeros(1, *self.image_size, dtype=torch.float32)
        
        return {
            'image': dummy_image,
            'target': dummy_target,
            'image_path': 'dummy',
            'mask_path': 'dummy',
            'source': 'dummy',
            'idx': idx
        }
    
    def get_sample_info(self, idx: int) -> pd.Series:
        """Get metadata for a specific sample"""
        return self.df.iloc[idx]
    
    def get_source_distribution(self) -> Dict[str, int]:
        """Get distribution of samples by source"""
        if 'source' in self.df.columns:
            return self.df['source'].value_counts().to_dict()
        else:
            return {'unknown': len(self.df)}


class CTCSegmentationDataModule:
    """
    Data module for CTC segmentation datasets with train/val/test splits
    """
    
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        batch_size: int = 16,
        num_workers: int = 4,
        target_type: str = "binary",
        image_size: Tuple[int, int] = (256, 256),
        augment_training: bool = True,
        cache_data: bool = False,
        include_square_mask: bool = True,
        square_size_range: Tuple[int, int] = (5, 50)
    ):
        """
        Args:
            train_df: Training data DataFrame
            val_df: Validation data DataFrame
            test_df: Test data DataFrame
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            target_type: Type of target ("binary", "distance", "both")
            image_size: Target image size
            augment_training: Whether to apply data augmentation to training data
            cache_data: Whether to cache data in memory
            include_square_mask: Whether to include square mask as additional input channel
            square_size_range: Range of square sizes (min, max) in pixels
        """
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_type = target_type
        self.image_size = image_size
        self.augment_training = augment_training
        self.cache_data = cache_data
        self.include_square_mask = include_square_mask
        self.square_size_range = square_size_range
        
        # Setup transforms
        self.train_transform = self._get_train_transforms() if augment_training else None
        self.val_transform = None  # No augmentation for validation/test
        
        logger.info(f"Created CTCSegmentationDataModule")
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df) if val_df is not None else 0}, "
                   f"Test: {len(test_df) if test_df is not None else 0}")
    
    def _get_train_transforms(self) -> transforms.Compose:
        """Get training data augmentation transforms"""
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            # Add more augmentations as needed
        ])
    
    def get_train_dataloader(self) -> DataLoader:
        """Get training data loader"""
        dataset = CTCSegmentationDataset(
            dataframe=self.train_df,
            transform=self.train_transform,
            target_type=self.target_type,
            image_size=self.image_size,
            cache_data=self.cache_data,
            include_square_mask=self.include_square_mask,
            square_size_range=self.square_size_range
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def get_val_dataloader(self) -> Optional[DataLoader]:
        """Get validation data loader"""
        if self.val_df is None:
            return None
        
        dataset = CTCSegmentationDataset(
            dataframe=self.val_df,
            transform=self.val_transform,
            target_type=self.target_type,
            image_size=self.image_size,
            cache_data=self.cache_data,
            include_square_mask=self.include_square_mask,
            square_size_range=self.square_size_range
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_test_dataloader(self) -> Optional[DataLoader]:
        """Get test data loader"""
        if self.test_df is None:
            return None
        
        dataset = CTCSegmentationDataset(
            dataframe=self.test_df,
            transform=self.val_transform,
            target_type=self.target_type,
            image_size=self.image_size,
            cache_data=self.cache_data,
            include_square_mask=self.include_square_mask,
            square_size_range=self.square_size_range
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the datasets"""
        stats = {
            'train_size': len(self.train_df),
            'val_size': len(self.val_df) if self.val_df is not None else 0,
            'test_size': len(self.test_df) if self.test_df is not None else 0,
            'total_size': len(self.train_df) + (len(self.val_df) if self.val_df is not None else 0) + 
                         (len(self.test_df) if self.test_df is not None else 0)
        }
        
        # Add source distribution
        if 'source' in self.train_df.columns:
            stats['train_source_dist'] = self.train_df['source'].value_counts().to_dict()
        
        return stats


# Utility functions
def create_ctc_dataloader(
    dataframe: pd.DataFrame,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    Convenience function to create a CTC segmentation data loader
    
    Args:
        dataframe: DataFrame with sample information
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        **kwargs: Additional arguments for CTCSegmentationDataset
    
    Returns:
        DataLoader instance
    """
    dataset = CTCSegmentationDataset(dataframe, **kwargs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle  # Drop last batch only when shuffling (training)
    )


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for handling variable-sized metadata
    """
    # Separate image/target tensors from metadata
    images = torch.stack([sample['image'] for sample in batch])
    targets = torch.stack([sample['target'] for sample in batch])
    
    # Collect metadata
    metadata = {}
    for key in ['image_path', 'mask_path', 'source', 'idx']:
        if key in batch[0]:
            metadata[key] = [sample[key] for sample in batch]
    
    return {
        'image': images,
        'target': targets,
        **metadata
    }


# Example usage and testing
if __name__ == "__main__":
    # Test with dummy data
    dummy_df = pd.DataFrame({
        'raw_img_path': ['test1.tif', 'test2.tif'],
        'mask_path': ['mask1.tif', 'mask2.tif'],
        'source': ['dataset1', 'dataset2']
    })
    
    try:
        dataset = CTCSegmentationDataset(
            dataframe=dummy_df,
            target_type="binary",
            image_size=(256, 256)
        )
        
        print(f"Created dataset with {len(dataset)} samples")
        print(f"Source distribution: {dataset.get_source_distribution()}")
        
        # Test data module
        data_module = CTCSegmentationDataModule(
            train_df=dummy_df,
            batch_size=2,
            num_workers=0  # For testing
        )
        
        stats = data_module.get_dataset_stats()
        print(f"Data module stats: {stats}")
        
    except Exception as e:
        print(f"Test failed (expected with dummy data): {e}")