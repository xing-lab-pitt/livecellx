import importlib.util
import types
import sys
from pathlib import Path
import numpy as np

# Provide a minimal stub for livecellx.preprocess.utils with normalize_img_to_uint8
prep_module = types.ModuleType("livecellx.preprocess.utils")

def normalize_img_to_uint8(img: np.ndarray, dtype=np.uint8) -> np.ndarray:
    std = np.std(img.flatten())
    if std != 0:
        img = (img - np.mean(img.flatten())) / std
    else:
        img = img - np.mean(img.flatten())
    img = img + abs(np.min(img.flatten()))
    if np.max(img) != 0:
        img = img / np.max(img) * 255
    return img.astype(dtype)

def label_mask_to_edt_mask(mask):
    # Minimal stub
    return mask

prep_module.normalize_img_to_uint8 = normalize_img_to_uint8
prep_module.label_mask_to_edt_mask = label_mask_to_edt_mask
sys.modules.setdefault("livecellx", types.ModuleType("livecellx"))
sys.modules.setdefault("livecellx.preprocess", types.ModuleType("livecellx.preprocess"))
sys.modules["livecellx.preprocess.utils"] = prep_module

# Load the core utils module directly
utils_path = Path(__file__).resolve().parents[1] / "livecellx" / "core" / "utils.py"
utils_spec = importlib.util.spec_from_file_location("core_utils", utils_path)
core_utils = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(core_utils)

crop_or_pad_img = core_utils.crop_or_pad_img


def test_pad_odd_difference():
    img = np.zeros((64, 64), dtype=np.uint8)
    result = crop_or_pad_img(img.copy(), (65, 65))
    assert result.shape == (65, 65)


def test_crop_odd_difference():
    img = np.zeros((64, 64), dtype=np.uint8)
    result = crop_or_pad_img(img.copy(), (63, 63))
    assert result.shape == (63, 63)


def test_padding_distribution():
    """Test that padding is distributed correctly for odd differences"""
    # Test padding from (10, 10) to (13, 13) - difference of 3
    img = np.ones((10, 10), dtype=np.uint8)
    result = crop_or_pad_img(img.copy(), (13, 13))
    
    # Should have 1 pad before, 2 pad after (or vice versa)
    # Verify all original content is preserved
    assert result.shape == (13, 13)
    # Check that the center region contains the original data
    center_start_row = (13 - 10) // 2  # 1
    center_end_row = center_start_row + 10  # 11
    center_start_col = (13 - 10) // 2  # 1
    center_end_col = center_start_col + 10  # 11
    
    # Original data should be in the center
    assert np.all(result[center_start_row:center_end_row, center_start_col:center_end_col] == 1)
    
    # Padded regions should be zeros
    assert np.all(result[:center_start_row, :] == 0)  # top padding
    assert np.all(result[center_end_row:, :] == 0)    # bottom padding
    assert np.all(result[:, :center_start_col] == 0)  # left padding
    assert np.all(result[:, center_end_col:] == 0)    # right padding


def test_padding_even_difference():
    """Test padding with even differences"""
    img = np.ones((10, 10), dtype=np.uint8)
    result = crop_or_pad_img(img.copy(), (14, 14))
    
    assert result.shape == (14, 14)
    # For even difference (4), should be 2 padding on each side
    assert np.all(result[2:12, 2:12] == 1)  # original data
    assert np.all(result[:2, :] == 0)       # top padding
    assert np.all(result[12:, :] == 0)      # bottom padding
    assert np.all(result[:, :2] == 0)       # left padding
    assert np.all(result[:, 12:] == 0)      # right padding


def test_asymmetric_padding():
    """Test padding with different dimensions"""
    img = np.ones((5, 8), dtype=np.uint8)
    result = crop_or_pad_img(img.copy(), (7, 10))
    
    assert result.shape == (7, 10)
    # Height: 5->7 (diff=2, pad 1 each side)
    # Width: 8->10 (diff=2, pad 1 each side)
    assert np.all(result[1:6, 1:9] == 1)  # original data
    assert np.all(result[0, :] == 0)      # top padding
    assert np.all(result[6, :] == 0)      # bottom padding
    assert np.all(result[:, 0] == 0)      # left padding
    assert np.all(result[:, 9] == 0)      # right padding
