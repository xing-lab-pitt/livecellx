import numpy as np
import pytest
from livecellx.core.utils import crop_or_pad_img


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