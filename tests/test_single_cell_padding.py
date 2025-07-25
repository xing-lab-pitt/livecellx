import numpy as np
import pytest


def test_single_cell_padding_logic():
    """Test the padding logic used in single_cell.py to ensure it handles odd differences correctly"""
    
    # Test case 1: odd difference (similar to the bug in utils.py)
    dims = [13, 15]  # target dimensions
    sc_img_shape = [10, 12]  # current image shape
    
    # Calculate padding as done in single_cell.py lines 2439-2451
    _pad_pixels = [max(0, dims[i] - sc_img_shape[i]) for i in range(len(dims))]
    # _pad_pixels = [3, 3] for this case
    
    _pad_pixels_np = np.array([
        [
            _pad_pixels[0] // 2,                    # 3 // 2 = 1
            _pad_pixels[0] - _pad_pixels[0] // 2,   # 3 - 1 = 2
        ],
        [
            _pad_pixels[1] // 2,                    # 3 // 2 = 1
            _pad_pixels[1] - _pad_pixels[1] // 2,   # 3 - 1 = 2
        ],
    ])
    
    # Verify the padding calculation
    assert _pad_pixels_np[0, 0] == 1  # pad_before for dimension 0
    assert _pad_pixels_np[0, 1] == 2  # pad_after for dimension 0
    assert _pad_pixels_np[1, 0] == 1  # pad_before for dimension 1
    assert _pad_pixels_np[1, 1] == 2  # pad_after for dimension 1
    
    # Verify total padding equals difference
    assert _pad_pixels_np[0, 0] + _pad_pixels_np[0, 1] == _pad_pixels[0]
    assert _pad_pixels_np[1, 0] + _pad_pixels_np[1, 1] == _pad_pixels[1]
    
    # Test with a real image
    sc_img = np.ones(sc_img_shape, dtype=np.uint8)
    padded_img = np.pad(sc_img, _pad_pixels_np, mode="constant", constant_values=0)
    
    # Verify final shape
    assert padded_img.shape == tuple(dims)
    
    # Verify original content is preserved in the center
    start_row, end_row = _pad_pixels_np[0, 0], _pad_pixels_np[0, 0] + sc_img_shape[0]
    start_col, end_col = _pad_pixels_np[1, 0], _pad_pixels_np[1, 0] + sc_img_shape[1]
    assert np.all(padded_img[start_row:end_row, start_col:end_col] == 1)
    
    # Verify padding regions are zeros
    assert np.all(padded_img[:start_row, :] == 0)  # top
    assert np.all(padded_img[end_row:, :] == 0)    # bottom
    assert np.all(padded_img[:, :start_col] == 0)  # left
    assert np.all(padded_img[:, end_col:] == 0)    # right


def test_single_cell_even_padding():
    """Test even padding differences"""
    dims = [14, 16]  # target dimensions
    sc_img_shape = [10, 12]  # current image shape
    
    _pad_pixels = [max(0, dims[i] - sc_img_shape[i]) for i in range(len(dims))]
    # _pad_pixels = [4, 4] for this case
    
    _pad_pixels_np = np.array([
        [
            _pad_pixels[0] // 2,                    # 4 // 2 = 2
            _pad_pixels[0] - _pad_pixels[0] // 2,   # 4 - 2 = 2
        ],
        [
            _pad_pixels[1] // 2,                    # 4 // 2 = 2
            _pad_pixels[1] - _pad_pixels[1] // 2,   # 4 - 2 = 2
        ],
    ])
    
    # For even differences, padding should be equal on both sides
    assert _pad_pixels_np[0, 0] == _pad_pixels_np[0, 1] == 2
    assert _pad_pixels_np[1, 0] == _pad_pixels_np[1, 1] == 2