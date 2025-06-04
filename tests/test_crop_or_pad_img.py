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

prep_module.normalize_img_to_uint8 = normalize_img_to_uint8
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
