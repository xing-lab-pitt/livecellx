#!/usr/bin/env python3
"""
Installation Verification Script for Active Learning Single-Cell Segmentation
Verifies all required dependencies are properly installed and compatible.
"""

import sys
import importlib
from typing import List, Tuple

def check_version(package_name: str, expected_version: str = None) -> Tuple[bool, str]:
    """Check if a package is installed and optionally verify version."""
    try:
        module = importlib.import_module(package_name.replace('-', '_'))
        version = getattr(module, '__version__', 'unknown')
        
        # Handle CUDA versions (e.g., 2.7.1+cu118 should match 2.7.1)
        if expected_version and '+' in version:
            base_version = version.split('+')[0]
            if base_version == expected_version:
                return True, version
        
        if expected_version and version != expected_version:
            return False, f"Expected {expected_version}, got {version}"
        
        return True, version
    except ImportError:
        return False, "Not installed"

def main():
    """Main verification function."""
    print("=" * 60)
    print("🔍 ACTIVE LEARNING ENVIRONMENT VERIFICATION")
    print("=" * 60)
    
    # Core dependencies with expected versions
    core_deps = [
        ("torch", "2.7.1"),
        ("torchvision", "0.22.1"),
        ("pytorch_lightning", "2.5.2"),
        ("torchmetrics", "1.8.0"),
        ("tensorboard", "2.20.0"),
        ("numpy", "2.0.2"),
        ("pandas", "2.3.1"),
        ("matplotlib", "3.9.2"),
        ("seaborn", "0.13.2"),
        ("cv2", None),  # opencv-python
        ("PIL", None),  # pillow
        ("tqdm", "4.67.1"),
        ("sklearn", "1.6.1"),  # scikit-learn
        ("skimage", "0.24.0"),  # scikit-image
    ]
    
    print(f"📦 Python Version: {sys.version.split()[0]}")
    print()
    
    all_passed = True
    
    print("📋 Core Dependencies:")
    print("-" * 40)
    
    for package, expected_version in core_deps:
        success, version = check_version(package, expected_version)
        
        if success:
            # For CUDA packages, consider them successful if base version matches
            if expected_version and '+' in version:
                base_version = version.split('+')[0]
                if base_version == expected_version:
                    status = "✅"
                else:
                    status = "⚠️ "
                    all_passed = False
            elif expected_version and version != expected_version:
                status = "⚠️ "
                all_passed = False
            else:
                status = "✅"
        else:
            status = "❌"
            all_passed = False
        
        print(f"{status} {package:<20} {version}")
    
    print()
    
    # CUDA check
    print("🚀 CUDA Compatibility:")
    print("-" * 40)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
            print(f"✅ CUDA Version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA Not Available (CPU-only mode)")
    except Exception as e:
        print(f"❌ CUDA Check Failed: {e}")
        all_passed = False
    
    print()
    
    # Framework compatibility check
    print("🔧 Framework Compatibility:")
    print("-" * 40)
    
    try:
        import pytorch_lightning as pl
        from ctc_segmentation_model import CTCSegmentationModel
        print("✅ PyTorch Lightning integration")
        print("✅ Custom segmentation model import")
    except Exception as e:
        print(f"❌ Framework compatibility issue: {e}")
        all_passed = False
    
    print()
    print("=" * 60)
    
    if all_passed:
        print("🎉 ALL CHECKS PASSED! Environment ready for Active Learning experiments.")
        print()
        print("🚀 Next Steps:")
        print("   1. Run quick validation: bash run_quick_benchmark.sh")
        print("   2. Run full benchmark: bash run_comprehensive_benchmark_v2.sh")
    else:
        print("⚠️  SOME ISSUES DETECTED. Please check the failed items above.")
        print()
        print("💡 Solutions:")
        print("   - Reinstall failed packages: pip install -r requirements.txt")
        print("   - Check CUDA installation for GPU acceleration")
        print("   - Verify environment activation: conda activate livecellx-al")
    
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)