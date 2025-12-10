<!-- <img src="https://github.com/xing-lab-pitt/livecellx/blob/main/docs/source/_static/logo.png" alt="LivecellX Logo" width="250"/> -->

# LivecellX

[![Supported Python versions](https://img.shields.io/badge/python-3.8%7C3.9%7C3.10-blue)](https://python.org)
[![Development Status](https://img.shields.io/badge/status-pre--alpha-yellow)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Pre-alpha)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Documentation Status](https://readthedocs.org/projects/livecellx/badge/?version=latest)](https://livecellx.readthedocs.io/en/latest/?badge=latest)

**LivecellX** is a comprehensive deep learning framework for live-cell image analysis, enabling researchers to segment, track, and analyze single-cell trajectories in high-throughput imaging datasets.

## 🔬 Key Capabilities

- **Instance Segmentation**: Deep learning-based cell detection and segmentation with correction networks
- **Correction Segmentation Network (CS-Net)**: Handle over- and under-segmentation errors through active learning
- **Cell Tracking**: Temporal correspondence and trajectory reconstruction across frames
- **Trajectory Analysis**: Single-cell morphology, dynamics, and quantitative metrics
- **Biological Event Detection**: Classify and detect rare cellular processes (e.g., mitosis and apoptosis detection)
- **Interactive Annotation**: Napari-based annotation tools with human-in-the-loop active learning
- **Video Analysis**: Support for CNN and Vision Transformer models on temporal data

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Core Modules](#core-modules)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

```python
# Datasets
from livecellx.core.datasets import LiveCellImageDataset
from livecellx.core.single_cell import SingleCellStatic, SingleCellTrajectoryCollection

# Tracking utilities (SORT and btrack)
from livecellx.track import track_SORT_bbox_from_scs, track_btrack_from_scs

# CS-Net single-cell correction API
from livecellx.model_zoo.segmentation.csn_sc_utils import correct_sc

# 1) Load image and (optional) label-mask datasets
img_ds = LiveCellImageDataset(img_dir="/path/to/images")
mask_ds = LiveCellImageDataset(img_dir="/path/to/label_masks")  # integer labels per object

# 2) Build SingleCellStatic objects from a mask at time t
from livecellx.segment.utils import process_scs_from_one_label_mask
time_t = 0
scs_t = process_scs_from_one_label_mask(mask_ds, img_ds, time_t)

# 3) Initialize trajectory collection and run tracking over a sequence
traj_collection = SingleCellTrajectoryCollection(img_dataset=img_ds)

# Example A: SORT tracking using bounding boxes inferred from contours
traj_collection = track_SORT_bbox_from_scs(
  single_cells=scs_t,
  raw_imgs=img_ds,
  mask_dataset=mask_ds,
  max_age=5,     # max frames to keep unmatched tracks
  min_hits=3,    # detections required to initiate a track
)

# Example B: btrack tracking using SingleCell objects
traj_collection = track_btrack_from_scs(
  single_cells=scs_t,
  raw_imgs=img_ds,
  mask_dataset=mask_ds,
  # config=...   # optional btrack configuration dict
)

# 4) Access trajectories and single-cell features
for traj_id, traj in traj_collection.items():
  print(traj_id, traj.length())

# 5) Correct a single-cell segmentation using CS-Net
sc = scs_t[0]

# Load a trained CS-Net model (aux variant shown)
from livecellx.model_zoo.segmentation.sc_correction_aux import CorrectSegNetAux
model = CorrectSegNetAux.load_from_checkpoint("/path/to/csn_checkpoint.ckpt")
model.eval().cuda()

# Apply correction with required parameters
sc_corrected_list = correct_sc(
  _sc=sc,
  model=model,
  padding=16,                 # pad crop around the single cell
  input_transforms=None,      # optional torchvision transforms
  gpu=True,                   # use CUDA if available
  return_outputs=False,       # set True to also get masks and labels
  h_threshold=1,              # watershed threshold for mask splitting
)
```

For detailed tutorials, visit our [official documentation](https://livecellx.readthedocs.io/en/latest/).

## Installation

### Prerequisites

- Python 3.8+
- For GPU support: CUDA 11.x or compatible version

### Basic Installation

```bash
# Install core dependencies
pip install -r requirements.txt
pip install -r napari_requirements.txt

# Install package in development mode
pip install -e .
```

### Optional Dependencies

#### PyTorch and TorchVision

Refer to [PyTorch Official Installation Guide](https://pytorch.org/get-started/locally) for platform-specific instructions. Examples:

```bash
# Default CPU/GPU installation
conda install pytorch torchvision -c pytorch

# CUDA 11.7 on Linux
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

Verify CUDA availability:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device_count())
```

#### Detectron2 (Optional)

Required for advanced segmentation features. See [Detectron2 Installation Guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

**Pre-built (Recommended):**
```bash
pip install detectron2
```

**From Source:**
```bash
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

#### FFmpeg (Optional)

Required for video output (AVI, MP4 formats):

```bash
conda install -c conda-forge ffmpeg
```

### Troubleshooting

If you encounter dependency conflicts between `numpy`, `lap`, or `numba`:

1. Install `numpy` first
2. Then install `lap`
3. Check and resolve `numba` version conflicts

## Core Modules

LivecellX is organized into specialized submodules:

| Module | Purpose |
|--------|---------|
| `livecellx.segment` | Cell segmentation and instance detection |
| `livecellx.track` | Cell tracking and trajectory linking |
| `livecellx.trajectory` | Trajectory analysis and metrics |
| `livecellx.classification` | Biological event classification |
| `livecellx.annotation` | Annotation tools and dataset management |
| `livecellx.preprocess` | Image preprocessing and normalization |
| `livecellx.plot` | Visualization utilities |
| `livecellx.viz` | Advanced visualization and interactive tools |

## Usage Examples

### Cell Segmentation and Correction

The Correction Segmentation Network (CS-Net) improves segmentation quality by handling over- and under-segmentation errors. See `notebooks/` for detailed examples.

Common utilities:

```python
from livecellx.segment.utils import get_contours_from_pred_masks
from livecellx.segment.csn_utils import make_csn_weight_map

# Convert instance masks to contours (for downstream tracking or visualization)
contours = get_contours_from_pred_masks(instance_pred_masks)

# Build CS-Net training weight maps directly from label masks
weight_map = make_csn_weight_map(label_mask)
```

### CS-Net with Auxiliary Classifier

CS-Net performs single-cell segmentation correction by leveraging the raw image, an initial segmentation mask, and learned correction signals. Our implementation includes an optional auxiliary classifier that helps the network learn global context and improves robustness in under/over-segmentation scenarios.

- Implementation: `livecellx/model_zoo/segmentation/sc_correction_aux.py` (`CorrectSegNetAux` PyTorch Lightning module)
- Dataset: `livecellx/model_zoo/segmentation/sc_correction_dataset.py` (prepares inputs/targets)
- Model backbones: `deeplabv3_resnet50` or `UNetWithAux`

Training loss options include CE, MSE, and BCE-with-logits with per-pixel weights. The auxiliary head predicts coarse object-level classes to guide spatial corrections.

```python
# Inference usage for single-cell correction
from livecellx.model_zoo.segmentation.csn_sc_utils import correct_sc

sc_corrected_list = correct_sc(
  _sc=sc,                    # SingleCellStatic instance
  model=model,               # CorrectSegNetAux or compatible CS-Net
  padding=16,
  input_transforms=None,
  gpu=True,
  return_outputs=False,
  h_threshold=1,
)

# The returned list contains corrected SingleCellStatic objects
print([sc_.bbox for sc_ in sc_corrected_list])
```

### Live-Cell Action Classification

Detect and classify biological processes using video-based deep learning:

```python
from livecellx.classification import detect_mitosis

# Detect mitosis events in cell trajectories
mitosis_results = detect_mitosis(trajectories, model='timesformer')
```

Supported models: Temporal Segment Networks, TimeSformer, Vision Transformer (ViT), ResNet50

### Interactive Annotation with Napari

Use the Napari-based annotation tool to create custom training datasets:

```bash
napari  # Open napari
# Use LivecellX annotation plugin to label images
```

## 📚 Documentation and Examples

- **Comprehensive Documentation**: https://livecellx.readthedocs.io/
- **Jupyter Notebooks**: See `notebooks/` directory for tutorials and applications
  - Segmentation examples
  - Tracking pipelines
  - Classification workflows
  - Benchmark comparisons

## Contributing

We welcome contributions! For development setup:

```bash
pip install pre-commit
pre-commit install
```

This ensures code quality checks before each commit.

## Citation

If you use LivecellX in your research, please cite our work. Check the repository for publication details.

## License

LivecellX is released under the GNU General Public License v3 (GPLv3). See `LICENSE` file for details.
