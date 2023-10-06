
<img src="https://github.com/xing-lab-pitt/livecellx/blob/main/docs/source/_static/left-logo.png" alt="" height="250"/>

# LivecellX

[![Supported Python versions](https://img.shields.io/badge/python-3.8%7C3.9%7C3.10-blue)](https://python.org)
[![Development Status](https://img.shields.io/badge/status-pre--alpha-yellow)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Pre-alpha)
[![Documentation Status](https://readthedocs.org/projects/livecellx/badge/?version=latest)](https://livecellx.readthedocs.io/en/latest/?badge=latest)

LivecellX is a comprehensive Python framework designed for segmenting, tracking, and analyzing single-cell trajectories in long-term live-cell imaging datasets.

For more information, installation instructions, and tutorials, please visit our [official documentation](https://livecellx.readthedocs.io/en/latest/).

> **Note:** This repository is in a pre-alpha stage. While it currently showcases basic use-cases like image segmentation and cell tracking, our complete version is slated for release in October 2023 alongside our manuscript. In the meantime, you may explore our [previous pipeline repository](https://github.com/xing-lab-pitt/xing-vimentin-dic-pipeline) maintained by Xing Lab.

## Installation

### General Requirements

If you encounter issues related to `lap` and `numpy`, or `numba` and `numpy`, please install `numpy` first, then `lap`. Follow the error messages to resolve any version conflicts between `numba` and `numpy`.

```bash
pip install -r requirements.txt
pip install -r napari_requirements.txt
pip install lap[all]
pip install -e .  # -e option allows for an editable installation, useful for development
```

#### **Pytorch and torchvision**  
Please refer to [Pytorch Official Website](https://pytorch.org/get-started/locally) to receive most recent installation instructions. Here we simply provide two examples used in our cases.  

Install via pip:  
```bash
conda install pytorch torchvision -c pytorch
```

On our 2080Ti/3090 workstations and CUDA 11.7:  
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

check if you are using cuda (refer to pytorch docs for TPU or other devices):
```bash
torch.cuda.is_available(), torch.cuda.current_device(), torch.cuda.device_count()
``````


#### **Detectron2 (optional)**  

Please refer to latest detectron2 documentation to install detectron2 for segmentation if you cannot build from source with the following commands.  

Prebuilt (Easier and preferred by us):  
https://detectron2.readthedocs.io/en/latest/tutorials/install.html#install-pre-built-detectron2-linux-only

Build from source:  
https://detectron2.readthedocs.io/en/latest/tutorials/install.html#build-detectron2-from-source

```
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

**For {avi, mp4} movie generation, ffmpeg is required. Conda installation cmd we used shown below. For other installation methods, please refer to the ffmpeg official website.**
```bash
conda install -c conda-forge ffmpeg
```

## Precommit [Dev]  
`pip install pre-commit`  
`pre-commit install`

## Expected input/output for each submodule

**Note**  
If you already have satisfying segmentation models or segmentation results, you may skip **Annotation** and **Segmentation** part below.
### Annotation
input: raw image files
After annotating imaging datasets, you should have json files in COCO format ready for segmentation training. 

#### Labelme
Apply labelme to your datasets following our annotation protocol. 
#### Convert labelme json to COCO format. 
A fixed version of labelme2coco implementation is included in our package. Please refer to our tutorial on how to convert your labelme json to COCO format.  
For CVAT, please export the annotation results as COCO, as shown in our annotation protocol.

### Segmentation
Segmentation has two phase. If you already have pytorch or tensorflow models trained on your dataset, you may skip training phase.

### training phase
input: COCO json files

output: pytorch model (.pth file)

### prediction phase
input: raw images, a trained machine-learning based model  
outputs: SingleCellStatic json outputs

### Track
input: SingleCellStatic
- contour
- bounding box

output: SingleCellTrajectoryColletion
- holding a collection of singleCellTrajectory each containing single cell time-lapse data
- trajectory-wise feature can be calculated after track stage or at trajectory stage.

### Trajectory
input: SingleCellTrajectoryColletion

output: 


### Visualizer
track.movie: generate_single_trajectory_movie()

visualizer: viz_traj, viz_traj_collection

{Documentation placeholder} [Move to docs/ and auto generate by readthedocs]

### Analyze trajectories based on specific research topics


## SingleCellStatic  
class designed to hold all information about a single cell at some timepoint  
**attributes**
- time point
- id (optional)
- contour coordinates
- cell bounding box
- img crop (lazy)
- feature map 
- original img (reference/pointer)

## SingleCellTrajectory
- timeframe_set

## SingleCellTrajectoryCollection