
<img src="https://github.com/xing-lab-pitt/livecellx/blob/main/docs/source/_static/left-logo.png" alt="" height="250"/>

# LivecellX

[![Supported Python versions](https://img.shields.io/badge/python-3.8%7C3.9%7C3.10-blue)](https://python.org)
[![Development Status](https://img.shields.io/badge/status-pre--alpha-yellow)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Pre-alpha)
[![Documentation Status](https://readthedocs.org/projects/livecellx/badge/?version=latest)](https://livecellx.readthedocs.io/en/latest/?badge=latest)

LivecellX is a comprehensive deep learning live-cell analysis framework written in Python, designed specifically for segmenting, tracking, and analyzing single-cell trajectories in live-cell imaging datasets.  


For more information, installation instructions, and tutorials, please visit our [official documentation](https://livecellx.readthedocs.io/en/latest/).

> **Note:** This repository is in a pre-alpha stage. While it currently showcases basic use-cases like image segmentation and cell tracking, our complete version is slated for release in Dec. 2023 alongside our manuscript and live-cell imaging annotated dataset. In the meantime, you may explore our [previous pipeline repository](https://github.com/xing-lab-pitt/xing-vimentin-dic-pipeline) maintained by Xing Lab.

## Installation

### General Requirements

If you encounter issues related to `lap` and `numpy`, or `numba` and `numpy`, please install `numpy` first, then `lap`. Follow the error messages to resolve any version conflicts between `numba` and `numpy`.

```bash
pip install -r requirements.txt
pip install -r napari_requirements.txt
pip install -e .  # -e option allows for an editable installation, useful for development
```

#### **Pytorch and torchvision**  
Please refer to [Pytorch Official Website](https://pytorch.org/get-started/locally) to receive most recent installation instructions. Here we simply provide two examples used in our cases.  

Install via pip:  
```bash
conda install pytorch torchvision -c pytorch
```

On our 2080Ti/3090/4090 workstations and CUDA 11.7:  
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
### High-level abstract
single-cell; segmentation; correction; tracking; morphology; dynamics; biological process detection

### Correct segmentation network
We address incorrect segmentation cases, particularly over-segmentation and under-segmentation cases, using a correction segmentation network (CSN) that follows a whole-image level segmentation provided by deep learning segmentation methods. The CSN framework simplifies the data collection process through active learning and a human-in-the-loop approach. To the best of our knowledge, we are providing the community with the first microscopy imaging correct segmentation dataset for over- and under- segmentation. 

### *LivecellAction: Guiding Deep Learning Models for Precise Detection of Rare Single-Cell Actions*
We provide a tool for users to annotate live-cell imaging datasets in Napari, and generate videos for deep learning training on video related tasks based on CNN or vision transformer based models.  

To classify biological processes, we design a framework to classify and detect biological processes. We show case how to apply this framework by applying it to the single-cell mitosis trajectory classification task. Our framework achieves near-perfect detection accuracy, exceeding 99.99%, on label-free live-cell mitosis classification task. You can follow our Jupyter notebooks in ./notebooks to reproduce our results and apply the pretrained models to detect mitosis events in your dataset. This framework can be applied to other biological processes. The models we include and benchmark in our paper: temporal segment netowrks, timeSformer, visual transformer (ViT) and resnet50. We designed a programming interface to apply trained models to our single cell trajectory data structures.


## Precommit [Dev]  
`pip install pre-commit`  
`pre-commit install`
