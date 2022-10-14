
# livecell-tracker

[![Supported Python versions](https://img.shields.io/badge/python-3.8%7C3.9%7C3.10-blue)](https://python.org)
[![Development Status](https://img.shields.io/badge/status-pre--alpha-yellow)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Pre-alpha)

## Installation

Please refer to latest detectron2 documentation to install detectron2 for segmentation if you cannot build from source with the following commands.
https://detectron2.readthedocs.io/en/latest/tutorials/install.html#build-detectron2-from-source

```
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

install pytorch  
`conda install pytorch torchvision -c pytorch`

install package requirements  
`pip install -r requirements.txt`


for (avi, mp4) movie generation  
`conda install -c conda-forge ffmpeg`