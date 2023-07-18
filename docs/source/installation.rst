Installation
============

General package requirements
----------------------------

.. note:: If you encounter issues related to `lap` and `numpy`, please install `numpy` first and then install `lap`. If there are any issues with `numba` and `numpy`, please follow the error messages and resolve `numba` and `numpy` version issues.

Install dependencies by running::

    pip install -r requirements.txt
    pip install -r napari_requirements.txt 
    pip install .

Alternatively, to install an editable version and develop the package, run::

    pip install -e .

Pytorch (including torchvision)
-------------------------------
Please refer to the `Pytorch Official Website <https://pytorch.org/>`_ for the most recent installation instructions. Here we simply provide two examples used in our cases.

Install via pip::

    conda install pytorch torchvision -c pytorch

On 2080Ti/3090 workstations and CUDA 11.7, run::

    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

Check if you are using cuda (refer to PyTorch docs for TPU or other devices) using the commands `torch.cuda.is_available()`, `torch.cuda.current_device()`, and `torch.cuda.device_count()`.

Detectron2 (optional)
---------------------

Please refer to the `latest detectron2 documentation <https://detectron2.readthedocs.io/en/latest/tutorials/install.html>`_ to install `detectron2` for segmentation if you cannot build from source with the following commands.

Prebuilt (Easier and preferred by us) can be found `detectron2 install documentation <https://detectron2.readthedocs.io/en/latest/tutorials/install.html#install-pre-built-detectron2-linux-only>`_.

Or, you can build from source by following the instructions `detectron2 build documentation <https://detectron2.readthedocs.io/en/latest/tutorials/install.html#build-detectron2-from-source>`_. For this, run::

    git clone https://github.com/facebookresearch/detectron2.git
    python -m pip install -e detectron2

For {avi, mp4} movie generation, run::

    conda install -c conda-forge ffmpeg

Precommit [Dev]
---------------

To install pre-commit, run::

    pip install pre-commit
    pre-commit install
