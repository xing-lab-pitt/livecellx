.. livecellx documentation master file, created by
   sphinx-quickstart on Thu Dec  1 22:22:40 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

LivecellX
======================================
**livecellx** is an end-to-end system for helping users extracting single-cell trajectories from long live-cell imaging data and computing as well as analyzing single-cell features in latent space.


**[More docs Coming soon!!! readthedocs website still underconstruction, suggestions welcome!]**


Manuscript
----------
Please see our manuscript [insert citation here] to learn more.

Key Features
------------

- Load large livecell imaging datasets on disk
- Segment and track cells in live-cell imaging data 
    - with any popular tools such as OpenCV, SORT, or btrack.
    - with our own deep learning-based segmentation model
- Correct single cell segmentation on single-cell level in our Napari UI operator
- Generate cell features
- Analyze and visualize single-cell trajectories.

Getting Started with livecell
-----------------------------
- Browse the [tutorials/index](<insert tutorials/index link here>) and [examples/index](<insert examples/index link here>).
- Contribute to the project on `github`_.

.. toctree::
    :caption: General
    :maxdepth: 2
    :hidden:

    installation
    API
    classes
    /livecellx_notebooks/tutorials/tutorial_general_usage


.. toctree::
    :caption: Gallery
    :maxdepth: 2
    :hidden:

    livecellx_notebooks/tutorials/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _gitHub: https://github.com/xing-lab-pitt/LiveCellTracker-dev