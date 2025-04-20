.. livecellx documentation master file, created by
   sphinx-quickstart on Thu Dec  1 22:22:40 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

LivecellX
======================================

**LivecellX** is a deep-learning-based, single-cell object-oriented framework designed for quantitative analysis of single-cell dynamics in long-term, label-free live-cell imaging datasets.

Manuscript
----------
For detailed methods and validations, please see our manuscript:

> Ni et al., LivecellX: A Deep-learning-based, Single-Cell Object-Oriented Framework for Quantitative Analysis in Live-Cell Imaging (*__stub__*, under review).

Key Features
------------
- **Segmentation & Tracking**: Accurate deep learning-based segmentation, integrated tracking algorithms (SORT, b-track).
- **Corrective Segmentation Network (CS-Net)**: Automatically correct over- and under-segmentation errors using context-aware deep learning models.
- **Trajectory-Level Correction**: Algorithms utilizing temporal consistency for error correction and accurate lineage reconstruction.
- **Biological Process Detection**: Automated detection and classification of cellular processes (mitosis, apoptosis).
- **High-dimensional Feature Extraction**: Including morphological (Active Shape Models), textural (Haralick, LBP), and deep learning-based features (VAE).
- **Object-Oriented Data Structure**: Intuitive and efficient management of single-cell trajectories and features, enabling multi-dataset integration.
- **Napari GUI Integration**: Interactive visualization, manual correction, and lineage tracing.
- **Parallelized Computation**: Efficient processing of large datasets using multi-core computation.

Getting Started
---------------
- Explore our [Tutorials](livecellx_notebooks/tutorials/index) and [Examples](livecellx_notebooks/examples/index).
- Contribute via [GitHub](https://github.com/xing-lab-pitt/livecellx).

.. toctree::
    :caption: General
    :maxdepth: 2
    :hidden:

    installation
    API
    classes
    livecellx_notebooks/tutorials/tutorial_general_usage
    livecellx_notebooks/tutorials/index
    livecellx_notebooks/examples/index

.. toctree::
    :caption: Gallery
    :maxdepth: 2
    :hidden:

    gallery/index

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`