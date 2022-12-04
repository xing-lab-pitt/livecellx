.. automodule:: livecell_tracker

API
===

Import livecell_tracker as::

   import livecell_tracker as lct



Core
~~~~~~~~~~~~~~~~~~


Single Cell Classes
------------------
    .. autosummary::
        :toctree: _autosummary
        :template: class.rst

            livecell_tracker.core.SingleCellStatic
            livecell_tracker.core.SingleCellTrajectory
            livecell_tracker.core.SingleCellTrajectoryCollection

Dataset Classes
------------------
    .. autosummary::
        :toctree: _autosummary
        :template: class.rst

            livecell_tracker.core.datasets.LiveCellImageDataset


Data IO
------------------
    .. autosummary::
    :toctree: _autosummary

    livecell_tracker.core.io_utils.save_tiff
    livecell_tracker.core.io_utils.save_png
    livecell_tracker.core.io_utils.save_general
        .. .. autoclass:: livecell_tracker.core.datasets.LiveCellImageDataset
        ..     :members:
        ..     :inherited-members:


        .. .. autoclass:: livecell_tracker.core.SingleCellStatic
        ..     :members:
        ..     :inherited-members:

    ..
    livecell_tracker.core.datasets
    livecell_tracker.core.datasets.LiveCellImageDataset


Model Zoo
~~~~~~~~~~~~~~~~~~
    .. autosummary::
        :toctree: _autosummary

        livecell_tracker.model_zoo.segmentation
        livecell_tracker.model_zoo.segmentation.sc_correction_dataset.CorrectSegNetDataset
        livecell_tracker.model_zoo.segmentation.sc_correction.CorrectSegNet
        livecell_tracker.model_zoo.segmentation.train_csn.main_train

Segmentation
~~~~~~~~~~~~~~~~~~
    .. autosummary::
        :toctree: _autosummary
        livecell_tracker.segment.*


Track
~~~~~~~~~~~~~~~~~~


Annotation
~~~~~~~~~~~~~~~~~~