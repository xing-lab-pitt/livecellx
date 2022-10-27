from livecell_tracker.core.single_cell import SingleCellStatic, SingleCellTrajectory, SingleCellTrajectoryCollection
import numpy as np
from napari.viewer import Viewer
from livecell_tracker.core.visualizer import Visualizer


class NapariVisualizer:
    def viz_traj(traj: SingleCellTrajectory, viewer: Viewer, viewer_kwargs=None):
        if viewer_kwargs is None:
            viewer_kwargs = dict()
        shapes = traj.get_sc_napari_shapes()
        shape_layer = viewer.add_shapes(shapes, **viewer_kwargs)
        return shape_layer

    def viz_traj_collection(traj_collection: SingleCellTrajectoryCollection, viewer: Viewer, viewer_kwargs=None):
        if viewer_kwargs is None:
            viewer_kwargs = dict()
        all_shapes = []
        for traj_id, traj in traj_collection:
            traj_shapes = traj.get_sc_napari_shapes()
            all_shapes.extend(traj_shapes)
        shape_layer = viewer.add_shapes(all_shapes, **viewer_kwargs)
        return shape_layer
