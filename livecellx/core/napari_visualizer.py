import datetime
from livecellx.core.single_cell import SingleCellStatic, SingleCellTrajectory, SingleCellTrajectoryCollection
import numpy as np
from napari.viewer import Viewer
from livecellx.livecell_logger import main_info
from livecellx.plot.visualizer import Visualizer


class NapariVisualizer:
    def viz_traj(traj: SingleCellTrajectory, viewer: Viewer, viewer_kwargs=None):
        if viewer_kwargs is None:
            viewer_kwargs = dict()
        shapes = traj.get_scs_napari_shapes()
        shape_layer = viewer.add_shapes(shapes, **viewer_kwargs)
        return shape_layer

    @staticmethod
    def map_colors(values, cmap="viridis"):
        import matplotlib
        import matplotlib.cm as cm

        if values is None or len(values) == 0:
            return []

        minima = min(values)
        maxima = max(values)

        norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        res_colors = [mapper.to_rgba(v) for v in values]
        return res_colors

    def gen_trajectories_shapes(
        trajectories: SingleCellTrajectoryCollection,
        viewer: Viewer,
        bbox=False,
        contour_sample_num=100,
        viewer_kwargs=None,
        text_parameters={
            "string": "{track_id:0.0f}\n{status}",
            "size": 12,
            "color": "white",
            "anchor": "center",
            "translation": [-2, 0],
        },
    ):
        if viewer_kwargs is None:
            viewer_kwargs = dict()
        all_shapes = []
        track_ids = []
        all_scs = []
        all_status = []
        for track_id, traj in trajectories:
            traj_shapes, scs = traj.get_scs_napari_shapes(
                bbox=bbox, contour_sample_num=contour_sample_num, return_scs=True
            )
            all_shapes.extend(traj_shapes)
            track_ids.extend([int(track_id)] * len(traj_shapes))
            all_scs.extend(scs)
            all_status.extend([""] * len(traj_shapes))
        properties = {"track_id": track_ids, "sc": all_scs, "status": all_status}

        # Track ID can be UUID, so we need to map it to an integer
        track_value_indices = [idx for idx, v in enumerate(track_ids)]
        main_info(f"Number of trajectories: {len(trajectories)}", indent_level=2)

        main_info("Calling viewer.add_shapes to add trajectories to napari", indent_level=2)

        # Record running time
        start_time = datetime.datetime.now()
        shape_layer = viewer.add_shapes(
            all_shapes,
            properties=properties,
            face_color=NapariVisualizer.map_colors(track_value_indices),
            face_colormap="viridis",
            shape_type="polygon",
            text=text_parameters,
            name="trajectories",
            **viewer_kwargs,
        )
        end_time = datetime.datetime.now()

        # Report time in seconds
        main_info(f"Time to add shapes: {(end_time - start_time).total_seconds()}", indent_level=2)

        return shape_layer
