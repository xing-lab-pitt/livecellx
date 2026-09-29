import datetime
from livecellx.core.single_cell import SingleCellStatic, SingleCellTrajectory, SingleCellTrajectoryCollection
import numpy as np
from napari.viewer import Viewer
from livecellx.livecell_logger import main_info, main_warning
from livecellx.plot.visualizer import Visualizer

try:
    from shapely.geometry import Polygon
except ImportError:
    Polygon = None


class NapariVisualizer:
    @staticmethod
    def _polygon_area(shape):
        points = np.asarray(shape, dtype=float)
        coords = points[:, 1:]
        rows = coords[:, 0]
        cols = coords[:, 1]
        return 0.5 * abs(float(np.dot(cols, np.roll(rows, -1)) - np.dot(rows, np.roll(cols, -1))))

    @staticmethod
    def _polygon_is_renderable(shape):
        points = np.asarray(shape, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
            return False
        if not np.isfinite(points).all():
            return False

        coords = points[:, 1:]
        if len(np.unique(coords, axis=0)) < 3 or NapariVisualizer._polygon_area(points) <= 1e-6:
            return False
        if Polygon is not None:
            try:
                return Polygon(coords[:, ::-1]).is_valid
            except Exception:
                return False
        return True

    @staticmethod
    def _rebuild_display_contour_from_mask(sc):
        """Build a valid display-only contour from the cell label mask."""
        from scipy.ndimage import binary_dilation
        from skimage.measure import find_contours

        if sc.mask_dataset is None or sc.bbox is None:
            return None, False
        label = sc.meta.get("label_in_mask")
        if label is None:
            return None, False

        try:
            label_image = sc.mask_dataset.get_img_by_time(sc.timeframe)
            row0, col0, row1, col1 = np.asarray(sc.bbox, dtype=int)
            row0 = max(0, row0)
            col0 = max(0, col0)
            row1 = min(label_image.shape[0], row1)
            col1 = min(label_image.shape[1], col1)
            cell_mask = label_image[row0:row1, col0:col1] == int(label)
            if not cell_mask.any():
                return None, False

            was_dilated = int(cell_mask.sum()) <= 2
            if was_dilated:
                cell_mask = binary_dilation(cell_mask, iterations=1)

            padded_mask = np.pad(cell_mask.astype(bool), 1)
            candidates = []
            for contour in find_contours(padded_mask.astype(float), level=0.5):
                global_contour = contour + np.asarray([row0 - 1, col0 - 1], dtype=float)
                # Napari closes polygon shapes itself. skimage may return an
                # explicitly closed contour whose final point repeats the
                # first; retaining both creates a zero-length closing edge
                # that can crash Triangle after display downsampling.
                if len(global_contour) > 1 and np.allclose(global_contour[0], global_contour[-1]):
                    global_contour = global_contour[:-1]
                napari_shape = np.column_stack(
                    (np.full(len(global_contour), float(sc.timeframe)), global_contour)
                )
                if NapariVisualizer._polygon_is_renderable(napari_shape):
                    candidates.append(napari_shape)
            if not candidates:
                return None, was_dilated
            return max(candidates, key=NapariVisualizer._polygon_area), was_dilated
        except Exception:
            return None, False

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
        rebuilt_count = 0
        dilated_count = 0
        bbox_fallback_count = 0
        for track_id, traj in trajectories:
            traj_shapes, scs = traj.get_scs_napari_shapes(
                bbox=bbox, contour_sample_num=contour_sample_num, return_scs=True
            )
            display_shapes = []
            for shape, sc in zip(traj_shapes, scs):
                if bbox or NapariVisualizer._polygon_is_renderable(shape):
                    display_shapes.append(shape)
                    continue

                rebuilt_shape, was_dilated = NapariVisualizer._rebuild_display_contour_from_mask(sc)
                if rebuilt_shape is not None:
                    # Match SingleCellStatic.get_napari_shape_contour_vec:
                    # reconstructed contours are display-only, but they must
                    # still honor the requested contour sampling density.
                    if contour_sample_num is not None and np.isfinite(contour_sample_num):
                        slice_step = max(int(len(rebuilt_shape) / contour_sample_num), 1)
                        rebuilt_shape = rebuilt_shape[::slice_step]

                    if NapariVisualizer._polygon_is_renderable(rebuilt_shape):
                        display_shapes.append(rebuilt_shape)
                        rebuilt_count += 1
                        dilated_count += int(was_dilated)
                        continue

                display_shapes.append(sc.get_napari_shape_bbox_vec())
                bbox_fallback_count += 1

            all_shapes.extend(display_shapes)
            track_ids.extend([int(track_id)] * len(display_shapes))
            all_scs.extend(scs)
            all_status.extend([""] * len(display_shapes))
        properties = {"track_id": track_ids, "sc": all_scs, "status": all_status}

        # Track ID can be UUID, so we need to map it to an integer
        track_value_indices = [idx for idx, v in enumerate(track_ids)]
        main_info(f"Number of trajectories: {len(trajectories)}", indent_level=2)
        if rebuilt_count or bbox_fallback_count:
            main_warning(
                f"Rebuilt display contours for {rebuilt_count} invalid masks "
                f"({dilated_count} tiny masks dilated by 1 pixel); "
                f"{bbox_fallback_count} unrecoverable masks use bbox display. "
                "Degenerate or self-intersecting polygons can crash Napari. "
                "Original masks, trajectories, and saved results were not modified."
            )

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
