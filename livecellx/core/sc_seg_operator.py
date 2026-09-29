import copy
import cv2
from functools import partial
from typing import Optional, Tuple, Union, Annotated
import magicgui as mgui
from magicgui import magicgui
from magicgui.widgets import Container, PushButton, Widget, create_widget
from napari.layers import Shapes
import torch
from livecellx.core.single_cell import SingleCellTrajectoryCollection, SingleCellStatic
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from napari.layers import Shapes

from livecellx.livecell_logger import main_info, main_warning, main_debug
from livecellx.core import SingleCellTrajectory, SingleCellStatic
from livecellx.segment.ou_simulator import find_label_mask_contours
from livecellx.segment.ou_utils import create_ou_input_from_sc
from livecellx.segment.utils import find_contours_opencv, filter_contours_by_size
from livecellx.core.datasets import SingleImageDataset


def correct_sc_segment(
    sc,
    model,
    create_ou_input_kwargs={
        "padding_pixels": 50,
        "dtype": float,
        "remove_bg": False,
        "one_object": True,
        "scale": 0,
    },
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
    import torch
    from torchvision import transforms
    from livecellx.model_zoo.segmentation.sc_correction_aux import CorrectSegNetAux

    #  padding_pixels=padding_pixels, dtype=dtype, remove_bg=remove_bg, one_object=one_object, scale=scale
    input_transforms = transforms.Compose(
        [
            transforms.Resize(size=(412, 412)),
        ]
    )

    temp_sc = sc.copy()
    new_contour = np.array(temp_sc.contour)
    new_contour = new_contour[:, -2:]  # remove slice index (time)
    temp_sc.update_contour(new_contour)
    temp_sc.update_bbox()
    res_bbox = temp_sc.bbox
    ou_input = create_ou_input_from_sc(temp_sc, **create_ou_input_kwargs)
    # ou_input = create_ou_input_from_sc(self.sc, **create_ou_input_kwargs)
    original_shape = ou_input.shape

    # TODO: change to comply with the training data preparation
    # for now we simply use one of the input types during training: raw_aug_duplicate.
    # Please read sc_correction_dataset impl.
    ou_input = input_transforms(torch.tensor([ou_input]))
    ou_input = torch.stack([ou_input, ou_input, ou_input], dim=1)
    ou_input = ou_input.float().cuda()

    back_transforms = transforms.Compose(
        [
            transforms.Resize(size=(original_shape[0], original_shape[1])),
        ]
    )
    seg_output, aux_output = None, None
    if isinstance(model, CorrectSegNetAux):
        model_output = model(ou_input)
        seg_output, aux_output = model_output
    else:
        seg_output = model(ou_input)
    seg_output = back_transforms(seg_output)
    if not model.apply_gt_seg_edt:
        seg_output = torch.sigmoid(seg_output)
    return ou_input, seg_output, res_bbox, aux_output


class ScSegOperator:
    """
    A class for performing segmentation on single cell images.

    Attributes
    ----------
    viewer : napari.Viewer
        The napari viewer.
    sc : SingleCellStatic
        The single cell static object.
    shape_layer : napari.layers.Shapes
        The napari shape layer for displaying the segmentation.
    """

    MANUAL_CORRECT_SEG_MODE = 0
    CSN_CORRECT_SEG_MODE = 1

    DEFAULT_CSN_MODEL = None

    @staticmethod
    def load_default_csn_model(path, cuda=True, has_aux=True):
        import torch
        from livecellx.model_zoo.segmentation.sc_correction import CorrectSegNet
        from livecellx.model_zoo.segmentation.sc_correction_aux import CorrectSegNetAux

        if has_aux:
            model = CorrectSegNetAux.load_from_checkpoint(path)
        else:
            model = CorrectSegNet.load_from_checkpoint(path)
        if cuda:
            model.cuda()
        model.eval()
        ScSegOperator.DEFAULT_CSN_MODEL = model
        return model

    def __init__(
        self,
        sc: SingleCellStatic,
        viewer,
        shape_layer: Optional[Shapes] = None,
        face_color=(0, 0, 1, 1),
        magicgui_container: Optional[Container] = None,
        csn_model=None,
        create_sc_layer=True,
        sct_observers: Optional[list] = None,
        force_fallback_shape=False,
        initial_shape_vec=None,
    ):
        """
        Parameters
        ----------
        viewer : napari.Viewer
            The napari viewer.
        sc : SingleCellStatic
            The single cell static object.
        """

        self.sc = sc
        self.viewer = viewer
        self.shape_layer = shape_layer
        self.face_color = face_color
        self.mode = self.MANUAL_CORRECT_SEG_MODE
        self.magicgui_container = magicgui_container
        self.csn_model = csn_model
        self.force_fallback_shape = force_fallback_shape
        self.initial_shape_vec = initial_shape_vec

        self.sct_observers = sct_observers
        if sct_observers is None:
            self.sct_observers = []

        if not (self.shape_layer is None):
            self.setup_edit_contour_shape_layer()

        if create_sc_layer:
            self.create_sc_layer()

    def __repr__(self) -> str:
        return f"ScSegOperator(sc={self.sc}, mode={self.mode})"

    @staticmethod
    def _shape_vec_polygon_area(shape_vec):
        shape_vec = np.asarray(shape_vec, dtype=float)
        if shape_vec.ndim != 2 or len(shape_vec) < 3:
            return 0.0
        coords = shape_vec[:, 1:3] if shape_vec.shape[1] >= 3 else shape_vec[:, -2:]
        if not np.all(np.isfinite(coords)):
            return 0.0
        y = coords[:, 0]
        x = coords[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    @staticmethod
    def _is_stable_polygon_coords(coords, min_area=1.0):
        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 3:
            return False
        if not np.all(np.isfinite(coords)):
            return False
        if len(np.unique(np.round(coords, decimals=3), axis=0)) < 3:
            return False

        y = coords[:, 0]
        x = coords[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
        if area <= min_area:
            return False

        edge_lengths = np.linalg.norm(coords - np.roll(coords, -1, axis=0), axis=1)
        positive_edges = edge_lengths[edge_lengths > 1e-6]
        if len(positive_edges) == 0:
            return False
        median_edge = float(np.median(positive_edges))
        max_edge = float(np.max(positive_edges))
        bbox_diag = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
        if median_edge > 0 and max_edge > max(30.0, 8.0 * median_edge, 0.75 * bbox_diag):
            return False
        return True

    @classmethod
    def _is_stable_shape_vec(cls, shape_vec):
        try:
            shape_vec = np.asarray(shape_vec, dtype=float)
        except Exception:
            return False
        if shape_vec.ndim != 2 or shape_vec.shape[1] < 3:
            return False
        return cls._is_stable_polygon_coords(shape_vec[:, 1:3])

    @staticmethod
    def _fallback_edit_contour(sc: SingleCellStatic, min_half_size=8):
        """Create a small valid square polygon for tiny/degenerate masks.

        Napari can crash when an editable polygon has fewer than three usable
        vertices or near-zero area.  This fallback opens an editable square near
        the cell location.  The single-cell contour is not changed until the
        user presses "save seg to sc".
        """
        center = None
        bbox = getattr(sc, "bbox", None)
        if bbox is not None:
            bbox = np.asarray(bbox, dtype=float)
            if bbox.size == 4 and np.all(np.isfinite(bbox)):
                center = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])

        if center is None:
            contour = np.asarray(getattr(sc, "contour", []), dtype=float)
            if contour.ndim == 2 and len(contour) > 0:
                coords = contour[:, -2:]
                if np.all(np.isfinite(coords)):
                    center = coords.mean(axis=0)

        if center is None:
            center = np.array([8.0, 8.0])

        half_size = float(min_half_size)
        if bbox is not None and bbox.size == 4 and np.all(np.isfinite(bbox)):
            half_size = max(half_size, float(bbox[2] - bbox[0]) / 2.0, float(bbox[3] - bbox[1]) / 2.0)

        y, x = center
        contour = np.array(
            [
                [y - half_size, x - half_size],
                [y - half_size, x + half_size],
                [y + half_size, x + half_size],
                [y + half_size, x - half_size],
            ],
            dtype=float,
        )

        try:
            image_shape = sc.get_img_shape()
            contour[:, 0] = np.clip(contour[:, 0], 0, image_shape[0] - 1)
            contour[:, 1] = np.clip(contour[:, 1], 0, image_shape[1] - 1)
        except Exception:
            pass
        return contour

    def _get_stable_edit_shape_vec(self, contour_sample_num=100):
        if self.force_fallback_shape:
            main_warning(
                f"Opening sc {self.sc.id} with a safe editable square instead of its original contour."
            )
            fallback_contour = self._fallback_edit_contour(self.sc)
            shape_vec = [[self.sc.timeframe] + list(point) for point in fallback_contour]
            return shape_vec, True

        if self.initial_shape_vec is not None and self._is_stable_shape_vec(self.initial_shape_vec):
            return np.asarray(self.initial_shape_vec, dtype=float), False

        try:
            shape_vec = self.sc.get_napari_shape_contour_vec(contour_sample_num=contour_sample_num)
        except Exception as exc:
            main_warning(f"Failed to build editable contour for sc {self.sc.id}: {exc}")
            shape_vec = []

        if self._is_stable_shape_vec(shape_vec):
            return shape_vec, False

        main_warning(
            f"sc {self.sc.id} has a tiny, disconnected, or degenerate contour for editing; "
            "opening a small editable square instead."
        )
        fallback_contour = self._fallback_edit_contour(self.sc)
        shape_vec = [[self.sc.timeframe] + list(point) for point in fallback_contour]
        return shape_vec, True

    def create_sc_layer(self, name=None, contour_sample_num=100):
        if name is None:
            name = f"sc_{self.sc.id}"
        shape_vec, used_fallback = self._get_stable_edit_shape_vec(contour_sample_num=contour_sample_num)
        shapes_data = [shape_vec]
        is_dummy_shape = False
        if len(shape_vec) == 0:
            main_warning(f"sc {self.sc.id} has no contour (or contour list length is 0)")

            # Add a square shape with area = 16
            tmp_contour = [
                [0, 0],
                [4, 0],
                [4, 4],
                [0, 4],
            ]  # Note: do not add [0, 0] to the contour list here, otherwise mysterious crashing on certain machines may occur.
            tmp_shape_data = [[self.sc.timeframe] + coord for coord in tmp_contour]
            shapes_data = [tmp_shape_data]
            is_dummy_shape = True
        elif used_fallback:
            main_info("Using fallback edit polygon; save only after manually correcting it.", indent_level=2)

        properties = {"sc": [self.sc]}
        shape_layer = self.viewer.add_shapes(
            shapes_data if len(shapes_data) > 0 else None,
            properties=properties,
            face_color=[self.face_color],
            shape_type="polygon",
            name=name,
        )
        self.shape_layer = shape_layer
        if is_dummy_shape:
            # delete the dummy shape
            self.shape_layer.data = []
        self.setup_edit_contour_shape_layer()
        print(">>> create sc layer done")

    def remove_sc_layer(self):
        if self.shape_layer is None:
            return
        self.viewer.layers.remove(self.shape_layer)
        self.shape_layer = None

    def update_shape_layer_by_sc(self, contour_sample_num=100):
        shape_vec, _ = self._get_stable_edit_shape_vec(contour_sample_num=contour_sample_num)
        self.shape_layer.data = [shape_vec]

    def correct_segment(self, model, create_ou_input_kwargs=None):
        import torch
        from torchvision import transforms

        #  padding_pixels=padding_pixels, dtype=dtype, remove_bg=remove_bg, one_object=one_object, scale=scale
        temp_sc = self.sc.copy()
        if create_ou_input_kwargs is None:
            # Use default values
            return correct_sc_segment(temp_sc, model)
        else:
            return correct_sc_segment(temp_sc, model, create_ou_input_kwargs=create_ou_input_kwargs)

    def replace_sc_contour(self, contour, padding_pixels=0, refresh=True):
        self.sc.contour = contour + self.sc.bbox[:2] - padding_pixels
        self.sc.update_bbox()
        if refresh:
            self.update_shape_layer_by_sc()

    def setup_edit_contour_shape_layer(self):
        # [DEPRECATED] Ke did not find a way to make this work
        # TODO: make sure only 1 shape in the shape layer...
        return
        # TODO
        from copy import deepcopy

        # Callback to check if shape_layer has more than one shape and remove the last one
        self.saved_data = deepcopy(self.shape_layer.data)

        def _shape_data_changed(event):
            print("_shape_data_changed fired")
            print("len of shape_layer.data:", len(self.shape_layer.data))
            if len(self.shape_layer.data) > 1:
                # self.shape_layer.events.data.disconnect(self._shape_data_changed)  # disconnect the callback
                print("[_shape_data_changed] len of saved_data:", len(self.saved_data))
                self.shape_layer.data = deepcopy(self.saved_data)
                # self.shape_layer.events.data.connect(self._shape_data_changed)
            elif len(self.shape_layer.data) == 1:
                self.saved_data = deepcopy(self.shape_layer.data)

        # If the shape_layer already exists, connect the callback
        if self.shape_layer is not None:
            self.shape_layer.events.data.connect(_shape_data_changed)

    def show_selected_mode_widget(self):

        # sc id
        self.magicgui_container[0].show()
        # switch mode
        self.magicgui_container[1].show()
        # focus on sc
        self.magicgui_container[7].show()
        if self.mode == self.MANUAL_CORRECT_SEG_MODE:
            # save_seg_to_sc
            self.magicgui_container[2].show()
            # csn_correct_seg
            self.magicgui_container[3].show()
            # clear_sc_layer
            self.magicgui_container[4].show()
            # restore_sc_contour
            self.magicgui_container[5].show()
            # filter_cells_by_size
            self.magicgui_container[6].show()
            # resample contour points
            self.magicgui_container[8].show()

    def hide_function_widgets(self):
        for i in range(2, len(self.magicgui_container)):
            self.magicgui_container[i].hide()

    def notify_sct_to_update(self):
        for observer in self.sct_observers:
            observer.update_shape_layer_by_sc(self.sc)

    def notify_sct_to_remove_sc_operator(self):
        for observer in self.sct_observers:
            observer.remove_sc_operator(self)

    @staticmethod
    def _get_contours_from_shape_layer(layer: Shapes):
        res_contours = []
        for shape in layer.data:
            vertices = np.array(shape)
            # ignore the first vertex, which is the slice index
            vertices = vertices[:, 1:3]
            res_contours.append(vertices)
        return res_contours

    def save_seg_callback(self, clip=True):
        """Save the segmentation to the single cell object."""
        import napari
        from PyQt5.QtWidgets import QMessageBox
        from livecellx.core.utils import clip_polygon

        print("<save_seg_callback fired>")
        # Get the contour coordinates from the shape layer
        contours = self._get_contours_from_shape_layer(self.shape_layer)
        if len(contours) != 1:
            message = "Warning: Expected 1 contour, found {}.".format(len(contours))
            QMessageBox.warning(None, "Warning", message)
            return
        assert len(contours) > 0, "No contour is found in the shape layer."
        contour = contours[0]  # n x 2

        # limit the contour coordinates to the image height and width
        if clip:
            main_info("Limiting the contour coordinates to the image height and width.", indent_level=2)
            main_debug("contour before clipping:" + str(contour.shape), indent_level=2)
            image_dim = self.sc.get_img_shape()
            # Clipping algorithm
            contour = clip_polygon(contour, image_dim[0], image_dim[1])

            # Ensure the contour is within the image
            contour[:, 0] = np.clip(contour[:, 0], 0, image_dim[0] - 1)
            contour[:, 1] = np.clip(contour[:, 1], 0, image_dim[1] - 1)

            # update the shape layer as well
            main_info("Updating the shape layer of sc...", indent_level=2)
            napari_vertices = [[self.sc.timeframe] + list(point) for point in contour]
            napari_vertices = np.array(napari_vertices)
            self.shape_layer.data = []
            self.shape_layer.add([(napari_vertices, "polygon")], shape_type=["polygon"])

        # Store the contour in the single cell object
        self.sc.update_contour(contour)

        # Notify the observers
        print("<save_seg_callback> notifying sct operator to update the sc")
        self.notify_sct_to_update()
        print("<save_seg_callback finished>")

    def csn_correct_seg_callback(self, padding_pixels=50, threshold=0.5):
        from livecellx.model_zoo.segmentation.csn_sc_utils import correct_sc, correct_sc_mask

        print("csn_correct_seg_callback fired")
        if self.csn_model is None and ScSegOperator.DEFAULT_CSN_MODEL is None:
            print("No CSN model is loaded. Please load a CSN model first.")
            return
        elif self.csn_model is None:
            print("Using default CSN model and loading it to the operator...")
            self.csn_model = ScSegOperator.DEFAULT_CSN_MODEL

        res_dict = correct_sc(
            self.sc,
            self.csn_model,
            padding=padding_pixels,
            input_transforms=self.csn_model.train_transforms,
            h_threshold=1.5,
            return_outputs=True,
        )
        corrected_scs = res_dict["scs"]
        label_str = res_dict["label_str"]
        main_info(f"v1 predicted segmentation class: <{str(label_str)}>")
        # contour = [0]
        new_shape_data = []
        contours = [_sc.contour for _sc in corrected_scs]
        for contour in contours:
            # replace the current shape_layer's data with the new contour
            napari_vertices = [[self.sc.timeframe] + list(point) for point in contour]
            napari_vertices = np.array(napari_vertices)
            new_shape_data.append((napari_vertices, "polygon"))

        self.shape_layer.data = []
        self.shape_layer.add(new_shape_data, shape_type=["polygon"])
        print("csn_correct_seg_callback done!")

    def clear_sc_layer_callback(self):
        self.shape_layer.data = []
        print("clear_sc_layer_callback done!")

    def restore_sc_contour_callback(self):
        self.update_shape_layer_by_sc()
        print("restore_sc_contour_callback done!")

    def filter_cells_by_size_callback(self, min_size, max_size):
        print("filter_cells_by_size_callback fired!")
        contours = self._get_contours_from_shape_layer(self.shape_layer)
        required_contours = filter_contours_by_size(contours, min_size, max_size)
        time = self.sc.timeframe
        new_shape_data = []
        for contour in required_contours:
            napari_vertices = [[time] + list(point) for point in contour]
            napari_vertices = np.array(napari_vertices)
            new_shape_data.append((napari_vertices, "polygon"))
        self.shape_layer.data = []
        self.shape_layer.add(new_shape_data, shape_type=["polygon"])
        print("filter_cells_by_size_callback done!")

    def focus_on_sc_callback(self):
        print("focus_on_sc_callback fired!")
        self.viewer.dims.set_point(0, self.sc.timeframe)
        print("focus_on_sc_callback done!")

    @staticmethod
    def resample_contour(contour, sample_num=50, start_idx=None):
        contour = np.asarray(contour, dtype=float)
        if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 3:
            main_info("The contour has fewer than 3 valid points; skip resampling.")
            return contour
        if start_idx is None:
            start_idx = np.random.randint(0, len(contour))
        if start_idx >= len(contour):
            main_info("The contour is empty or the start_idx is out of range.")
            return contour

        # rotate contour so that the start_idx is at the beginning
        contour = np.roll(contour, -start_idx, axis=0)

        sample_num = max(int(sample_num), 3)
        slice_step = int(len(contour) / sample_num)
        slice_step = max(slice_step, 1)  # make sure slice_step is at least 1
        if sample_num is not None:
            contour = contour[::slice_step]
        return contour

    def resample_contours_callback(self, sample_num):
        print("resample_contours_callback fired!")
        contours = self._get_contours_from_shape_layer(self.shape_layer)
        resampled_contours = []
        for contour in contours:
            resampled = self.resample_contour(contour, sample_num=sample_num)
            if self._is_stable_polygon_coords(resampled):
                resampled_contours.append(resampled)
            else:
                main_warning("Skipping an invalid contour during resampling.")
        time = self.sc.timeframe
        new_shape_data = []
        for contour in resampled_contours:
            napari_vertices = [[time] + list(point) for point in contour]
            napari_vertices = np.array(napari_vertices)
            new_shape_data.append((napari_vertices, "polygon"))
        if len(new_shape_data) == 0:
            main_warning("No valid contours remain after resampling; keeping the current shape layer unchanged.")
            return
        self.shape_layer.data = []
        self.shape_layer.add(new_shape_data, shape_type=["polygon"])

        # select the newly added last contour

        # The contours may all be empty? (double check) so we need to check the length first
        if len(self.shape_layer.data) > 0:
            self.shape_layer.selected_data = [len(self.shape_layer.data) - 1]
        print("resample_contours_callback done!")

    def close(self):
        # remove the shaper layer
        self.viewer.layers.remove(self.shape_layer)
        # self.magicgui_container.hide()
        # self.magicgui_container.close()
        if self.magicgui_container is not None:
            try:
                self.viewer.window.remove_dock_widget(self.magicgui_container.native)
            except Exception as e:
                main_warning("Exception when removing dock widget:", e)
        self.notify_sct_to_remove_sc_operator()


def create_sc_seg_napari_ui(sc_operator: ScSegOperator):
    """Usage
    # viewer = napari.view_image(dic_dataset.to_dask(), name="dic_image", cache=True)
    # shape_layer = NapariVisualizer.viz_trajectories(traj_collection, viewer, contour_sample_num=20)
    # sct_operator = SctOperator(traj_collection, shape_layer, viewer)
    # sct_operator.setup_shape_layer(shape_layer)

    Parameters
    ----------
    sct_operator : SctOperator
        _description_
    """

    @magicgui(call_button="save seg to sc")
    def save_seg_to_sc():
        print("[button] save callback fired!")
        sc_operator.save_seg_callback()

    @magicgui(call_button="auto correct seg")
    def csn_correct_seg(
        threshold: Annotated[float, {"widget_type": "FloatSpinBox", "max": int(1e4)}] = 0.5,
        padding: Annotated[int, {"widget_type": "SpinBox", "max": int(1e4)}] = 50,
    ):
        print("[button] csn callback fired!")
        main_info("csn output threshold:" + str(threshold), indent_level=2)
        sc_operator.csn_correct_seg_callback(threshold=threshold, padding_pixels=padding)

    @magicgui(
        auto_call=True,
        mode={"choices": ["segmentation"]},
    )
    def switch_mode_widget(mode):
        print("switch mode callback fired!")
        print("mode changed to", mode)
        if mode == "segmentation":
            sc_operator.mode = sc_operator.MANUAL_CORRECT_SEG_MODE
        sc_operator.show_selected_mode_widget()

    @magicgui(call_button="clear sc layer")
    def clear_sc_layer():
        print("[button] clear sc layer callback fired!")
        sc_operator.clear_sc_layer_callback()

    @magicgui(call_button="restore sc contour")
    def restore_sc_contour():
        print("[button] restore sc contour callback fired!")
        sc_operator.restore_sc_contour_callback()

    @magicgui(call_button="filter cells by size")
    def filter_cells_by_size(
        lower: Annotated[int, {"widget_type": "SpinBox", "max": int(1e6)}] = 100,
        upper: Annotated[int, {"widget_type": "SpinBox", "max": int(1e6)}] = 100000,
    ):
        print("[button] filter cells by size callback fired!")
        sc_operator.filter_cells_by_size_callback(lower, upper)

    @magicgui(call_button="focus on sc")
    def focus_on_sc():
        print("[button] focus on sc callback fired!")
        sc_operator.focus_on_sc_callback()

    @magicgui(call_button=None)
    def show_sc_id(sc_id="No SC"):
        return

    @magicgui(call_button="resample contours")
    def resample_contours(sample_num: Annotated[int, {"widget_type": "SpinBox", "max": int(1e6)}] = 15):
        print("[button] resample contours callback fired!")
        sc_operator.resample_contours_callback(sample_num)

    def on_close_callback():
        print("on_close_callback fired!")

    container = Container(
        widgets=[
            show_sc_id,
            switch_mode_widget,
            save_seg_to_sc,
            csn_correct_seg,
            clear_sc_layer,
            restore_sc_contour,
            filter_cells_by_size,
            focus_on_sc,
            resample_contours,
        ],
        labels=False,
    )
    container.native.setParent(None)
    container.native.deleteLater = lambda: on_close_callback()
    show_sc_id.sc_id.value = str(sc_operator.sc.id)[:12] + "-..."
    # hide call button
    show_sc_id.call_button.hide()
    sc_operator.magicgui_container = container
    sc_operator.hide_function_widgets()
    sc_operator.show_selected_mode_widget()
    sc_operator.viewer.window.add_dock_widget(container, name="Sc Operator")
