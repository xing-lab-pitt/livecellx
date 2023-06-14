import copy
import cv2
from functools import partial
from typing import Optional, Union, Annotated
import magicgui as mgui
from magicgui import magicgui
from magicgui.widgets import Container, PushButton, Widget, create_widget
from napari.layers import Shapes
from livecell_tracker.core.single_cell import SingleCellTrajectoryCollection, SingleCellStatic
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from napari.layers import Shapes

from livecell_tracker.livecell_logger import main_info
from livecell_tracker.core import SingleCellTrajectory, SingleCellStatic
from livecell_tracker.segment.ou_utils import create_ou_input_from_sc
from livecell_tracker.segment.utils import find_contours_opencv
from livecell_tracker.core.datasets import SingleImageDataset


class ScSegOperator:
    """
    A class for performing segmentation on single cell images.

    Attributes
    ----------
    viewer : napari.Viewer
        The napari viewer.
    single_cell_static : SingleCellStatic
        The single cell static object.
    shape_layer : napari.layers.Shapes
        The napari shape layer for displaying the segmentation.
    """

    MANUAL_CORRECT_SEG_MODE = 0
    CSN_CORRECT_SEG_MODE = 1

    DEFAULT_CSN_MODEL = None

    @staticmethod
    def load_default_csn_model(path, cuda=True):
        import torch
        from livecell_tracker.model_zoo.segmentation.sc_correction import CorrectSegNet

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
        magicgui_container=None,
        csn_model=None,
        create_sc_layer=True,
        sct_observers: Optional[list] = None,
    ):
        """
        Parameters
        ----------
        viewer : napari.Viewer
            The napari viewer.
        single_cell_static : SingleCellStatic
            The single cell static object.
        """

        self.sc = sc
        self.viewer = viewer
        self.shape_layer = shape_layer
        self.face_color = face_color
        self.mode = self.MANUAL_CORRECT_SEG_MODE
        self.magicgui_container = magicgui_container
        self.csn_model = csn_model

        self.sct_observers = sct_observers
        if sct_observers is None:
            self.sct_observers = []

        if not (self.shape_layer is None):
            self.setup_edit_contour_shape_layer()

        if create_sc_layer:
            self.create_sc_layer()

    def create_sc_layer(self, name=None, contour_sample_num=100):
        if name is None:
            name = f"sc_{self.sc.id}"
        shape_vec = self.sc.get_napari_shape_contour_vec(contour_sample_num=contour_sample_num)
        properties = {"sc": [self.sc]}
        print("shape vec", shape_vec)
        shape_layer = self.viewer.add_shapes(
            [shape_vec],
            properties=properties,
            face_color=[self.face_color],
            shape_type="polygon",
            name=name,
        )
        self.shape_layer = shape_layer
        self.setup_edit_contour_shape_layer()
        print(">>> create sc layer done")

    def remove_sc_layer(self):
        if self.shape_layer is None:
            return
        self.viewer.layers.remove(self.shape_layer)
        self.shape_layer = None

    def update_shape_layer_by_sc(self, contour_sample_num=100):
        shape_vec = self.sc.get_napari_shape_contour_vec(contour_sample_num=contour_sample_num)
        self.shape_layer.data = [shape_vec]

    def correct_segment(
        self,
        model,
        create_ou_input_kwargs={
            "padding_pixels": 50,
            "dtype": float,
            "remove_bg": False,
            "one_object": True,
            "scale": 0,
        },
    ):
        import torch
        from torchvision import transforms

        #  padding_pixels=padding_pixels, dtype=dtype, remove_bg=remove_bg, one_object=one_object, scale=scale

        input_transforms = transforms.Compose(
            [
                transforms.Resize(size=(412, 412)),
            ]
        )
        temp_sc = self.sc.copy()
        new_contour = np.array(self.shape_layer.data[0])
        new_contour = new_contour[:, -2:]  # remove slice index (time)
        temp_sc.update_contour(new_contour)
        temp_sc.update_bbox()
        res_bbox = temp_sc.bbox
        ou_input = create_ou_input_from_sc(temp_sc, **create_ou_input_kwargs)
        # ou_input = create_ou_input_from_sc(self.sc, **create_ou_input_kwargs)
        original_shape = ou_input.shape

        ou_input = input_transforms(torch.tensor([ou_input]))
        ou_input = torch.stack([ou_input, ou_input, ou_input], dim=1)
        ou_input = ou_input.float().cuda()

        back_transforms = transforms.Compose(
            [
                transforms.Resize(size=(original_shape[0], original_shape[1])),
            ]
        )
        output = model(ou_input)
        output = back_transforms(output)
        return output, res_bbox

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

    @staticmethod
    def _get_contours_from_shape_layer(layer: Shapes):
        res_contours = []
        for shape in layer.data:
            vertices = np.array(shape)
            # ignore the first vertex, which is the slice index
            vertices = vertices[:, 1:3]
            res_contours.append(vertices)
        return res_contours

    def save_seg_callback(self):
        """Save the segmentation to the single cell object."""
        print("<save_seg_callback fired>")
        # Get the contour coordinates from the shape layer
        contours = self._get_contours_from_shape_layer(self.shape_layer)
        assert len(contours) > 0, "No contour is found in the shape layer."
        contour = contours[0]
        # Store the contour in the single cell object
        self.sc.update_contour(contour)
        print("<save_seg_callback finished>")

        # Notify the observers
        self.notify_sct_to_update()

    def csn_correct_seg_callback(self, padding_pixels=50):
        print("csn_correct_seg_callback fired")
        if self.csn_model is None and ScSegOperator.DEFAULT_CSN_MODEL is None:
            print("No CSN model is loaded. Please load a CSN model first.")
            return
        elif self.csn_model is None:
            print("Using default CSN model and loading it to the operator...")
            self.csn_model = ScSegOperator.DEFAULT_CSN_MODEL

        create_ou_input_kwargs = {
            "padding_pixels": padding_pixels,
            "dtype": float,
            "remove_bg": False,
            "one_object": True,
            "scale": 0,
        }
        output, res_bbox = self.correct_segment(self.csn_model, create_ou_input_kwargs=create_ou_input_kwargs)
        bin_mask = output[0].cpu().detach().numpy()[0] > 0.5
        contours = find_contours_opencv(bin_mask.astype(bool))
        # contour = [0]
        new_shape_data = []
        for contour in contours:
            contour_in_original_image = contour + res_bbox[:2] - padding_pixels
            # replace the current shape_layer's data with the new contour
            napari_vertices = [[self.sc.timeframe] + list(point) for point in contour_in_original_image]
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

        required_contours = []
        for contour in contours:
            contour = contour.astype(np.float32)
            area = cv2.contourArea(contour)
            print("area:", area)
            if area >= min_size and area <= max_size:
                required_contours.append(contour)

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
        if start_idx is None:
            start_idx = np.random.randint(0, len(contour))
        if len(contour) == 0 or start_idx > len(contour):
            main_info("The contour is empty or the start_idx is out of range.")
            return contour

        # rotate contour so that the start_idx is at the beginning
        contour = np.roll(contour, -start_idx, axis=0)

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
            resampled_contours.append(self.resample_contour(contour, sample_num=sample_num))
        time = self.sc.timeframe
        new_shape_data = []
        for contour in resampled_contours:
            napari_vertices = [[time] + list(point) for point in contour]
            napari_vertices = np.array(napari_vertices)
            new_shape_data.append((napari_vertices, "polygon"))
        self.shape_layer.data = []
        self.shape_layer.add(new_shape_data, shape_type=["polygon"])

        # select the newly added last contour

        # The contours may all be empty? (double check) so we need to check the length first
        if len(self.shape_layer.data) > 0:
            self.shape_layer.selected_data = [len(self.shape_layer.data) - 1]
        print("resample_contours_callback done!")


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
    def csn_correct_seg():
        print("[button] csn callback fired!")
        sc_operator.csn_correct_seg_callback()

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
    def resample_contours(sample_num: Annotated[int, {"widget_type": "SpinBox", "max": int(1e6)}] = 100):
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
    show_sc_id.sc_id.value = str(sc_operator.sc.id)[:12] + "-..."
    # hide call button
    show_sc_id.call_button.hide()
    sc_operator.magicgui_container = container
    sc_operator.hide_function_widgets()
    sc_operator.show_selected_mode_widget()
    sc_operator.viewer.window.add_dock_widget(container, name="Sc Operator")
