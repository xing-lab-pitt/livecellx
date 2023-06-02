import copy
from functools import partial
from typing import Optional, Union
from magicgui import magicgui
from magicgui.widgets import Container, PushButton, Widget, create_widget
from napari.layers import Shapes
from livecell_tracker.core.single_cell import SingleCellTrajectoryCollection, SingleCellStatic
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from napari.layers import Shapes

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

    def __init__(
        self,
        sc: SingleCellStatic,
        viewer,
        shape_layer: Optional[Shapes] = None,
        face_color=(0, 0, 1, 1),
        magicgui_container=None,
        csn_model=None,
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

        if not (self.shape_layer is None):
            self.setup_edit_contour_shape_layer()

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
        self.magicgui_container[0].show()
        self.magicgui_container[1].show()
        if self.mode == self.MANUAL_CORRECT_SEG_MODE:
            self.magicgui_container[2].show()
            self.magicgui_container[3].show()
            self.magicgui_container[4].show()
            self.magicgui_container[5].show()

    def hide_function_widgets(self):
        for i in range(2, len(self.magicgui_container)):
            self.magicgui_container[i].hide()

    def save_seg_callback(self):
        """Save the segmentation to the single cell object."""

        def _get_contour_from_shape_layer(layer: Shapes):
            """Get contour coordinates from a shape layer in napari."""
            vertices = np.array(layer.data[0])

            # ignore the first vertex, which is the slice index
            vertices = vertices[:, 1:3]
            return vertices

        # Get the contour coordinates from the shape layer
        contour = _get_contour_from_shape_layer(self.shape_layer)
        # Store the contour in the single cell object
        self.sc.contour = contour
        self.sc.update_bbox()

    def csn_correct_seg_callback(self, padding_pixels=50):
        print("csn_correct_seg_callback fired")
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

    @magicgui(call_button=None)
    def show_sc_id(sc_id="No SC"):
        return

    container = Container(
        widgets=[
            show_sc_id,
            switch_mode_widget,
            save_seg_to_sc,
            csn_correct_seg,
            clear_sc_layer,
            restore_sc_contour,
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
