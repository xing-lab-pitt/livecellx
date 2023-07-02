import copy
from functools import partial
from typing import List, Optional, Union
import numpy as np

import magicgui as mgui
from magicgui import magicgui
from magicgui.widgets import Container, PushButton, Widget, create_widget
from napari.layers import Shapes
from pathlib import Path

from livecell_tracker.core.sc_seg_operator import ScSegOperator, create_sc_seg_napari_ui
from livecell_tracker.core.single_cell import SingleCellTrajectoryCollection, SingleCellStatic, SingleCellTrajectory
from livecell_tracker.livecell_logger import main_warning, main_info


class SctOperator:
    CONNECT_MODE = 0
    DISCONNECT_MODE = 1
    ADD_MOTHER_DAUGHER_MODE = 2
    DELETE_TRAJECTORY_MODE = 3
    ANNOTATE_CLICK_MODE = 4

    def __init__(
        self,
        traj_collection: SingleCellTrajectoryCollection,
        shape_layer: Shapes,
        viewer,
        operator="connect",
        magicgui_container=None,
        sc_operators=None,
        img_dataset=None,
        time_span=None,
        meta=None,
    ):
        self.select_info = []  # [cur_sct, cur_sc, selected_shape_index]
        self.operator = operator
        self.shape_layer: Optional[Shapes] = shape_layer
        self.setup_shape_layer(shape_layer)
        self.traj_collection = traj_collection
        self.viewer = viewer
        self.magicgui_container = magicgui_container
        self.mode = SctOperator.CONNECT_MODE
        self.annotate_click_samples = {}
        if sc_operators is None:
            sc_operators = []
        self.sc_operators = sc_operators
        self.img_dataset = img_dataset
        self.time_span = time_span
        if meta is None:
            self.meta = {}
        else:
            self.meta = meta

    def remove_sc_operator(self, sc_operator):
        self.sc_operators.remove(sc_operator)

    def clear_sc_opeartors(self):
        # the close method changes the length of the list, so we need to make a copy
        cur_sc_operators = list(self.sc_operators)
        for sc_operator in cur_sc_operators:
            print("clearing sc operator: ", sc_operator)
            sc_operator.close()

        # # explicitly clear the list is not necessary
        # # sc_opeartor close should remove itself from the list
        # self.sc_operators = []
        if len(self.sc_operators) != 0:
            main_warning("sc_operators not empty after clear_sc_operators (should be done via sc opeartor close)")

    def close(self):
        self.viewer.layers.remove(self.shape_layer)
        if self.magicgui_container is not None:
            try:
                self.viewer.window.remove_dock_widget(self.magicgui_container.native)
            except Exception as e:
                main_warning("[SctOperator] Exception when removing dock widget:", e)

    def get_all_scs(self):
        """Return all single cell objects in the current trajec_collection"""
        all_scts = self.traj_collection.get_all_trajectories()
        all_scs = []
        for sct in all_scts:
            all_scs.extend(sct.get_all_scs())
        return all_scs

    def selected_scs(self):
        cur_properties = self.shape_layer.current_properties
        return cur_properties["sc"]

    def select_shape(self, event, shape_layer=None):
        """Select a shape in the shape layer, and update the shape color and status.
        self.select_info consists of [cur_sct, cur_sc, selected_shape_index]"""
        if shape_layer is None:
            shape_layer = self.shape_layer
        print("current shape layer shape properties: ", event)
        current_properties = shape_layer.current_properties
        assert len(current_properties["sc"]) == 1 and len(current_properties["track_id"]) == 1
        if len(shape_layer.selected_data) > 1:
            print("Please select only one shape at a time for connecting trajectories")
            return
        if len(shape_layer.selected_data) == 0:
            print("No shape selected, please select a shape to connect trajectories")
            return
        selected_shape_index = list(shape_layer.selected_data)[0]

        shape_indices_in_select_info = list([info[2] for info in self.select_info])

        if selected_shape_index in shape_indices_in_select_info:
            # Skip if the shape is already selected
            print("shape already selected, please select another shape")
            return

            # Deselect the shape if it is already selected
            # TODO: deselect selected_shape_index. The following code works but with a small issue that when clicking on a shape for the first time, the selection will blink (select and then deselect instead of select)
            # print("deselecting shape...")
            # tmp_idx = shape_indices_in_select_info.index(selected_shape_index)
            # sct, sc, _ = self.select_info.pop(tmp_idx)
            # tmp_face_color = list(self.shape_layer.face_color)
            # tmp_face_color[selected_shape_index] = self.original_face_colors[selected_shape_index]
            # self.shape_layer.face_color = tmp_face_color
            # # self.shape_layer.properties["status"][selected_shape_index] = "unselected"
            # tmp_properties = dict(self.shape_layer.properties)
            # tmp_properties["status"][selected_shape_index] = ""
            # self.shape_layer.properties = tmp_properties
            # print("<select_info> complete deselecting track:", sct.track_id)
            # return

        cur_sc = current_properties["sc"][0]
        cur_track_id = current_properties["track_id"][0]
        cur_sct = self.traj_collection[cur_track_id]

        print("setting face color of selected shape...")
        if self.mode == self.CONNECT_MODE:
            selection_face_color = (1, 0, 0, 1)
            selection_status_text = "connect"
        elif self.mode == self.DISCONNECT_MODE:
            selection_face_color = (0, 1, 0, 1)
            selection_status_text = "disconnect"
        elif self.mode == self.ADD_MOTHER_DAUGHER_MODE:
            print("len of select_info", len(self.select_info))
            if len(self.select_info) == 0:
                selection_face_color = (1, 0, 0, 1)
                selection_status_text = "mother"
            else:
                selection_face_color = (0, 0, 1, 1)
                selection_status_text = "daughter"
        elif self.mode == self.DELETE_TRAJECTORY_MODE:
            selection_face_color = (0, 0, 0, 1)
            selection_status_text = "delete?"
        elif self.mode == self.ANNOTATE_CLICK_MODE:
            selection_face_color = (102 / 255, 179 / 255, 1, 1)
            selection_status_text = "selected"

        face_colors = list(shape_layer.face_color)
        face_colors[selected_shape_index] = selection_face_color
        shape_layer.face_color = face_colors

        properties = shape_layer.properties.copy()
        properties["status"][selected_shape_index] = selection_status_text
        shape_layer.properties = properties
        shape = shape_layer.data[selected_shape_index]
        # slice_index = viewer.dims.current_step[0]
        self.select_info.append((cur_sct, cur_sc, selected_shape_index))
        print("<selection complete>")
        return cur_sct, cur_sc, selected_shape_index

    def update_shape_layer(self, shape_index, track_id, sc, face_color):
        properties = self.shape_layer.properties
        properties["track_id"][shape_index] = track_id
        properties["sc"][shape_index] = sc
        properties["status"][shape_index] = ""
        self.shape_layer.properties = properties
        face_colors = list(self.shape_layer.face_color)
        face_colors[shape_index] = face_color
        self.shape_layer.face_color = face_colors

    def update_shape_layer_by_track_id(self, track_id, face_color, new_track_id):
        properties = self.shape_layer.properties.copy()
        face_colors = list(self.shape_layer.face_color)
        new_track_ids = properties["track_id"].copy()
        for shape_index in range(len(self.shape_layer.properties["track_id"])):
            if properties["track_id"][shape_index] == track_id:
                sc = properties["sc"][shape_index]
                new_track_ids[shape_index] = new_track_id
                face_colors[shape_index] = face_color
        properties["track_id"] = new_track_ids
        self.shape_layer.properties = properties
        self.shape_layer.face_color = face_colors
        print("<update track_id properties complete>")

    def lookup_sc_shape_index(self, sc) -> Optional[int]:
        properties = self.shape_layer.properties
        scs = properties["sc"]
        update_shape_index = None
        for shape_index, tmp_sc in enumerate(scs):
            if tmp_sc == sc and update_shape_index is not None:
                main_warning("multiple sc with the same sc object found in shape layer")
            if tmp_sc.id == sc.id:
                update_shape_index = shape_index
            if tmp_sc.id == sc.id and tmp_sc != sc:
                main_warning("sc with same id but different shape found in shape layer")
        return update_shape_index

    def update_shape_layer_by_sc(self, sc: SingleCellStatic):
        print("<update shape layer by sc>")

        # clear selected data first because adding/deleting shapes will change the shape index
        self.clear_selection()

        update_shape_index = self.lookup_sc_shape_index(sc)
        if update_shape_index is None:
            main_warning("sc not found in shape layer")
            return

        # update the sc's shape data in self.shape_layer
        # Note: the following line triggers self.select_shape
        self.shape_layer.selected_data = {update_shape_index}
        self.clear_selection()
        # update_shape_properties = dict(self.shape_layer.current_properties)
        cur_sc_properties = dict(self.shape_layer.properties)
        cur_sc_properties = {key: [value[update_shape_index]] for key, value in cur_sc_properties.items()}

        self.shape_layer.remove_selected()
        sc_napari_data = np.array(sc.get_napari_shape_contour_vec())

        # TODO: optimize the code below and figure out why the code below is slow in Napari UI
        # TODO: double check shape_layer.add does not support "properties=?" arg?
        self.shape_layer.add([sc_napari_data], shape_type="polygon")  # , properties=update_shape_properties)

        # TODO: double check if new shape index is always the last one
        new_shape_index = len(self.shape_layer.data) - 1
        assert new_shape_index is not None, "new shape index is None"
        properties = dict(self.shape_layer.properties)
        for key in properties.keys():
            properties[key][new_shape_index] = cur_sc_properties[key][0]
        self.shape_layer.properties = properties

        # # Deprecated code below; rollback if required
        # # simply update all the data
        # shape_data = list(self.shape_layer.data)
        # shape_data[update_shape_index] = np.array(sc.get_napari_shape_contour_vec())
        # print("<setting shapes...>")
        # self.shape_layer.data = shape_data
        self.store_shape_layer_info()
        print("<update shape layer by sc complete>")

    def connect_two_scts(self):
        assert len(self.select_info) == 2, "Please select two shapes to connect."
        sct1, sc1, shape_index1 = self.select_info[0]
        sct2, sc2, shape_index2 = self.select_info[1]
        if sct1 == sct2:
            print("Skipping connecting two shapes from the same trajectory...")
            return
        print("connecting two shapes from different trajectories...")
        sct1_span = sct1.get_timeframe_span()
        sct2_span = sct2.get_timeframe_span()

        if sct1_span[1] < sct2_span[0] or sct2_span[1] < sct1_span[0]:
            res_traj = sct1.copy()
            res_traj.add_nonoverlapping_sct(sct2)
            self.traj_collection.pop_trajectory(sct1.track_id)
            self.traj_collection.pop_trajectory(sct2.track_id)
            self.traj_collection.add_trajectory(res_traj)

            # self.viewer.layers.remove(self.shape_layer)
            # self.shape_layer = NapariVisualizer.viz_trajectories(self.traj_collection, self.viewer, contour_sample_num=20)
            # self.setup_shape_layer(self.shape_layer)
            new_face_color = self.original_face_colors[shape_index1]
            self.clear_selection()
            self.update_shape_layer_by_track_id(
                sct1.track_id, face_color=new_face_color, new_track_id=res_traj.track_id
            )
            self.update_shape_layer_by_track_id(
                sct2.track_id, face_color=new_face_color, new_track_id=res_traj.track_id
            )
            self.store_shape_layer_info()

        else:
            raise NotImplementedError("Two trajectories are overlapping, notImplemented for now...")
        print("connect operation complete!")

    def clear_selection(self):
        print("clearing selection...")
        self.select_info = []
        self.shape_layer.face_color = list(self.original_face_colors)
        self.shape_layer.properties = self.original_properties
        print("<clear complete>")

    # TODO: remove_scs not fully tested in notebook
    def remove_scs(self, scs: List[SingleCellStatic]):
        remove_shape_indices = []
        for sc in scs:
            shape_index = self.lookup_sc_shape_index(sc)
            if shape_index is None:
                continue
            remove_shape_indices.append(shape_index)
        remove_shape_indices = list(set(remove_shape_indices))
        remove_shape_indices = sorted(remove_shape_indices, reverse=True)
        self.shape_layer.selected_data = remove_shape_indices
        self.shape_layer.remove_selected()
        self.clear_selection()

        for shape_index in remove_shape_indices:
            self.original_face_colors.pop(shape_index)
            for key in self.original_properties.keys():
                self.original_properties[key].pop(shape_index)

    # TODO: remove_empty_contour_sct not fully tested
    def remove_empty_contour_sct(self):
        remove_tids = []
        remove_scs = []
        for tid, sct in self.traj_collection:
            assert len(sct.get_all_scs()) == 1, "sct should only have one sc when you call this function"
            sc = sct.get_all_scs()[0]
            if len(sc.contour) == 0:
                remove_tids.append(tid)
                remove_scs.append(sc)
        for id in remove_tids:
            main_info(f"removing empty contour sct with id {id}")
            self.traj_collection.pop_trajectory(id)

    def setup_shape_layer(self, shape_layer: Shapes):
        self.shape_layer = shape_layer
        shape_layer.events.current_properties.connect(self.select_shape)
        self.store_shape_layer_info()

    def store_shape_layer_info(self, update_slice=slice(0, None, 1)):
        # check if original_face_colors is initialized
        if not hasattr(self, "original_face_colors"):
            self.original_face_colors = copy.deepcopy(list(self.shape_layer.face_color))
        if not hasattr(self, "original_scs"):
            self.original_scs = list(self.shape_layer.properties["sc"])
        if not hasattr(self, "original_properties"):
            self.original_properties = copy.deepcopy(self.shape_layer.properties.copy())
        if not hasattr(self, "original_shape_data"):
            self.original_shape_data = copy.deepcopy(self.shape_layer.data.copy())

        # w/o deepcopy, the original_face_colors will be changed when shape_layer.face_color is changed...
        self.original_face_colors[update_slice] = copy.deepcopy(list(self.shape_layer.face_color))[update_slice]
        # Do not save the deep copied version of the single cells! We just keep one copy of the single cells in the shape layer.
        self.original_scs[update_slice] = list(self.shape_layer.properties["sc"])[update_slice]
        for key in self.original_properties.keys():
            self.original_properties[key][update_slice] = copy.deepcopy(self.shape_layer.properties.copy())[key][
                update_slice
            ]
        self.original_shape_data[update_slice] = copy.deepcopy(self.shape_layer.data.copy())[update_slice]
        self.original_properties["sc"][update_slice] = self.original_scs[update_slice]

    def restore_shapes_data(self):
        print("<restoring sct shapes>")
        self.shape_layer.data = self.original_shape_data
        self.shape_layer.properties = self.original_properties
        self.shape_layer.face_color = self.original_face_colors
        print("<restoring sct shapes complete>")

    def disconnect_sct(self):
        assert len(self.select_info) == 1, "Please select one shape to disconnect."
        sct, sc, old_shape_index = self.select_info[0]
        print("disconnecting shape...")
        old_traj = self.traj_collection.pop_trajectory(sct.track_id)
        new_sct1, new_sct2 = old_traj.split(sc.timeframe)
        self.traj_collection.add_trajectory(new_sct1)
        self.traj_collection.add_trajectory(new_sct2)

        color_1, color_2 = self.original_face_colors[old_shape_index], self.original_face_colors[old_shape_index]
        new_span_1 = new_sct1.get_timeframe_span()
        new_span_2 = new_sct2.get_timeframe_span()

        old_track_id = sct.track_id

        # obtain all shapes belonged to old trajectory
        old_track_shape_indices = []
        for i in range(len(self.shape_layer.properties["track_id"])):
            if self.shape_layer.properties["track_id"][i] == old_track_id:
                old_track_shape_indices.append(i)

        # update the shapes belonged to the new trajectory 1
        mutable_properties = self.shape_layer.properties.copy()
        mutable_face_colors = list(self.shape_layer.face_color)
        traj1_shape_indices = [idx for idx in old_track_shape_indices if idx < old_shape_index]
        for shape_index in traj1_shape_indices:
            mutable_properties["track_id"][shape_index] = new_sct1.track_id
            mutable_face_colors[shape_index] = color_1
        # update the shapes belonged to the new trajectory 1
        traj2_shape_indices = [idx for idx in old_track_shape_indices if idx >= old_shape_index]
        for shape_index in traj2_shape_indices:
            mutable_properties["track_id"][shape_index] = new_sct2.track_id
            mutable_face_colors[shape_index] = color_2

        mutable_properties["status"][old_shape_index] = ""
        self.shape_layer.properties = mutable_properties
        self.shape_layer.face_color = mutable_face_colors

        # # slow version below by removing and adding the entire shape layer
        # self.viewer.layers.remove(self.shape_layer)
        # self.shape_layer = NapariVisualizer.viz_trajectories(self.traj_collection, self.viewer, contour_sample_num=20)
        # self.setup_shape_layer(self.shape_layer)

        self.store_shape_layer_info()
        self.clear_selection()
        print("<disconnect operation complete>")

    def add_mother_daughter_relation(self):
        assert len(self.select_info) >= 2, "Please select >2 shapes to add mother daughter relation."
        mother_sct, mother_sc, mother_shape_index = self.select_info[0]
        for i in range(1, len(self.select_info)):
            daughter_sct, daughter_sc, daughter_shape_index = self.select_info[i]
            assert mother_sct != daughter_sct, "mother and daughter cannot be from the same trajectory!"
            mother_sct.add_daughter(daughter_sct)
            daughter_sct.add_mother(mother_sct)
        self.clear_selection()
        print("<add mother-daughter relation operation complete>")

    def delete_selected_sct(self):
        # sct, sc, shape_index = self.select_info[0]
        selected_track_ids = [sct.track_id for sct, sc, shape_index in self.select_info]
        print("deleting shape...")
        selected_track_id_set = set(selected_track_ids)
        # remove all the shapes with track_id == sct.track_id
        self.shape_layer.selected_data = []
        for i in range(len(self.shape_layer.properties["track_id"]) - 1, -1, -1):
            if self.shape_layer.properties["track_id"][i] in selected_track_id_set:
                self.shape_layer.selected_data.add(i)
        self.shape_layer.remove_selected()
        for track_id in selected_track_ids:
            self.traj_collection.pop_trajectory(track_id)
        self.store_shape_layer_info()
        self.clear_selection()
        print("<delete operation complete>")

    def annotate_click(self, label):
        print("<annotating click>: adding a sample")
        sample = []
        for selected_shape in self.select_info:
            sct, sc, shape_index = selected_shape
            sample.append(sc)
            if "_labels" not in sc.meta:
                sc.meta["_labels"] = []
            sc.meta["_labels"].append(label)
        if label not in self.annotate_click_samples:
            self.annotate_click_samples[label] = []
        self.annotate_click_samples[label].append(sample)
        self.clear_selection()
        print("<annotate click operation complete>")

    def edit_selected_sc(self):
        # get the selected shape
        current_properties = self.shape_layer.current_properties
        if len(current_properties) == 0:
            main_warning("Please select a shape to edit its properties.")
            return
        if len(current_properties) > 1:
            main_warning("More than one shape is selected. The first selected shape is used for editing.")
        cur_sc = current_properties["sc"][0]
        sc_operator = self.edit_sc(cur_sc)

        # hide the shape layer
        self.shape_layer.visible = False
        return sc_operator

    def edit_sc(self, cur_sc):
        sc_operator = ScSegOperator(cur_sc, viewer=self.viewer, create_sc_layer=True, sct_observers=[self])
        create_sc_seg_napari_ui(sc_operator)
        self.sc_operators.append(sc_operator)
        return sc_operator

    def toggle_shapes_text(self):
        self.shape_layer.text.visible = not self.shape_layer.text.visible

    def save_annotations(
        self,
        sample_out_dir: Union[Path, str],
        filename_pattern: str = "sample_{sample_index}.json",
        sample_dataset_dir: Optional[Union[Path, str]] = None,
    ):
        print("<saving annotations>")
        if isinstance(sample_out_dir, str):
            sample_out_dir = Path(sample_out_dir)
        sample_out_dir.mkdir(exist_ok=True)
        if sample_dataset_dir is None:
            sample_dataset_dir = sample_out_dir / "datasets"
        elif isinstance(sample_dataset_dir, str):
            sample_dataset_dir = Path(sample_dataset_dir)
        sample_paths = []

        for label in self.annotate_click_samples:
            samples = self.annotate_click_samples[label]
            label_dir: Path = sample_out_dir / label
            label_dir.mkdir(exist_ok=True)
            for i, sample in enumerate(samples):
                sample_json_path = label_dir / (filename_pattern.format(sample_index=i))
                SingleCellStatic.write_single_cells_json(sample, sample_json_path, dataset_dir=sample_dataset_dir)
                sample_paths.append(sample_json_path)
        print("<saving annotations complete>")
        return sample_paths

    def add_new_sc(self):
        """Adds a new single cell to a single cell trajectory."""
        print("<adding new sc>")
        assert self.time_span is not None, "Please set the time span first."
        min_time = self.time_span[0]
        # min_time = 0 # TODO: if we regulate that img_dataset is always used, then we can use this line
        cur_time = self.viewer.dims.current_step[0] + min_time
        new_sc = SingleCellStatic(timeframe=cur_time, contour=[], img_dataset=self.img_dataset)
        sc_operator = self.edit_sc(new_sc)

        # add a new sct to sctc
        new_sct = SingleCellTrajectory(
            track_id=self.traj_collection._next_track_id(),
            img_dataset=self.img_dataset,
        )
        new_sct.add_sc(new_sc.timeframe, new_sc)
        self.traj_collection.add_trajectory(new_sct)
        new_sct.add_sc(new_sc.timeframe, new_sc)

        # create a dummy shape for the new sc in the shape layer
        old_layer_properties = self.shape_layer.properties
        new_sc_layer_sc_properties = list(old_layer_properties["sc"]) + [new_sc]
        new_sc_layer_track_properties = list(old_layer_properties["track_id"]) + [new_sct.track_id]
        new_sc_layer_status_properties = list(old_layer_properties["status"]) + [""]
        new_sc_layer_properties = {
            "sc": new_sc_layer_sc_properties,
            "track_id": new_sc_layer_track_properties,
            "status": new_sc_layer_status_properties,
        }
        sc_dummy_napari_data = [np.array([[new_sc.timeframe, -50, -50], [new_sc.timeframe, -10, -10]])]
        # self.shape_layer.data = list(self.shape_layer.data) + sc_napari_data
        self.shape_layer.add(sc_dummy_napari_data, shape_type="polygon")
        self.shape_layer.properties = new_sc_layer_properties

        # WARNING: only update the newly added sc's shape layer info
        # because it will cause problems e.g. other function status staying forever on the shape layer
        self.original_face_colors.append(self.original_face_colors[0])  # TODO: randomly generate a color?

        self.original_properties["sc"] = np.append(self.original_properties["sc"], new_sc)
        self.original_properties["track_id"] = np.append(self.original_properties["track_id"], new_sct.track_id)
        self.original_properties["status"] = np.append(self.original_properties["status"], "")

        self.original_scs.append(new_sc)
        self.original_shape_data.append(sc_dummy_napari_data[0])
        self.store_shape_layer_info(update_slice=slice(-1, None, None))
        print("<adding new sc complete>")
        return sc_operator

    def hide_function_widgets(self):
        # Always show the first two widgets
        for i in range(2, len(self.magicgui_container)):
            self.magicgui_container[i].hide()

    def show_selected_mode_widget(self):

        # Always show the edit selected sc widget (7th)
        self.magicgui_container[7].show()
        # Always show restore_sct_shapes (8th)
        self.magicgui_container[8].show()
        # Always show toggle_shapes_text (9th)
        self.magicgui_container[9].show()
        # Always show clear sc operators (10th)
        self.magicgui_container[10].show()
        # Always show add new sc (11th)
        self.magicgui_container[11].show()

        if self.mode == self.CONNECT_MODE:
            self.magicgui_container[2].show()
        elif self.mode == self.DISCONNECT_MODE:
            self.magicgui_container[3].show()
        elif self.mode == self.ADD_MOTHER_DAUGHER_MODE:
            self.magicgui_container[4].show()
        elif self.mode == self.DELETE_TRAJECTORY_MODE:
            self.magicgui_container[5].show()
        elif self.mode == self.ANNOTATE_CLICK_MODE:
            self.magicgui_container[6].show()
        else:
            raise ValueError("Invalid mode!")


def create_sct_napari_ui(sct_operator: SctOperator):
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

    @magicgui(call_button="connect")
    def connect_widget():
        print("connect callback fired!")
        sct_operator.connect_two_scts()

    @magicgui(call_button="clear selection")
    def clear_selection_widget():
        print("clear selection callback fired!")
        sct_operator.clear_selection()

    @magicgui(call_button="disconnect")
    def disconnect_widget():
        print("disconnect callback fired!")
        sct_operator.disconnect_sct()

    @magicgui(call_button="add mother/daughter relation")
    def add_mother_daughter_relation_widget():
        print("add mother/daughter relation callback fired!")
        sct_operator.add_mother_daughter_relation()

    @magicgui(call_button="delete trajectory")
    def delete_trajectory_widget():
        print("delete trajectory callback fired!")
        sct_operator.delete_selected_sct()

    @magicgui(call_button="click&annotate")
    def annotate_click_widget(label="mitosis"):
        print("annotate callback fired!")
        # sct_operator.delete_selected_sct()
        sct_operator.annotate_click(label=label)

    @magicgui(call_button="edit selected sc")
    def edit_selected_sc():
        print("edit sc fired!")
        # sct_operator.delete_selected_sct()
        sct_operator.edit_selected_sc()

    @magicgui(call_button="restore sct shapes")
    def restore_sct_shapes():
        print("restore sct shapes fired!")
        sct_operator.restore_shapes_data()

    @magicgui(call_button="toggle shapes text")
    def toggle_shapes_text():
        print("toggle shapes text fired!")
        sct_operator.toggle_shapes_text()

    @magicgui(call_button="clear sc operators")
    def clear_sc_operators():
        print("clear sc operators fired!")
        sct_operator.clear_sc_opeartors()

    @magicgui(call_button="add new sc")
    def add_new_sc():
        print("add new sc fired!")
        sct_operator.add_new_sc()

    @magicgui(
        auto_call=True,
        mode={
            "choices": ["connect", "disconnect", "add mother/daughter relation", "delete trajectory", "click&annotate"]
        },
    )
    def switch_mode_widget(mode):
        print("switch mode callback fired!")
        print("mode changed to", mode)
        if mode == "connect":
            sct_operator.mode = sct_operator.CONNECT_MODE
        elif mode == "disconnect":
            sct_operator.mode = sct_operator.DISCONNECT_MODE
        elif mode == "add mother/daughter relation":
            sct_operator.mode = sct_operator.ADD_MOTHER_DAUGHER_MODE
        elif mode == "delete trajectory":
            sct_operator.mode = sct_operator.DELETE_TRAJECTORY_MODE
        elif mode == "click&annotate":
            sct_operator.mode = sct_operator.ANNOTATE_CLICK_MODE
        sct_operator.hide_function_widgets()
        sct_operator.show_selected_mode_widget()
        sct_operator.clear_selection()

    container = Container(
        widgets=[
            switch_mode_widget,
            clear_selection_widget,
            connect_widget,
            disconnect_widget,
            add_mother_daughter_relation_widget,
            delete_trajectory_widget,
            annotate_click_widget,
            edit_selected_sc,
            restore_sct_shapes,
            toggle_shapes_text,
            clear_sc_operators,
            add_new_sc,
        ],
        labels=False,
    )

    sct_operator.magicgui_container = container
    sct_operator.hide_function_widgets()
    sct_operator.show_selected_mode_widget()
    sct_operator.viewer.window.add_dock_widget(container, name="SCT Operator")


def create_scts_operator_viewer(
    sctc: SingleCellTrajectoryCollection, img_dataset=None, viewer=None, time_span=None
) -> SctOperator:
    import napari
    from livecell_tracker.core.napari_visualizer import NapariVisualizer
    from livecell_tracker.core.single_cell import SingleCellTrajectoryCollection, SingleCellTrajectory

    if not (time_span is None):
        if img_dataset is None:
            # TODO: confirm and report the following issue to Napari
            main_warning(
                "img_dataset is None: a known bug may occur if at some point SingleCellTrajectory does not contain any shape. Napari is going to ignore the time point entirely and create one fewer slices in its data structure. This may mess up functionality in sctc operator"
            )
        new_scts = SingleCellTrajectoryCollection()
        for _, sct in sctc:
            new_scts.add_trajectory(sct.subsct(time_span[0], time_span[1]))
        sctc = new_scts

    # if the img_dataset is not None, then we can use it to determine the time span
    if img_dataset is not None:
        time_span = img_dataset.time_span()
    if viewer is None:
        if img_dataset is not None:
            viewer = napari.view_image(img_dataset.to_dask(), name="img_image", cache=True)
        else:
            viewer = napari.Viewer()

    shape_layer = NapariVisualizer.gen_trajectories_shapes(sctc, viewer, contour_sample_num=20)
    shape_layer.mode = "select"
    sct_operator = SctOperator(sctc, shape_layer, viewer, img_dataset=img_dataset, time_span=time_span)
    create_sct_napari_ui(sct_operator)
    return sct_operator


def create_scs_edit_viewer(
    single_cells: List[SingleCellStatic], img_dataset=None, viewer=None, time_span=None
) -> SctOperator:
    """
    Creates a viewer for editing SingleCellStatic objects.
    The single cells are stored in sct_operators, meaning when the users change the scs in the viewer, the changes will be reflected in the single cell list input.

    Args:
        single_cells (List[SingleCellStatic]): A list of SingleCellStatic objects to be edited.
        img_dataset (Optional): An optional image dataset to be displayed in the viewer.
        viewer (Optional): An optional napari viewer to be used for displaying the image dataset and shapes.
        time_span (Optional): An optional tuple of start and end timepoints to be displayed in the viewer.

    Returns:
        SctOperator: An instance of the SctOperator class for editing SingleCellStatic objects.
    """
    import napari
    from livecell_tracker.core.napari_visualizer import NapariVisualizer
    from livecell_tracker.core.single_cell import SingleCellTrajectoryCollection, SingleCellTrajectory

    # Create a temporary SingleCellTrajectoryCollection for editing the SingleCellStatic objects
    temp_sc_trajs_for_correct = SingleCellTrajectoryCollection()
    for idx, sc in enumerate(single_cells):
        sct = SingleCellTrajectory(track_id=idx, timeframe_to_single_cell={sc.timeframe: sc})
        temp_sc_trajs_for_correct.add_trajectory(sct)

    # Create an SctOperator instance for editing the SingleCellStatic objects
    sct_operator = create_scts_operator_viewer(temp_sc_trajs_for_correct, img_dataset, viewer, time_span)
    return sct_operator
