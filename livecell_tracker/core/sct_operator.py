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
from livecell_tracker.livecell_logger import main_warning


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
    ):
        self.select_info = []  # [cur_sct, cur_sc, selected_shape_index]
        self.operator = operator
        self.shape_layer: Optional[Shapes] = shape_layer
        self.setup_shape_layer(shape_layer)
        self.traj_collection = traj_collection
        self.viewer = viewer
        self.magicgui_container = magicgui_container
        self.mode = SctOperator.CONNECT_MODE
        self.annotate_click_samples = []
        if sc_operators is None:
            sc_operators = []
        self.sc_operators = sc_operators
        self.img_dataset = img_dataset
        self.time_span = time_span

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

        shape_indices_in_select_info = set([info[2] for info in self.select_info])
        if selected_shape_index in shape_indices_in_select_info:
            print("shape already selected, please select another shape")
            return

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

    def update_shape_layer_by_sc(self, sc: SingleCellStatic):
        print("<update shape layer by sc>")
        properties = self.shape_layer.properties
        scs = properties["sc"]

        def lookup_sc_index(sc):
            update_shape_index = None
            for shape_index, tmp_sc in enumerate(scs):
                if tmp_sc.id == sc.id:
                    update_shape_index = shape_index
                if tmp_sc.id == sc.id and tmp_sc != sc:
                    main_warning("sc with same id but different shape found in shape layer")
            return update_shape_index

        update_shape_index = lookup_sc_index(sc)
        if update_shape_index is None:
            main_warning("sc not found in shape layer")
            return

        # update the sc's shape data in self.shape_layer
        self.shape_layer.selected_data = {update_shape_index}
        self.shape_layer.remove_selected()
        sc_napari_data = np.array(sc.get_napari_shape_contour_vec())
        update_shape_properties = self.shape_layer.current_properties
        update_shape_properties["sc"] = [sc]

        # TODO: optimize the code below and figure out why the code below is slow in Napari UI
        # TODO: double check shape_layer.add does not support "properties=?" arg?
        self.shape_layer.add([sc_napari_data], shape_type="polygon")  # , properties=update_shape_properties)
        new_shape_index = lookup_sc_index(sc)
        properties = self.shape_layer.properties
        for key in properties.keys():
            properties[key][new_shape_index] = update_shape_properties[key][0]
        self.shape_layer.properties = properties

        # # Deprecated code below; rollback if required
        # # simply update all the data
        # shape_data = list(self.shape_layer.data)
        # shape_data[update_shape_index] = np.array(sc.get_napari_shape_contour_vec())
        # print("<setting shapes...>")
        # self.shape_layer.data = shape_data
        self.clear_selection()
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

    def setup_shape_layer(self, shape_layer: Shapes):
        self.shape_layer = shape_layer
        shape_layer.events.current_properties.connect(self.select_shape)
        self.store_shape_layer_info()

    def store_shape_layer_info(self):
        # w/o deepcopy, the original_face_colors will be changed when shape_layer.face_color is changed...
        self.original_face_colors = copy.deepcopy(list(self.shape_layer.face_color))
        # Do not save the deep copied version of the single cells! We just keep one copy of the single cells in the shape layer.
        self.original_scs = self.shape_layer.properties["sc"]
        self.original_properties = copy.deepcopy(self.shape_layer.properties.copy())
        self.original_shape_data = copy.deepcopy(self.shape_layer.data.copy())
        self.original_properties["sc"] = self.original_scs

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

    def annotate_click(self):
        print("<annotating click>: adding a sample")
        sample = []
        for selected_shape in self.select_info:
            sct, sc, shape_index = selected_shape
            sample.append(sc)
        self.annotate_click_samples.append(sample)
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
        return sc_operator

    def edit_sc(self, cur_sc):
        sc_operator = ScSegOperator(cur_sc, viewer=self.viewer, create_sc_layer=True, sct_observers=[self])
        create_sc_seg_napari_ui(sc_operator)
        self.sc_operators.append(sc_operator)
        return sc_operator

    def restore_shapes_data(self):
        print("<restoring sct shapes>")
        self.shape_layer.data = self.original_shape_data
        self.shape_layer.properties = self.original_shape_properties
        self.shape_layer.face_color = self.original_shape_face_color
        print("<restoring sct shapes complete>")

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

        samples = self.annotate_click_samples
        sample_paths = []

        for i, sample in enumerate(samples):
            sample_json_path = sample_out_dir / (filename_pattern.format(sample_index=i))
            SingleCellStatic.write_single_cells_json(sample, sample_json_path, dataset_dir=sample_dataset_dir)
            sample_paths.append(sample_json_path)
        print("<saving annotations complete>")
        return sample_paths

    def add_new_sc(self):
        """Adds a new single cell to a single cell trajectory."""
        print("<adding new sc>")
        assert self.time_span is not None, "Please set the time span first."
        min_time = self.time_span[0]
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
        self.store_shape_layer_info()
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
    def annotate_click_widget():
        print("annotate callback fired!")
        # sct_operator.delete_selected_sct()
        sct_operator.annotate_click()

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
    scts: SingleCellTrajectoryCollection, img_dataset=None, viewer=None, time_span=None
) -> SctOperator:
    import napari
    from livecell_tracker.core.napari_visualizer import NapariVisualizer
    from livecell_tracker.core.single_cell import SingleCellTrajectoryCollection, SingleCellTrajectory

    if time_span is None:
        if img_dataset is not None:
            sorted_times = img_dataset.get_sorted_times()
            time_span = (sorted_times[0], sorted_times[-1])
        else:
            # TODO: use scts' time span
            time_span = (0, np.inf)

    if viewer is None:
        if img_dataset is not None:
            viewer = napari.view_image(img_dataset.to_dask(), name="img_image", cache=True)
    else:
        viewer = napari.Viewer()
    shape_layer = NapariVisualizer.gen_trajectories_shapes(scts, viewer, contour_sample_num=20)
    shape_layer.mode = "select"

    sct_operator = SctOperator(scts, shape_layer, viewer, img_dataset=img_dataset, time_span=time_span)
    create_sct_napari_ui(sct_operator)
    return sct_operator


def create_scs_edit_viewer(single_cells: List[SingleCellStatic], img_dataset=None, viewer=None) -> SctOperator:
    import napari
    from livecell_tracker.core.napari_visualizer import NapariVisualizer
    from livecell_tracker.core.single_cell import SingleCellTrajectoryCollection, SingleCellTrajectory

    temp_sc_trajs_for_correct = SingleCellTrajectoryCollection()
    for idx, sc in enumerate(single_cells):
        sct = SingleCellTrajectory(track_id=idx, timeframe_to_single_cell={sc.timeframe: sc})
        temp_sc_trajs_for_correct.add_trajectory(sct)
    sct_operator = create_scts_operator_viewer(temp_sc_trajs_for_correct, img_dataset, viewer)
    return sct_operator
