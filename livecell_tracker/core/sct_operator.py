import copy
from functools import partial
from typing import Optional
from magicgui import magicgui
from magicgui.widgets import Container, PushButton, Widget, create_widget
from napari.layers import Shapes
from livecell_tracker.core.single_cell import SingleCellTrajectoryCollection


class SctOperator:
    CONNECT_MODE = 0
    DISCONNECT_MODE = 1
    ADD_MOTHER_DAUGHER_MODE = 2

    def __init__(
        self,
        traj_collection: SingleCellTrajectoryCollection,
        shape_layer: Shapes,
        viewer,
        operator="connect",
        magicgui_container=None,
    ):
        self.select_info = []  # [cur_sct, cur_sc, selected_shape_index]
        self.operator = operator
        self.shape_layer: Optional[Shapes] = shape_layer
        self.setup_shape_layer(shape_layer)
        self.traj_collection = traj_collection
        self.viewer = viewer
        self.magicgui_container = magicgui_container
        self.mode = SctOperator.CONNECT_MODE

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
        self.original_face_colors = copy.deepcopy(list(shape_layer.face_color))
        self.original_properties = copy.deepcopy(shape_layer.properties.copy())

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

    def hide_function_widgets(self):
        for i in range(2, len(self.magicgui_container)):
            self.magicgui_container[i].hide()

    def show_selected_mode_widget(self):
        if self.mode == self.CONNECT_MODE:
            self.magicgui_container[2].show()
        elif self.mode == self.DISCONNECT_MODE:
            self.magicgui_container[3].show()
        elif self.mode == self.ADD_MOTHER_DAUGHER_MODE:
            self.magicgui_container[4].show()
        else:
            raise ValueError("Invalid mode!")
