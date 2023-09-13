import os
import pandas as pd
from datetime import datetime
from typing import Optional, Union, Callable
from livecell_tracker.livecell_logger import main_info, main_warning, main_debug
from livecell_tracker.core import SingleCellTrajectory, SingleCellStatic
from livecell_tracker.trajectory.feature_extractors import compute_skimage_regionprops, compute_haralick_features
from magicgui import magicgui
from magicgui.widgets import Container, LineEdit, Label, Slider, ComboBox, PushButton, Table
from livecell_tracker.viz.table import sc_static_table_widget, sct_table_widget

from typing import Optional
from livecell_tracker.core import SingleCellStatic, SingleCellTrajectory, SingleCellTrajectoryCollection


class Displayer:
    def __init__(self, viewer, magicgui_container: Optional[Container] = None):
        self.viewer = viewer
        self.magicgui_container = magicgui_container


class ScDisplayer(Displayer):
    """
    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer.
    sc : SingleCellStatic
        The single cell static object.
    """

    def __init__(self, sc: SingleCellStatic, viewer, magicgui_container: Optional[Container] = None):
        super().__init__(viewer, magicgui_container)
        self.sc = sc

        if self.sc.regionprops is None:
            self.sc.update_regionprops()

        skimage_features = compute_skimage_regionprops(sc=self.sc)
        self.sc.add_feature("skimage", skimage_features)
        self.sc_table_widget = sc_static_table_widget(sc=self.sc)


class SctDisplayer(Displayer):
    """
    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer.
    trajectory : SingleCellTrajectory
        The single cell trajectory object.
    """

    def __init__(self, trajectory: SingleCellTrajectory, viewer, magicgui_container: Optional[Container] = None):
        super().__init__(viewer, magicgui_container)
        self.trajectory = trajectory

        for timeframe, single_cell in self.trajectory.timeframe_to_single_cell.items():
            if single_cell.regionprops is None:
                single_cell.update_regionprops()

        self.trajectory.compute_features("skimage", compute_skimage_regionprops)
        self.sct_table_widget = sct_table_widget(trajectory=self.trajectory)


def save_sctc_features_to_csv(sctc: SingleCellTrajectoryCollection):
    for track_id, trajectory in sctc:
        trajectory.compute_features("skimage", compute_skimage_regionprops)
    feature_table = sctc.get_feature_table()

    # Format current datetime to string and use it for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(os.getcwd(), f"sctc_feature_table_{timestamp}.csv")

    feature_table.to_csv(filename)


@magicgui(call_button="Save Features to CSV")
def save_features_button(sctc: SingleCellTrajectoryCollection) -> PushButton:
    save_sctc_features_to_csv(sctc)
    button = PushButton(text="Save Features to CSV")
    return button


def create_displayer_napari_ui(displayer: Displayer, sctc: SingleCellTrajectoryCollection):
    # Create labels for identification
    sc_id = Label(value=f"{'sc id:'}{str(displayer.sc.id)}")

    # Widgets list
    widgets = [sc_id]

    # Add sc table widget if available
    if hasattr(displayer, "sc_table_widget"):
        widgets.append(displayer.sc_table_widget)

    # Add trajectory table widget if available
    if hasattr(displayer, "trajectory_table_widget"):
        widgets.append(displayer.trajectory_table_widget)

    # Add the save features button
    save_btn = save_features_button(sctc=sctc)
    assert save_btn is not None, "The button widget is None!"
    widgets.append(save_btn)

    print(widgets)

    # Create the main container
    container = Container(widgets=widgets)

    container.native.setParent(None)
    displayer.magicgui_container = container
    displayer.viewer.window.add_dock_widget(container, name="Sc Displayer")


def create_sc_displayer_napari_ui(sc_displayer: ScDisplayer):
    # Create labels for identification
    sc_id = Label(value=f"{'sc id:'}{str(sc_displayer.sc.id)}")

    # Create the main container
    container = Container(widgets=[sc_id])

    # Add sc table widget if available
    if hasattr(sc_displayer, "sc_table_widget"):
        container.append(sc_displayer.sc_table_widget)

    container.native.setParent(None)
    sc_displayer.viewer.window.add_dock_widget(container, name="Sc Displayer")

    return container


def create_sct_displayer_napari_ui(sct_displayer: SctDisplayer):
    # Create labels for identification
    sct_id = Label(value=f"{'sct id:'}{str(sct_displayer.trajectory.track_id)}")

    # Create the main container
    container = Container(widgets=[sct_id])

    # Add trajectory table widget if available
    if hasattr(sct_displayer, "sct_table_widget"):
        container.append(sct_displayer.sct_table_widget)

    container.native.setParent(None)
    sct_displayer.viewer.window.add_dock_widget(container, name="Sct Displayer")

    return container


def create_sctc_displayer_napari_ui(displayer, sctc: SingleCellTrajectoryCollection):
    # Create the save features button
    save_btn = save_features_button(sctc=sctc)

    # Create the main container
    container = Container(widgets=[save_btn])

    container.native.setParent(None)
    displayer.viewer.window.add_dock_widget(container, name="Sctc Displayer")

    return container
