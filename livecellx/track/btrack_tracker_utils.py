"""
Utility functions for tracking single cells using the btrack algorithm.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import tqdm

import btrack
from btrack.constants import BayesianUpdates

from livecellx.core.datasets import LiveCellImageDataset
from livecellx.core.single_cell import (
    SingleCellStatic,
    SingleCellTrajectory,
    SingleCellTrajectoryCollection,
)
from livecellx.livecell_logger import main_info, main_warning


def get_bbox_from_contour(contour: list) -> np.ndarray:
    """Get bounding box from a contour.

    Parameters
    ----------
    contour : list
        A list of (x, y) points, with shape #pts x 2

    Returns
    -------
    np.ndarray
        Bounding box of the input contour, with length=4 [x1, y1, x2, y2]
    """
    contour = np.array(contour)
    return np.array(
        [
            contour[:, 0].min(),
            contour[:, 1].min(),
            contour[:, 0].max(),
            contour[:, 1].max(),
        ]
    )


def single_cell_to_btrack_object(
    sc: SingleCellStatic, feature_names: Optional[List[str]] = None
) -> btrack.btypes.PyTrackObject:
    """Convert a SingleCellStatic object to a btrack PyTrackObject.

    Parameters
    ----------
    sc : SingleCellStatic
        The single cell object to convert
    feature_names : Optional[List[str]], optional
        List of feature names to include, by default None which includes all features

    Returns
    -------
    btrack.btypes.PyTrackObject
        A btrack PyTrackObject with the same properties as the SingleCellStatic
    """
    # Get the frame (timeframe)
    frame = sc.timeframe

    # Get the position (center of the bounding box)
    bbox = sc.bbox
    x = (bbox[0] + bbox[2]) / 2
    y = (bbox[1] + bbox[3]) / 2
    z = 0  # We assume 2D tracking

    # Create a btrack object
    obj = btrack.btypes.PyTrackObject()

    # Set the position and time
    obj.x = float(x)
    obj.y = float(y)
    obj.z = float(z)
    obj.t = int(frame)

    # Set the ID if available
    if sc.id is not None:
        # btrack requires integer IDs
        # The ID should already be an integer at this point (converted in track_btrack_from_scs)
        if not isinstance(sc.id, int):
            main_warning(
                f"btrack requires integer IDs, but got sc.id={sc.id} of type {type(sc.id)}. "
                f"This may cause issues with tracking."
            )
        obj.ID = int(sc.id)

    # Add features if available
    if hasattr(sc, "feature_dict") and sc.feature_dict:
        sc_pd_features = sc.get_feature_pd_series()

        # Add features to the object's properties
        if feature_names is not None:
            for feature in feature_names:
                if feature in sc_pd_features:
                    obj.properties[feature] = float(sc_pd_features[feature])
                else:
                    assert False, f"Feature {feature} not found in SingleCellStatic object"

    return obj


def btrack_object_to_single_cell(
    obj: btrack.btypes.PyTrackObject,
    contour: Optional[np.ndarray] = None,
    img_dataset: Optional[LiveCellImageDataset] = None,
    mask_dataset: Optional[LiveCellImageDataset] = None,
) -> SingleCellStatic:
    """Convert a btrack PyTrackObject to a SingleCellStatic object.

    Parameters
    ----------
    obj : btrack.btypes.PyTrackObject
        The btrack object to convert
    contour : Optional[np.ndarray], optional
        The contour of the cell, by default None
    img_dataset : Optional[LiveCellImageDataset], optional
        The image dataset, by default None
    mask_dataset : Optional[LiveCellImageDataset], optional
        The mask dataset, by default None

    Returns
    -------
    SingleCellStatic
        A SingleCellStatic object with the same properties as the btrack object
    """
    # Get the timeframe
    timeframe = int(obj.t)

    # Get the bounding box (estimate from position if contour not provided)
    if contour is not None:
        bbox = get_bbox_from_contour(contour)
    else:
        # Create a small bounding box around the position
        x, y = obj.x, obj.y
        bbox = np.array([x - 5, y - 5, x + 5, y + 5])

    # Get the ID
    cell_id = obj.ID if obj.ID >= 0 else None

    # Create a feature dictionary from the object's features
    feature_dict = {}
    for feature in obj.properties:
        feature_dict[feature] = obj.properties[feature]

    # Create the SingleCellStatic object
    sc = SingleCellStatic(
        id=cell_id,
        timeframe=timeframe,
        bbox=bbox,
        contour=contour,
        img_dataset=img_dataset,
        mask_dataset=mask_dataset,
        feature_dict=feature_dict,
    )

    return sc


def track_btrack_from_scs(
    single_cells: List[SingleCellStatic],
    raw_imgs: Optional[LiveCellImageDataset] = None,
    mask_dataset: Optional[LiveCellImageDataset] = None,
    config: Optional[Dict] = None,
    feature_names: Optional[List[str]] = None,
    max_search_radius: float = 100.0,
    return_dataframe: bool = False,
) -> Union[SingleCellTrajectoryCollection, Tuple[SingleCellTrajectoryCollection, pd.DataFrame]]:
    """Track single cells using the btrack algorithm.

    Parameters
    ----------
    single_cells : List[SingleCellStatic]
        List of single cell objects to track
    raw_imgs : Optional[LiveCellImageDataset], optional
        The raw image dataset, by default None
    mask_dataset : Optional[LiveCellImageDataset], optional
        The mask dataset, by default None
    config : Optional[Dict], optional
        Configuration for the btrack tracker, by default None
    feature_names : Optional[List[str]], optional
        List of feature names to use for tracking, by default None
    max_search_radius : float, optional
        Maximum search radius for tracking, by default 100.0
    return_dataframe : bool, optional
        Whether to return a pandas DataFrame with the tracking results, by default False

    Returns
    -------
    Union[SingleCellTrajectoryCollection, Tuple[SingleCellTrajectoryCollection, pd.DataFrame]]
        A collection of single cell trajectories, and optionally a DataFrame with the tracking results
    """
    main_info("Converting single cells to btrack objects...")

    # Store original IDs and assign integer IDs for btrack
    # btrack requires integer IDs, but our single cell objects can have string/UUID IDs
    btrack_id_to_original_id = {}
    original_sc_map = {}
    for i, sc in enumerate(single_cells):
        # Store the original ID in the meta dictionary
        if not hasattr(sc, "meta"):
            sc.meta = {}
        sc.meta["original_id"] = sc.id

        # Assign a new integer ID
        btrack_id_to_original_id[i] = sc.id
        btrack_id_to_original_id[i] = sc.id

        sc.id = i
        original_sc_map[sc.id] = sc

    # Convert single cells to btrack objects
    objects = []
    # Map from btrack object ID to original cell ID
    for sc in single_cells:
        obj = single_cell_to_btrack_object(sc, feature_names)
        objects.append(obj)

    btrack_id_to_btrack_object = {obj.ID: obj for obj in objects}
    # Create and configure the tracker
    if config is None:
        # Default configuration for cell tracking
        config = {
            "motion_model": {
                "name": "cell_motion",
                "dt": 1.0,
                "measurements": 3,  # x, y, z
                "states": 6,  # x, y, z, dx, dy, dz
                "accuracy": 7.5,
                "prob_not_assign": 0.1,
                "max_lost": 5,
                "A": {
                    "matrix": [
                        1,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                    ]
                },
                "H": {"matrix": [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]},
                "P": {
                    "sigma": 150.0,
                    "matrix": [
                        0.1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0.1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0.1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1.0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1.0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1.0,
                    ],
                },
                "G": {"sigma": 15.0, "matrix": [0.5, 0.5, 0.5, 1.0, 1.0, 1.0]},
                "R": {"sigma": 5.0, "matrix": [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]},
            },
            "optimizer": {"name": "hungarian", "params": {"max_search_radius": max_search_radius}},
            "update_mode": BayesianUpdates.EXACT,
        }

    # If feature names are provided, configure the tracker to use them
    if feature_names:
        main_info(f"Using features for tracking: {feature_names}")
        config["features"] = feature_names

    # Create the tracker with the configuration
    tracker = btrack.BayesianTracker()
    tracker.configure(config)

    # Append the objects to the tracker
    tracker.append(objects)

    main_info("Tracking objects...")
    # Track the objects
    tracker.track()

    # Get the tracks
    tracks = tracker.tracks
    main_info(f"Found {len(tracks)} tracks")

    # Create a SingleCellTrajectoryCollection
    traj_collection = SingleCellTrajectoryCollection()

    # Convert tracks to SingleCellTrajectory objects
    for track in tqdm.tqdm(tracks, desc="Converting tracks to trajectories"):
        track_id = track.ID
        trajectory = SingleCellTrajectory(track_id=track_id, img_dataset=raw_imgs)

        # Add each object in the track to the trajectory
        for _track_i, obj_id in enumerate(track.refs):
            if track.dummy[_track_i]:
                continue
            obj = btrack_id_to_btrack_object[obj_id]
            timeframe = int(obj.t)
            # Handle dummy case
            if obj.dummy:
                continue
            # Try to find the original single cell object
            original_sc = original_sc_map[obj.ID]

            if original_sc is not None:
                # Use the original single cell object
                sc = original_sc
                # Store the btrack ID in the uns dictionary
                sc.uns["btrack_id"] = track_id
            else:
                # If original not found, create a new SingleCellStatic object
                raise ValueError(
                    f"Original single cell object not found for track ID {track_id} and timeframe {timeframe}"
                )

            # Add the single cell to the trajectory
            trajectory.timeframe_to_single_cell[timeframe] = sc

        # Add the trajectory to the collection
        traj_collection.track_id_to_trajectory[track_id] = trajectory

    # Restore original IDs
    main_info("Restoring original IDs...")
    for sc in single_cells:
        if hasattr(sc, "meta") and "original_id" in sc.meta:
            sc.id = sc.meta["original_id"]

    # Optionally return a DataFrame with the tracking results
    if return_dataframe:
        # Create a DataFrame from the tracks
        df = pd.DataFrame()

        for track in tracks:
            for _track_i, obj_id in enumerate(track.refs):
                if track.dummy[_track_i]:
                    continue
                obj = btrack_id_to_btrack_object[obj_id]
                row = {
                    "track_id": track.ID,
                    "frame": obj.t,
                    "x": obj.x,
                    "y": obj.y,
                    "z": obj.z,
                }

                # Add original cell ID if available
                if obj.ID in btrack_id_to_original_id:
                    row["original_id"] = btrack_id_to_original_id[obj.ID]

                # Add features
                for feature in obj.properties:
                    row[feature] = obj.properties[feature]

                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        return traj_collection, df

    return traj_collection


def track_btrack_from_features(
    features_df: pd.DataFrame,
    raw_imgs: Optional[LiveCellImageDataset] = None,
    mask_dataset: Optional[LiveCellImageDataset] = None,
    config: Optional[Dict] = None,
    feature_columns: Optional[List[str]] = None,
    position_columns: List[str] = ["x", "y"],
    frame_column: str = "frame",
    id_column: Optional[str] = "id",
    max_search_radius: float = 100.0,
    return_dataframe: bool = False,
) -> Union[SingleCellTrajectoryCollection, Tuple[SingleCellTrajectoryCollection, pd.DataFrame]]:
    """Track objects based on features in a DataFrame.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing object features with columns for position, frame, and features
    raw_imgs : Optional[LiveCellImageDataset], optional
        The raw image dataset, by default None
    mask_dataset : Optional[LiveCellImageDataset], optional
        The mask dataset, by default None
    config : Optional[Dict], optional
        Configuration for the btrack tracker, by default None
    feature_columns : Optional[List[str]], optional
        List of column names to use as features for tracking, by default None
    position_columns : List[str], optional
        List of column names for object positions, by default ['x', 'y']
    frame_column : str, optional
        Column name for the frame/time information, by default 'frame'
    id_column : Optional[str], optional
        Column name for object IDs, by default 'id'
    max_search_radius : float, optional
        Maximum search radius for tracking, by default 100.0
    return_dataframe : bool, optional
        Whether to return a pandas DataFrame with the tracking results, by default False

    Returns
    -------
    Union[SingleCellTrajectoryCollection, Tuple[SingleCellTrajectoryCollection, pd.DataFrame]]
        A collection of single cell trajectories, and optionally a DataFrame with the tracking results
    """
    main_info("Converting DataFrame to btrack objects...")

    # Validate required columns
    for col in position_columns + [frame_column]:
        if col not in features_df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    # Create a mapping from original IDs to integer IDs
    original_ids = {}
    id_counter = 0

    if id_column is not None and id_column in features_df.columns:
        for _, row in features_df.iterrows():
            original_id = row[id_column]
            if original_id not in original_ids:
                original_ids[original_id] = id_counter
                id_counter += 1

    # Create btrack objects from the DataFrame
    objects = []

    for _, row in features_df.iterrows():
        # Get position and frame
        x, y = [row[col] for col in position_columns]
        z = 0  # Assume 2D tracking
        frame = int(row[frame_column])

        # Create a btrack object
        obj = btrack.btypes.PyTrackObject()

        # Set the position and time
        obj.x = float(x)
        obj.y = float(y)
        obj.z = float(z)
        obj.t = int(frame)

        # Set ID if available
        if id_column is not None and id_column in features_df.columns:
            original_id = row[id_column]
            # Use the integer ID from the mapping
            obj.ID = int(original_ids[original_id])

        # Add features
        if feature_columns is not None:
            for feature in feature_columns:
                if feature in features_df.columns:
                    obj.properties[feature] = float(row[feature])

        objects.append(obj)
    # Store mapping from btrack
    btrack_id_to_btrack_object = {obj.ID: obj for obj in objects}
    # Create and configure the tracker
    if config is None:
        # Default configuration for cell tracking
        config = {
            "motion_model": {
                "name": "cell_motion",
                "dt": 1.0,
                "measurements": 3,  # x, y, z
                "states": 6,  # x, y, z, dx, dy, dz
                "accuracy": 7.5,
                "prob_not_assign": 0.1,
                "max_lost": 5,
                "A": {
                    "matrix": [
                        1,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                    ]
                },
                "H": {"matrix": [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]},
                "P": {
                    "sigma": 150.0,
                    "matrix": [
                        0.1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0.1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0.1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1.0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1.0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1.0,
                    ],
                },
                "G": {"sigma": 15.0, "matrix": [0.5, 0.5, 0.5, 1.0, 1.0, 1.0]},
                "R": {"sigma": 5.0, "matrix": [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]},
            },
            "optimizer": {"name": "hungarian", "params": {"max_search_radius": max_search_radius}},
            "update_mode": BayesianUpdates.EXACT,
        }

    # If feature columns are provided, configure the tracker to use them
    if feature_columns:
        main_info(f"Using features for tracking: {feature_columns}")

        # Add feature weights to the configuration
        if "features" not in config:
            config["features"] = {}

        # Set equal weights for all features by default
        for feature in feature_columns:
            if feature not in config["features"]:
                config["features"][feature] = 1.0

    # Create the tracker with the configuration
    tracker = btrack.BayesianTracker()
    tracker.configure(config)

    # Append the objects to the tracker
    tracker.append(objects)

    main_info("Tracking objects...")
    # Track the objects
    tracker.track()

    # Get the tracks
    tracks = tracker.tracks
    main_info(f"Found {len(tracks)} tracks")

    # Create a SingleCellTrajectoryCollection
    traj_collection = SingleCellTrajectoryCollection()

    # Convert tracks to SingleCellTrajectory objects
    for track in tqdm.tqdm(tracks, desc="Converting tracks to trajectories"):
        track_id = track.ID
        trajectory = SingleCellTrajectory(track_id=track_id, img_dataset=raw_imgs)

        # Add each object in the track to the trajectory
        for _track_i, obj_id in enumerate(track.refs):
            if track.dummy[_track_i]:
                continue
            obj = btrack_id_to_btrack_object[obj.ID]
            timeframe = int(obj.t)

            # Create a feature dictionary from the object's features
            feature_dict = {}
            for feature in obj.properties:
                feature_dict[feature] = obj.properties[feature]

            # Create a bounding box around the position
            x, y = obj.x, obj.y
            bbox = np.array([x - 5, y - 5, x + 5, y + 5])

            # Get the original ID if available
            original_id = None
            if obj.ID >= 0:
                # Find the original ID from the mapping
                for orig_id, int_id in original_ids.items():
                    if int_id == obj.ID:
                        original_id = orig_id
                        break

            # Create a SingleCellStatic object
            sc = SingleCellStatic(
                id=original_id,
                timeframe=timeframe,
                bbox=bbox,
                img_dataset=raw_imgs,
                mask_dataset=mask_dataset,
                feature_dict=feature_dict,
            )

            # Store the btrack ID in the uns dictionary
            if not hasattr(sc, "uns"):
                sc.uns = {}
            sc.uns["btrack_id"] = track_id

            # Add the single cell to the trajectory
            trajectory.timeframe_to_single_cell[timeframe] = sc

        # Add the trajectory to the collection
        traj_collection.track_id_to_trajectory[track_id] = trajectory

    # Optionally return a DataFrame with the tracking results
    if return_dataframe:
        # Create a DataFrame from the tracks
        result_df = pd.DataFrame()

        for track in tracks:
            for _track_i, obj_id in enumerate(track.refs):
                if track.dummy[_track_i]:
                    continue
                obj = btrack_id_to_btrack_object[obj_id]
                row = {
                    "track_id": track.ID,
                    "frame": obj.t,
                    "x": obj.x,
                    "y": obj.y,
                    "z": obj.z,
                }

                # Add features
                for feature in obj.properties:
                    row[feature] = obj.properties[feature]

                result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

        return traj_collection, result_df

    return traj_collection
