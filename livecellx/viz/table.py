from magicgui import magicgui
from magicgui.widgets import Table
import pandas as pd
from livecellx.core import SingleCellStatic, SingleCellTrajectory
from livecellx.trajectory.feature_extractors import compute_skimage_regionprops


@magicgui
def sc_static_table_widget(sc: SingleCellStatic):
    # Create a DataFrame from SingleCellStatic instance
    data = {
        "Timeframe": sc.timeframe,
        "BoundingBox": sc.bbox,
        "Centroid": sc.regionprops.centroid,
        "Dataset Name": sc.img_dataset.get_dataset_name(),
    }
    # Convert to DataFrame
    df = pd.DataFrame(list(data.items()), columns=["Attribute", "Value"])

    # Add the feature_dict as separate rows
    feature_dict = sc.get_feature_pd_series().to_dict()
    feature_df = pd.DataFrame(list(feature_dict.items()), columns=["Attribute", "Value"])
    df = pd.concat([df, feature_df], ignore_index=True)

    # Create a Table widget and populate it with the DataFrame
    table = Table(value=df)
    return table


@magicgui
def sct_table_widget(trajectory: SingleCellTrajectory):
    # Get feature table from SingleCellTrajectory
    trajectory.compute_features("skimage", compute_skimage_regionprops)
    feature_table = trajectory.get_sc_feature_table()

    # Create a DataFrame to store the attributes and features
    data = {
        "Timeframe": [],
        "BoundingBox": [],
        "Centroid": [],
        "Dataset Name": [],
    }

    # Initialize the feature columns with empty lists
    for feature_key in feature_table.columns:
        data[feature_key] = []

    # Iterate through the SingleCellStatic instances in the trajectory
    for timeframe, single_cell in trajectory.timeframe_to_single_cell.items():
        data["Timeframe"].append(timeframe)
        data["BoundingBox"].append(single_cell.bbox)
        data["Centroid"].append(single_cell.regionprops.centroid)
        data["Dataset Name"].append(single_cell.img_dataset.get_dataset_name())

        # Extract features from the feature table and add to data
        row_index = "_".join([str(trajectory.track_id), str(timeframe)])
        for feature_key in feature_table.columns:
            data[feature_key].append(feature_table.at[row_index, feature_key])

    df = pd.DataFrame(data)

    # Create a Table widget and populate it with the DataFrame
    table = Table(value=df)
    return table
