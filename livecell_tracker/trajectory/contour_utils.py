from typing import List
from livecell_tracker.core import (
    SingleCellTrajectory,
    SingleCellStatic,
    SingleCellTrajectoryCollection,
)
from livecell_tracker.trajectory.contour.contour_class import Contour
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


def get_cellTool_contour_points(traj: SingleCellTrajectory, contour_num_points=500) -> List[Contour]:
    sorted_timeframes = sorted(traj.timeframe_set)
    cellTool_contours = []
    for timeframe in sorted_timeframes:
        single_cell = traj.get_single_cell(timeframe)
        contour = Contour(points=single_cell.contour, units="pixels")
        contour.resample(num_points=contour_num_points)
        contour.axis_align()
        cellTool_contours.append(contour)
        # points = contour.points
    return cellTool_contours


def viz_contours(cell_contours: List[Contour], **kwargs):
    for contour in cell_contours:
        plt.plot(contour.points[:, 0], contour.points[:, 1], **kwargs)
    plt.show()
    
def get_morphology_PCA(trajectory_collection, contour_num_points, trajectory_threshold=1, num_components=0.98, dim = 2):
    """Obtain PCA for the morphology contour features obtained through active shape model for an entire trajectory data for an image dataset. 

    Args:
        trajectory_collection (obj): trajectory collection of a dataset
        contour_num_points (int): number of landmark contour points
        trajectory_threshold (int, optional): _description_. Defaults to 1.
        num_components (float, optional): the amount of variance that needs to be explained is greater than the percentage specified by num_components.

    Returns:
        List: PCA values for sct contours
    """
    all_flat_contours = [] # all flattened contours, shape: # num_trajectories x (#timeframes * #contour_num_points)

    for track_id_num in trajectory_collection.get_track_ids():
        traj = trajectory_collection.get_trajectory(track_id_num)

        # getting cell contours using active shape model
        sct_contours = get_cellTool_contour_points(traj, contour_num_points=contour_num_points) #shape: #time x #contours

        if len(sct_contours) > trajectory_threshold:
            # Flatten each 2D array and stack them horizontally
            flattened_contour = [contour.points.flatten() for contour in sct_contours]
            final_contour = np.hstack(flattened_contour)
            all_flat_contours.append(final_contour)

    all_sctc_contours = np.concatenate(all_flat_contours, axis=0).reshape(-1, dim * contour_num_points)

    # getting PCA
    _pca_model = PCA(n_components=num_components, svd_solver="full")

    transformed_pca_contour_data = _pca_model.fit_transform(all_sctc_contours)
    print("Variance ratios and their sum = ", _pca_model.explained_variance_ratio_, sum(_pca_model.explained_variance_ratio_))

    pca_sctc = [
        _pca_model.transform(
            final_contour.reshape((int(len(final_contour) / (dim * contour_num_points)), dim * contour_num_points))
        )
        for final_contour in all_flat_contours
    ]

    return pca_sctc
