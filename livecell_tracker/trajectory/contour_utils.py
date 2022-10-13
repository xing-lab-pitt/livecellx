from typing import List
from livecell_tracker.core import (
    SingleCellTrajectory,
    SingleCellStatic,
    SingleCellTrajectoryCollection,
)
from livecell_tracker.trajectory.contour.contour_class import Contour


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
