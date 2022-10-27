from livecell_tracker.core.single_cell import SingleCellStatic, SingleCellTrajectory, SingleCellTrajectoryCollection
import numpy as np
from napari.viewer import Viewer
import matplotlib.pyplot as plt


class Visualizer:
    @staticmethod
    def show_trajectory_on_grid(
        trajectory: SingleCellTrajectory,
        nc=4,
        start_timeframe=1,
        time_interval=1,
        bbox_padding=20,
    ):

        span_range = trajectory.get_timeframe_span_range()
        traj_start, traj_end = span_range
        nr = int(np.ceil((traj_end - start_timeframe + 1) / nc))

        fig, axes = plt.subplots(nr, nc, figsize=(nc * 4, nr * 4))
        # axes are 1d array if nr=1. transform to 2d array
        if nr == 1:
            axes = np.array([axes])
        if start_timeframe < traj_start:
            print(
                "start timeframe larger than the first timeframe of the trajectory, replace start_timeframe with the first timeframe..."
            )
            start_timeframe = span_range[0]

        for r in range(nr):
            for c in range(nc):
                ax = axes[r, c]
                ax.axis("off")
                timeframe = start_timeframe + time_interval * (r * nc + c)
                if timeframe > traj_end:
                    print(
                        "timeframe: {timeframe} larger than the last timeframe of the trajectory, stopping...".format(
                            timeframe=timeframe
                        )
                    )
                    break
                if timeframe not in trajectory.timeframe_set:
                    print(
                        "timeframe: {timeframe} does not present in the trajectory, skipping...".format(
                            timeframe=timeframe
                        )
                    )
                    continue
                sc = trajectory.get_single_cell(timeframe)
                sc_img = sc.get_img_crop(padding=bbox_padding)
                ax.imshow(sc_img)
                contour_coords = sc.get_img_crop_contour_coords(padding=bbox_padding)
                ax.scatter(contour_coords[:, 1], contour_coords[:, 0], s=1, c="r")
                # trajectory_collection[timeframe].plot(axes[r, c])
                ax.set_title(f"timeframe: {timeframe}")
        fig.tight_layout(pad=0.5, h_pad=0.4, w_pad=0.4)
