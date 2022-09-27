import json
from typing import Callable, Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from skimage.measure._regionprops import RegionProperties

from livecell_tracker.segment.datasets import LiveCellImageDataset


class SingleCellStatic:
    """Single cell at one time frame."""

    def __init__(
        self,
        timeframe: int,
        bbox: np.array = None,
        regionprops: RegionProperties = None,
        img_dataset: LiveCellImageDataset = None,
        feature_dict: dict = {},
    ) -> None:
        """_summary_

        Parameters
        ----------
        timeframe : int
            _description_
        bbox : np.array, optional
            [x1, y1, x2, y2], by default None
        regionprops : RegionProperties, optional
            _description_, by default None
        img_dataset : _type_, optional
            _description_, by default None
        feature_dict : dict, optional
            _description_, by default {}
        """
        self.regionprops = regionprops
        self.timeframe = timeframe
        self.img_dataset = img_dataset
        self.raw_img = self.get_img()
        self.feature_dict = feature_dict
        self.bbox = bbox

        # infer bbox from regionprops
        if (bbox is None) and (regionprops is not None):
            self.bbox = regionprops.bbox
        self.img_crop = SingleCellStatic.gen_skimage_bbox_img_crop(
            self.bbox, self.raw_img
        )

    def get_img(self):
        return self.img_dataset[self.timeframe]

    def get_bbox(self) -> np.array:
        return np.array(self.bbox)

    def gen_skimage_bbox_img_crop(bbox, img):
        min_x, max_x, min_y, max_y = (
            int(bbox[0]),
            int(bbox[2]),
            int(bbox[1]),
            int(bbox[3]),
        )
        img_crop = img[min_x:max_x, min_y:max_y]
        return img_crop

    def to_json_dict(self):
        """returns a dict that can be converted to json"""
        res = {
            "timeframe": int(self.timeframe),
            "bbox": list(np.array(self.bbox, dtype=float)),
            "feature_dict": self.feature_dict,
            "dataset_name": str(self.img_dataset.get_dataset_name()),
            "dataset_path": str(self.img_dataset.get_dataset_path()),
        }
        return res

    def to_json(self, path=None):
        if path is None:
            return json.dumps(self.to_json_dict())
        else:
            with open(path, "w+") as f:
                json.dump(self.to_json_dict(), f)

class SingleCellTrajectory:
    """
    Single cell trajectory containing trajectory information for one single cell at all timeframes.
    """

    def __init__(
        self,
        raw_img_dataset: LiveCellImageDataset,
        track_id: int = None,
        timeframe_to_single_cell: Dict[int, SingleCellStatic] = {},
    ) -> None:
        self.timeframe_set = set()
        self.timeframe_to_single_cell = timeframe_to_single_cell
        self.raw_img_dataset = raw_img_dataset
        self.raw_total_timeframe = len(raw_img_dataset)
        self.track_id = track_id

    def add_timeframe_data(self, timeframe, cell: SingleCellStatic):
        self.timeframe_to_single_cell[timeframe] = cell
        self.timeframe_set.add(timeframe)

    def get_img(self, timeframe):
        return self.raw_img_dataset[timeframe]

    def get_timeframe_span_range(self):
        return (min(self.timeframe_set), max(self.timeframe_set))

    def get_timeframe_span_length(self):
        min_t, max_t = self.get_timeframe_span_range()
        return max_t - min_t

    def get_single_cell(self, timeframe: int) -> SingleCellStatic:
        return self.timeframe_to_single_cell[timeframe]

    def to_dict(self):
        res = {
            "track_id": int(self.track_id),
            "timeframe_to_single_cell": {
                timeframe: sc.to_json_dict()
                for timeframe, sc in self.timeframe_to_single_cell.items()
            },
            "dataset_info": self.raw_img_dataset.to_json_dict()
        }
        return res

    def to_json(self, path=None):
        if path is None:
            return json.dumps(self.to_dict())
        else:
            with open(path, "w+") as f:
                json.dump(self.to_dict(), f)

    def generate_single_trajectory_movie(
        self,
        save_path="./tmp.gif",
        min_length=None,
        ax=None,
        fig=None,
        ani_update_func: Callable = None,  # how you draw each frame
    ):
        if min_length is not None:
            if self.get_timeframe_span_length() < min_length:
                print("[Viz] skipping the current trajectory track_id: ", self.track_id)
                return None
        if ax is None:
            fig, ax = plt.subplots()

        def init():
            return []

        def default_update(sc_tp: SingleCellStatic):
            frame_idx, raw_img, bbox, img_crop = (
                sc_tp.timeframe,
                sc_tp.raw_img,
                sc_tp.bbox,
                sc_tp.img_crop,
            )
            ax.cla()
            frame_text = ax.text(
                -10,
                -10,
                "frame: {}".format(frame_idx),
                fontsize=10,
                color="red",
                ha="center",
                va="center",
            )
            ax.imshow(img_crop)
            return []

        if ani_update_func is None:
            ani_update_func = default_update

        frame_data = []
        for frame_idx in self.timeframe_to_single_cell:
            sc_timepoint = self.get_single_cell(frame_idx)
            img = self.raw_img_dataset[frame_idx]
            bbox = sc_timepoint.get_bbox()
            frame_data.append(sc_timepoint)

        ani = FuncAnimation(
            fig, default_update, frames=frame_data, init_func=init, blit=True
        )
        print("saving to: %s..." % save_path)
        ani.save(save_path)


class SingleCellTrajectoryCollection:
    pass
