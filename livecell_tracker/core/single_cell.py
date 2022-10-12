import json
from typing import Callable, Dict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from skimage.measure._regionprops import RegionProperties

from livecell_tracker.core.datasets import LiveCellImageDataset


# TODO: possibly refactor load_from_json methods into a mixin class
class SingleCellStatic:
    """Single cell at one time frame."""

    HARALICK_FEATURE_KEY = "_haralick"
    MORPHOLOGY_FEATURE_KEY = "_morphology"
    AUTOENCODER_FEATURE_KEY = "_autoencoder"

    def __init__(
        self,
        timeframe: int,
        bbox: np.array = None,
        regionprops: RegionProperties = None,
        img_dataset: LiveCellImageDataset = None,
        mask_dataset: LiveCellImageDataset = None,
        feature_dict: Dict[str, np.array] = dict(),
        contour: np.array = None,
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
        self.mask_dataset = mask_dataset
        self.raw_img = self.get_img()
        self.feature_dict = feature_dict
        self.bbox = bbox
        self.contour = np.array(contour, dtype=float)

        # infer bbox from regionprops
        if (bbox is None) and (regionprops is not None):
            self.bbox = regionprops.bbox
        self.img_crop = None
        self.mask_crop = None

    def get_img(self):
        return self.img_dataset[self.timeframe]

    def get_mask(self):
        return self.mask_dataset[self.timeframe]

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

    def get_img_crop(self):
        if self.img_crop is None:
            self.img_crop = SingleCellStatic.gen_skimage_bbox_img_crop(self.bbox, self.raw_img)
        return self.img_crop

    def get_mask_crop(self):
        if self.mask_crop is None:
            self.mask_crop = SingleCellStatic.gen_skimage_bbox_img_crop(self.bbox, self.get_mask())
        return self.mask_crop

    def update_bbox(self, bbox):
        self.bbox = bbox
        self.img_crop = None
        self.mask_crop = None

    def to_json_dict(self):
        """returns a dict that can be converted to json"""
        res = {
            "timeframe": int(self.timeframe),
            "bbox": list(np.array(self.bbox, dtype=float)),
            "feature_dict": self.feature_dict,
            "dataset_name": str(self.img_dataset.get_dataset_name()),
            "dataset_path": str(self.img_dataset.get_dataset_path()),
            "contour": self.contour.tolist(),
        }
        return res

    def load_from_json_dict(self, json_dict):
        self.timeframe = json_dict["timeframe"]
        self.bbox = np.array(json_dict["bbox"], dtype=float)
        self.feature_dict = json_dict["feature_dict"]
        self.img_dataset = LiveCellImageDataset(dir_path=json_dict["dataset_path"], name=json_dict["dataset_name"])
        self.mask_dataset = LiveCellImageDataset(json_dict["dataset_name"] + "_mask", json_dict["dataset_path"])
        return self

    def to_json(self, path=None):
        if path is None:
            return json.dumps(self.to_json_dict())
        else:
            with open(path, "w+") as f:
                json.dump(self.to_json_dict(), f)

    def extract_feature(self):
        raise NotImplementedError

    def show(self, ax: plt.Axes = None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.imshow(self.get_img_crop(), **kwargs)
        return ax


class SingleCellTrajectory:
    """
    Single cell trajectory containing trajectory information for one single cell at all timeframes.
    """

    def __init__(
        self,
        track_id: int = None,
        timeframe_to_single_cell: Dict[int, SingleCellStatic] = None,
        raw_img_dataset: LiveCellImageDataset = None,
        mask_dataset: LiveCellImageDataset = None,
        extra_datasets: Dict[str, LiveCellImageDataset] = None,
    ) -> None:
        self.timeframe_set = set()
        if timeframe_to_single_cell is None:
            self.timeframe_to_single_cell = dict()
        self.raw_img_dataset = raw_img_dataset
        self.raw_total_timeframe = len(raw_img_dataset) if raw_img_dataset is not None else None
        self.track_id = track_id
        self.mask_dataset = mask_dataset
        self.extra_datasets = extra_datasets

    def add_timeframe_data(self, timeframe, cell: SingleCellStatic):
        self.timeframe_to_single_cell[timeframe] = cell
        self.timeframe_set.add(timeframe)

    def get_img(self, timeframe):
        return self.raw_img_dataset[timeframe]

    def get_mask(self, timeframe):
        assert self.mask_dataset is not None, "missing mask dataset in single cell trajectory"
        return self.mask_dataset[timeframe]

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
                int(float(timeframe)): sc.to_json_dict() for timeframe, sc in self.timeframe_to_single_cell.items()
            },
            "dataset_info": self.raw_img_dataset.to_json_dict(),
        }
        return res

    def to_json(self, path=None):
        if path is None:
            return json.dumps(self.to_dict())
        else:
            with open(path, "w+") as f:
                json.dump(self.to_dict(), f)

    def load_from_json_dict(self, json_dict):
        self.track_id = json_dict["track_id"]
        self.raw_img_dataset = LiveCellImageDataset().load_from_json_dict(json_dict["dataset_info"])
        self.raw_total_timeframe = len(self.raw_img_dataset)
        self.timeframe_to_single_cell = {
            int(timeframe): SingleCellStatic(int(timeframe), img_dataset=self.raw_img_dataset).load_from_json_dict(sc)
            for timeframe, sc in json_dict["timeframe_to_single_cell"].items()
        }
        self.timeframe_set = set(self.timeframe_to_single_cell.keys())
        return self


class SingleCellTrajectoryCollection:
    def __init__(self) -> None:
        self.track_id_to_trajectory = dict()

    def add_trajectory(self, trajectory: SingleCellTrajectory):
        self.track_id_to_trajectory[trajectory.track_id] = trajectory

    def get_trajectory(self, track_id) -> SingleCellTrajectory:
        return self.track_id_to_trajectory[track_id]

    def __contains__(self, track_id):
        return track_id in self.track_id_to_trajectory

    def to_json_dict(self):
        return {
            "track_id_to_trajectory": {
                int(track_id): trajectory.to_dict() for track_id, trajectory in self.track_id_to_trajectory.items()
            }
        }

    def to_json(self, path):
        with open(path, "w+") as f:
            json.dump(self.to_json_dict(), f)

    def load_from_json_dict(self, json_dict):
        self.track_id_to_trajectory = {
            int(float(track_id)): SingleCellTrajectory().load_from_json_dict(trajectory_dict)
            for track_id, trajectory_dict in json_dict["track_id_to_trajectory"].items()
        }
        return self
