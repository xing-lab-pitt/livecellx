import json
from typing import Callable, Dict, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import pandas as pd
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

        # TODO: discuss and decide whether to keep mask dataset
        # self.mask_dataset = mask_dataset

        self.feature_dict = feature_dict
        self.bbox = bbox
        self.contour = np.array(contour, dtype=float)

        # infer bbox from regionprops
        if (bbox is None) and (regionprops is not None):
            self.bbox = regionprops.bbox

        # TODO: enable img_crops caching ONLY in RAM mode, otherwise caching these causes memory issues
        # self.raw_img = self.get_img()
        # self.img_crop = None
        # self.mask_crop = None

    def get_img(self):
        return self.img_dataset[self.timeframe]

    def get_mask(self):
        return self.mask_dataset[self.timeframe]

    def get_bbox(self) -> np.array:
        return np.array(self.bbox)

    def gen_skimage_bbox_img_crop(bbox, img, padding=0):
        min_x, max_x, min_y, max_y = (
            int(bbox[0]),
            int(bbox[2]),
            int(bbox[1]),
            int(bbox[3]),
        )
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        img_crop = img[min_x : max_x + padding, min_y : max_y + padding, ...]
        return img_crop

    def get_img_crop(self, padding=0):
        img_crop = SingleCellStatic.gen_skimage_bbox_img_crop(self.bbox, self.get_img(), padding=padding)
        # TODO: enable in RAM mode
        # if self.img_crop is None:
        #     self.img_crop = img_crop
        return img_crop

    def get_mask_crop(self):
        # TODO: enable in RAM mode
        # if self.mask_crop is None:
        #     self.mask_crop = SingleCellStatic.gen_skimage_bbox_img_crop(self.bbox, self.get_mask())
        return SingleCellStatic.gen_skimage_bbox_img_crop(self.bbox, self.get_mask())

    def update_bbox(self, bbox):
        self.bbox = bbox
        self.img_crop = None
        self.mask_crop = None

    def to_json_dict(self, dataset_json=False):
        """returns a dict that can be converted to json"""
        res = {
            "timeframe": int(self.timeframe),
            "bbox": list(np.array(self.bbox, dtype=float)),
            "feature_dict": self.feature_dict,
            "contour": self.contour.tolist(),
        }
        if dataset_json:
            res["dataset_json"] = self.img_dataset.to_json_dict()
        return res

    def load_from_json_dict(self, json_dict, img_dataset=None):
        self.timeframe = json_dict["timeframe"]
        self.bbox = np.array(json_dict["bbox"], dtype=float)
        self.feature_dict = json_dict["feature_dict"]
        if img_dataset is None and "dataset_json" in json_dict:
            self.img_dataset = LiveCellImageDataset().load_from_json_dict(json_dict["dataset_json"])
        else:
            self.img_dataset = img_dataset

        # TODO: discuss and decide whether to keep mask dataset
        # self.mask_dataset = LiveCellImageDataset(
        #     json_dict["dataset_name"] + "_mask", json_dict["dataset_path"]
        # )
        self.contour = np.array(json_dict["contour"], dtype=float)
        return self

    def write_json(self, path=None):
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

    def get_img_crop_contour_coords(self, padding=0):
        xs = self.contour[:, 0] - max(0, self.bbox[0] - padding)
        ys = self.contour[:, 1] - max(0, self.bbox[1] - padding)
        return np.array([xs, ys]).T

    def get_contour_mask(self):
        import scipy.ndimage as ndimage

        contour = self.contour
        res_mask = np.zeros(self.get_img().shape, dtype=bool)
        # create a contour image by using the contour coordinates rounded to their nearest integer value
        res_mask[np.round(contour[:, 0]).astype("int"), np.round(contour[:, 1]).astype("int")] = 1
        # fill in the hole created by the contour boundary
        res_mask = ndimage.binary_fill_holes(res_mask)
        res_mask_crop = SingleCellStatic.gen_skimage_bbox_img_crop(self.bbox, res_mask)
        return res_mask_crop

    def get_contour_img(self, background_val=0):
        contour_mask = self.get_contour_mask()
        contour_img = self.get_img_crop()
        contour_img[np.logical_not(contour_mask)] = background_val
        return contour_img

    def add_feature(self, name, features: Union[np.array, pd.Series]):
        if not isinstance(features, (np.ndarray, pd.Series)):
            raise TypeError("features must be a numpy array or pandas series")
        self.feature_dict[name] = features

    def get_feature_pd_series(self):
        res_series = None
        for feature_name in self.feature_dict:
            features = self.feature_dict[feature_name]
            if isinstance(features, np.ndarray):
                tmp_series = pd.Series(self.feature_dict[feature_name])
            elif isinstance(features, pd.Series):
                tmp_series = features
            tmp_series = tmp_series.add_prefix(feature_name + "_")
            if res_series is None:
                res_series = tmp_series
            else:
                res_series = pd.concat([res_series, tmp_series])
        return res_series

    def get_napari_shape_vec(self):
        x1, y1, x2, y2 = self.bbox
        return [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]

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

    def __len__(self):
        return self.get_timeframe_span_length()

    def __getitem__(self, timeframe: int) -> SingleCellStatic:
        if timeframe not in self.timeframe_set:
            raise KeyError(f"single cell at timeframe {timeframe} does not exist in the trajectory")
        return self.get_single_cell(timeframe)

    def __iter__(self):
        return iter(self.timeframe_to_single_cell.values())

    def compute_features(self, feature_key: str, func: Callable):
        for sc in iter(self.timeframe_to_single_cell.values()):
            sc.add_feature(feature_key, func(sc))

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

    def write_json(self, path=None):
        if path is None:
            return json.dumps(self.to_dict())
        else:
            with open(path, "w+") as f:
                json.dump(self.to_dict(), f)

    def load_from_json_dict(self, json_dict, img_dataset=None, share_img_dataset=True):
        self.track_id = json_dict["track_id"]
        if img_dataset:
            self.raw_img_dataset = img_dataset
        else:
            self.raw_img_dataset = LiveCellImageDataset().load_from_json_dict(json_dict["dataset_info"])
        self.raw_total_timeframe = len(self.raw_img_dataset)
        self.timeframe_to_single_cell = {}
        for timeframe, sc in json_dict["timeframe_to_single_cell"].items():
            self.timeframe_to_single_cell[int(timeframe)] = SingleCellStatic(
                int(timeframe), img_dataset=self.raw_img_dataset
            ).load_from_json_dict(sc, img_dataset=self.raw_img_dataset)
            if img_dataset is None and share_img_dataset:
                img_dataset = self.raw_img_dataset
        self.timeframe_set = set(self.timeframe_to_single_cell.keys())
        return self

    def get_sc_feature_table(self):
        feature_table = None
        for sc in self:
            feature_series = sc.get_feature_pd_series()
            if feature_table is None:
                feature_table = pd.DataFrame(feature_series, columns=[str(sc.timeframe)])
            else:
                feature_table[str(sc.timeframe)] = feature_series
        feature_table = feature_table.transpose()
        return feature_table

    def get_sc_bboxes(self):
        bbox_list = []
        for sc in self:
            bbox_list.append(sc.bbox)
        return bbox_list

    def get_sc_napari_shapes(self):
        shape_dict = {}
        for sc in self:
            shape_dict[sc.timeframe] = sc.get_napari_shape_vec()
        return shape_dict

class SingleCellTrajectoryCollection:
    def __init__(self) -> None:
        self.track_id_to_trajectory = dict()
        self._iter_index = 0

    def __contains__(self, track_id):
        return track_id in self.track_id_to_trajectory

    def __getitem__(self, track_id):
        return self.get_trajectory(track_id)

    def __len__(self):
        return len(self.track_id_to_trajectory)

    def __iter__(self):
        return iter(self.track_id_to_trajectory.values())

    def add_trajectory(self, trajectory: SingleCellTrajectory):
        self.track_id_to_trajectory[trajectory.track_id] = trajectory

    def get_trajectory(self, track_id) -> SingleCellTrajectory:
        return self.track_id_to_trajectory[track_id]

    def to_json_dict(self):
        return {
            "track_id_to_trajectory": {
                int(track_id): trajectory.to_dict() for track_id, trajectory in self.track_id_to_trajectory.items()
            }
        }

    def write_json(self, path):
        with open(path, "w+") as f:
            json.dump(self.to_json_dict(), f)

    def load_from_json_dict(self, json_dict):
        self.track_id_to_trajectory = {}
        for track_id, trajectory_dict in json_dict["track_id_to_trajectory"].items():
            self.track_id_to_trajectory[int(float(track_id))] = SingleCellTrajectory().load_from_json_dict(
                trajectory_dict
            )
        return self

    def histogram_traj_length(self, ax=None, **kwargs):
        import seaborn as sns

        id_to_sc_trajs = self.track_id_to_trajectory
        all_traj_lengths = np.array([_traj.get_timeframe_span_length() for _traj in id_to_sc_trajs.values()])
        if ax is None:
            fig, ax = plt.subplots()
        sns.histplot(all_traj_lengths, bins=100, ax=ax, **kwargs)
        return ax
