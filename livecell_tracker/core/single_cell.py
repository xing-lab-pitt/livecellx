import itertools
import json
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import pandas as pd
from skimage.measure._regionprops import RegionProperties
from skimage.measure import regionprops

from livecell_tracker.core.datasets import LiveCellImageDataset
from livecell_tracker.core.sc_key_manager import SingleCellMetaKeyManager


# TODO: possibly refactor load_from_json methods into a mixin class
class SingleCellStatic:
    """Single cell at one time frame"""

    HARALICK_FEATURE_KEY = "_haralick"
    MORPHOLOGY_FEATURE_KEY = "_morphology"
    AUTOENCODER_FEATURE_KEY = "_autoencoder"
    CACHE_IMG_CROP_KEY = "_cached_img_crop"

    def __init__(
        self,
        timeframe: Optional[int] = None,
        bbox: Optional[np.array] = None,
        regionprops: Optional[RegionProperties] = None,
        img_dataset: Optional[LiveCellImageDataset] = None,
        mask_dataset: Optional[LiveCellImageDataset] = None,
        dataset_dict: Optional[Dict[str, LiveCellImageDataset]] = None,
        feature_dict: Optional[Dict[str, np.array]] = dict(),
        contour: Optional[np.array] = None,
        meta: Optional[Dict[str, object]] = None,
        uns: Optional[Dict[str, object]] = None,
        id: Optional[int] = None,  # TODO: automatically assign id (uuid),
        cache: Optional[Dict[str, object]] = None,  # TODO: now only image crop is cached
    ) -> None:
        """_summary_

        Parameters
        ----------
        timeframe : int
            _description_
        bbox : np.array, optional
            [x1, y1, x2, y2], by default None
            follwoing skimage convention: "Bounding box (min_row, min_col, max_row, max_col). Pixels belonging to the bounding box are in the half-open interval [min_row; max_row) and [min_col; max_col)."
        regionprops : RegionProperties, optional
            _description_, by default None
        img_dataset : _type_, optional
            _description_, by default None
        feature_dict : dict, optional
            _description_, by default {}
        contour:
            an array of contour coordinates [(x1, y1), (x2, y2), ...)], in a WHOLE image (not in a cropped image)
        """
        self.cache = cache
        self.regionprops = regionprops
        self.timeframe = timeframe
        self.img_dataset = img_dataset

        # TODO: discuss and decide whether to keep mask dataset
        self.mask_dataset = mask_dataset

        self.feature_dict = feature_dict
        self.bbox = bbox
        self.contour = np.array(contour, dtype=int)

        # infer bbox from regionprops
        if (bbox is None) and (regionprops is not None):
            self.bbox = regionprops.bbox
        elif (bbox is None) and contour is not None:
            self.update_contour(self.contour, update_bbox=True)
        # TODO: enable img_crops caching ONLY in RAM mode, otherwise caching these causes memory issues
        # self.raw_img = self.get_img()
        # self.img_crop = None
        # self.mask_crop = None
        self.meta = meta
        if self.meta is None:
            self.meta = dict()

        self.uns: dict = uns
        if self.uns is None:
            self.uns = dict()

        if dataset_dict:
            self.dataset_dict = dataset_dict
        else:
            self.dataset_dict = dict()

        if "raw" not in self.dataset_dict:
            self.dataset_dict["raw"] = self.img_dataset
        if "mask" not in self.dataset_dict:
            self.dataset_dict["mask"] = self.mask_dataset
        if self.img_dataset is None and "raw" in self.dataset_dict:
            self.img_dataset = self.dataset_dict["raw"]
        if self.mask_dataset is None and "mask" in self.dataset_dict:
            self.mask_dataset = self.dataset_dict["mask"]

    def compute_regionprops(self, crop=True):
        props = regionprops(
            label_image=self.get_contour_mask(crop=crop).astype(int), intensity_image=self.get_contour_img(crop=crop)
        )

        # TODO: multiple cell parts? WARNING in the future
        assert len(props) == 1, "contour mask should contain only one region"
        return props[0]

    # TODO: optimize compute overlap mask functions by taking union of two single cell's merged bboxes and then only operate on the union region to make the process faster
    def compute_overlap_mask(self, other_cell: "SingleCellStatic"):
        mask = self.get_contour_mask(crop=False).astype(bool)
        return np.logical_and(mask, other_cell.get_contour_mask(crop=False).astype(bool))

    def compute_overlap_percent(self, other_cell: "SingleCellStatic"):
        mask = self.get_contour_mask(crop=False).astype(bool)
        overlap_mask = self.compute_overlap_mask(other_cell)
        return np.sum(overlap_mask) / np.sum(mask)

    def compute_iou(self, other_cell: "SingleCellStatic"):
        mask = self.get_contour_mask(crop=False).astype(bool)
        overlap_mask = self.compute_overlap_mask(other_cell)
        return np.sum(overlap_mask) / (
            np.sum(mask) + np.sum(other_cell.get_contour_mask(crop=False).astype(bool)) - np.sum(overlap_mask)
        )

    def update_regionprops(self):
        self.regionprops = self.compute_regionprops()

    def get_contour(self) -> np.array:
        return np.copy(self.contour)

    def get_img(self):
        return self.img_dataset.get_img_by_time(self.timeframe)

    def get_mask(self, dtype=bool):
        if not self.mask_dataset is None:
            return self.mask_dataset[self.timeframe]
        elif self.contour is not None:
            shape = self.get_img().shape
            mask = np.zeros(shape, dtype=dtype)
            mask[self.contour[0, :], self.contour[1, :]] = True
            return mask
        else:
            raise ValueError("mask dataset and contour are both None")

    def get_bbox(self) -> np.array:
        if self.bbox is None:
            self.update_bbox()
        return np.array(self.bbox)

    @staticmethod
    def gen_skimage_bbox_img_crop(bbox, img, padding=0, preprocess_img_func=None):
        if preprocess_img_func is not None:
            img = preprocess_img_func(img)
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

    def get_img_crop(self, padding=0, bbox=None, **kwargs):
        if self.cache and SingleCellStatic.CACHE_IMG_CROP_KEY in self.uns:
            return self.uns[(SingleCellStatic.CACHE_IMG_CROP_KEY, padding)].copy()
        if bbox is None:
            bbox = self.bbox
        img_crop = SingleCellStatic.gen_skimage_bbox_img_crop(bbox=bbox, img=self.get_img(), padding=padding, **kwargs)
        # TODO: enable in RAM mode
        # if self.img_crop is None:
        #     self.img_crop = img_crop
        if self.cache:
            self.uns[(SingleCellStatic.CACHE_IMG_CROP_KEY, padding)] = img_crop
        return img_crop

    def get_mask_crop(self, bbox=None, dtype=bool, **kwargs):
        # TODO: enable in RAM mode
        # if self.mask_crop is None:
        #     self.mask_crop = SingleCellStatic.gen_skimage_bbox_img_crop(self.bbox, self.get_mask())
        if bbox is None:
            bbox = self.bbox
        return SingleCellStatic.gen_skimage_bbox_img_crop(bbox, self.get_mask(), **kwargs).astype(dtype=dtype)

    def update_bbox(self, bbox=None):
        if bbox is None and self.contour is not None:
            self.bbox = self.get_bbox_from_contour(self.contour)
        else:
            self.update_contour(self.contour, update_bbox=True)

        # TODO: enable in RAM mode
        # self.img_crop = None
        # self.mask_crop = None

    def update_contour(self, contour, update_bbox=True, dtype=int):
        self.contour = np.array(contour, dtype=dtype)
        # TODO: 3D?
        if update_bbox:
            self.bbox = self.get_bbox_from_contour(self.contour)

    def to_json_dict(self, dataset_json=True):
        """returns a dict that can be converted to json"""
        res = {
            "timeframe": int(self.timeframe),
            "bbox": list(np.array(self.bbox, dtype=float)),
            "feature_dict": self.feature_dict,
            "contour": self.contour.tolist(),
            "meta": self.meta,
        }
        if dataset_json:
            res["dataset_json"] = self.img_dataset.to_json_dict()
        return res

    def load_from_json_dict(self, json_dict, img_dataset=None):
        self.timeframe = json_dict["timeframe"]
        self.bbox = np.array(json_dict["bbox"], dtype=float)
        self.feature_dict = json_dict["feature_dict"]
        self.contour = np.array(json_dict["contour"], dtype=float)

        if "meta" in json_dict:
            self.meta = json_dict["meta"]
        if img_dataset is None and "dataset_json" in json_dict:
            self.img_dataset = LiveCellImageDataset().load_from_json_dict(json_dict["dataset_json"])
        else:
            self.img_dataset = img_dataset

        # TODO: discuss and decide whether to keep mask dataset
        # self.mask_dataset = LiveCellImageDataset(
        #     json_dict["dataset_name"] + "_mask", json_dict["dataset_path"]
        # )
        return self

    @staticmethod
    # TODO: check forward declaration change: https://peps.python.org/pep-0484/#forward-references
    def load_single_cells_json(path: str) -> List["SingleCellStatic"]:
        """load a json file containing a list of single cells

        Parameters
        ----------
        path :
        path to json file

        Returns
        -------
        _type_
            _description_
        """
        import json

        with open(path, "r") as f:
            sc_dict_list = json.load(f)
        single_cells = [SingleCellStatic().load_from_json_dict(data) for data in sc_dict_list]
        return single_cells

    @staticmethod
    def write_single_cells_json(single_cells: List["SingleCellStatic"], path: str):
        """write a json file containing a list of single cells

        Parameters
        ----------
        path :
        path to json file
        """
        import json

        with open(path, "w+") as f:
            json.dump([sc.to_json_dict() for sc in single_cells], f)

    def write_json(self, path=None):
        if path is None:
            return json.dumps(self.to_json_dict())
        else:
            with open(path, "w+") as f:
                json.dump(self.to_json_dict(), f)

    def get_contour_coords_on_crop(self, bbox=None, padding=0):
        if bbox is None:
            bbox = self.get_bbox()
        xs = self.contour[:, 0] - max(0, bbox[0] - padding)
        ys = self.contour[:, 1] - max(0, bbox[1] - padding)
        return np.array([xs, ys]).T

    def get_bbox_on_crop(self, bbox=None, padding=0):
        contours = self.get_contour_coords_on_crop(bbox=bbox, padding=padding)
        return self.get_bbox_from_contour(contours)

    def get_contour_coords_on_img_crop(self, padding=0) -> np.array:
        """a utility function to calculate pixel coord in image crop's coordinate system
            to draw contours on an image crop
        Parameters
        ----------
        padding : int, optional
            _description_, by default 0

        Returns
        -------
            returns contour coordinates in the cropped image's coordinate system
        """
        xs = self.contour[:, 0] - max(0, self.bbox[0] - padding)
        ys = self.contour[:, 1] - max(0, self.bbox[1] - padding)
        return np.array([xs, ys]).T

    def get_contour_mask_closed_form(self, padding=0, crop=True) -> np.array:
        """If contour points are pixel-wise closed, use this function to fill the contour."""
        import scipy.ndimage as ndimage

        contour = self.contour
        res_mask = np.zeros(self.get_img().shape, dtype=bool)
        # create a contour image by using the contour coordinates rounded to their nearest integer value
        res_mask[np.round(contour[:, 0]).astype("int"), np.round(contour[:, 1]).astype("int")] = 1
        # fill in the hole created by the contour boundary
        res_mask = ndimage.binary_fill_holes(res_mask)
        if crop:
            res_mask_crop = SingleCellStatic.gen_skimage_bbox_img_crop(self.bbox, res_mask, padding=padding)
            return res_mask_crop
        else:
            return res_mask

    @staticmethod
    def get_bbox_from_contour(contour, dtype=int):
        """get the bounding box of a contour"""
        return np.array(
            [np.min(contour[:, 0]), np.min(contour[:, 1]), np.max(contour[:, 0]) + 1, np.max(contour[:, 1]) + 1]
        ).astype(dtype)

    @staticmethod
    def gen_contour_mask(contour, img=None, shape=None, bbox=None, padding=0, crop=True, mask_val=255) -> np.array:
        from skimage.draw import line, polygon

        assert img is not None or shape is not None, "either img or shape must be provided"
        if bbox is None:
            if crop:
                bbox = SingleCellStatic.get_bbox_from_contour(contour)
            else:
                bbox = [0, 0, img.shape[0], img.shape[1]]

        if shape is None:
            res_shape = img.shape
        else:
            res_shape = shape

        res_mask = np.zeros(res_shape, dtype=bool)
        rows, cols = polygon(contour[:, 0], contour[:, 1])
        res_mask[rows, cols] = mask_val
        res_mask = SingleCellStatic.gen_skimage_bbox_img_crop(bbox, res_mask, padding=padding)
        return res_mask

    def get_contour_mask(self, padding=0, crop=True, bbox=None) -> np.array:
        """if contour points are not closed, use this function to fill the polygon points in self.contour"""
        contour = self.contour
        return SingleCellStatic.gen_contour_mask(contour, self.get_img(), bbox=bbox, padding=padding, crop=crop)

    def get_contour_img(self, crop=True, bg_val=0, **kwargs) -> np.array:
        """return a contour image with background set to background_val"""
        contour_mask = self.get_contour_mask(crop=crop, **kwargs).astype(bool)
        contour_img = self.get_img_crop(**kwargs) if crop else self.get_img()
        contour_img[np.logical_not(contour_mask)] = bg_val
        return contour_img

    get_sc_img = get_contour_img
    get_sc_mask = get_contour_mask

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
        # add time frame information
        res_series["t"] = self.timeframe
        return res_series

    def get_napari_shape_vec(self, coords):
        # TODO: napari shapes layer convention discussion...looks weird
        napari_shape_vec = [[self.timeframe] + list(coord) for coord in coords]
        return napari_shape_vec

    def get_napari_shape_bbox_vec(self):
        x1, y1, x2, y2 = self.bbox
        coords = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
        return self.get_napari_shape_vec(coords)

    def get_napari_shape_contour_vec(self, contour_sample_num=None):
        contour = self.contour
        if contour_sample_num is not None:
            contour = contour[:: int(len(contour) / contour_sample_num)]
        return self.get_napari_shape_vec(contour)

    def segment_by_detectron(self):
        pass

    def segment_by_cellpose(self):
        pass

    def show(self, crop=False, padding=0, ax: plt.Axes = None, **kwargs):
        if ax is None:
            ax = plt.gca()
        if crop:
            ax.imshow(self.get_img_crop(padding=padding), **kwargs)
        else:
            ax.imshow(self.get_img(), **kwargs)
        return ax

    def show_mask(self, crop=False, padding=0, ax: plt.Axes = None, **kwargs):
        if ax is None:
            ax = plt.gca()
        if crop:
            ax.imshow(self.get_mask_crop(padding=padding), **kwargs)
        else:
            ax.imshow(self.get_mask(), **kwargs)
        return ax

    def show_contour_mask(self, padding=0, ax: plt.Axes = None, crop=True, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.imshow(self.get_contour_mask(crop=crop, padding=padding), **kwargs)
        return ax

    def show_contour_img(self, padding=0, ax: plt.Axes = None, crop=True, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.imshow(self.get_contour_img(crop=crop, padding=padding), **kwargs)
        return ax

    def show_whole_img(self, ax: plt.Axes = None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.imshow(self.get_img(), **kwargs)
        return ax

    def show_whole_mask(self, ax: plt.Axes = None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.imshow(self.get_mask(), **kwargs)

    def show_panel(self, padding=0, figsize=(20, 10), **kwargs):
        crop = True
        fig, axes = plt.subplots(1, 5, figsize=figsize)
        self.show(ax=axes[0], crop=False, padding=padding, **kwargs)
        axes[0].set_title("img")
        self.show_mask(ax=axes[1], crop=False, padding=padding, **kwargs)
        axes[1].set_title("mask")
        self.show_contour_img(ax=axes[2], crop=crop, padding=padding, **kwargs)
        axes[2].set_title("contour_img")
        self.show_contour_mask(ax=axes[3], crop=crop, padding=padding, **kwargs)
        axes[3].set_title("contour_mask")
        self.show_mask(ax=axes[4], crop=True, padding=padding, **kwargs)
        axes[4].set_title("mask_crop")
        return axes

    def copy(self):
        import copy

        return copy.copy(self)

    def get_center(self, crop=True):
        return np.array(self.compute_regionprops(crop=crop).centroid)

    def _sc_matplotlib_bbox_patch(self, edgecolor="r", linewidth=1, **kwargs) -> patches.Rectangle:
        """
        A util function to return matplotlib rectangle patch for one sc's bounding box
        Parameters
        ----------
        edgecolor : str, optional
            _description_, by default 'r'
        linewidth : int, optional
            _description_, by default 1
        kwargs :
        """
        return patches.Rectangle(
            (self.bbox[1], self.bbox[0]),
            (self.bbox[3] - self.bbox[1]),
            (self.bbox[2] - self.bbox[0]),
            linewidth=linewidth,
            edgecolor=edgecolor,
            **kwargs,
        )


class SingleCellTrajectory:
    """
    Single cell trajectory containing trajectory information for one single cell at all timeframes.
    """

    def __init__(
        self,
        track_id: int = None,
        timeframe_to_single_cell: Dict[int, SingleCellStatic] = None,
        img_dataset: LiveCellImageDataset = None,
        mask_dataset: LiveCellImageDataset = None,
        extra_datasets: Dict[str, LiveCellImageDataset] = None,
        mother_trajectories=None,
        daughter_trajectories=None,
    ) -> None:
        self.timeframe_set = set()
        if timeframe_to_single_cell is None:
            self.timeframe_to_single_cell = dict()
        else:
            self.timeframe_to_single_cell = timeframe_to_single_cell
        self.img_dataset = img_dataset
        self.img_total_timeframe = len(img_dataset) if img_dataset is not None else None
        self.track_id = track_id
        self.mask_dataset = mask_dataset
        self.extra_datasets = extra_datasets
        if mother_trajectories is None:
            self.mother_trajectories: Set["SingleCellTrajectory"] = set()
        else:
            self.mother_trajectories = mother_trajectories

        if daughter_trajectories is None:
            self.daughter_trajectories: Set["SingleCellTrajectory"] = set()
        else:
            self.daughter_trajectories = daughter_trajectories

    def __len__(self):
        return self.get_timeframe_span_length()

    def __getitem__(self, timeframe: int) -> SingleCellStatic:
        if timeframe not in self.timeframe_set:
            raise KeyError(f"single cell at timeframe {timeframe} does not exist in the trajectory")
        return self.get_single_cell(timeframe)

    def __iter__(self):
        return iter(self.timeframe_to_single_cell.items())

    def compute_features(self, feature_key: str, func: Callable):
        """_summary_

        Parameters
        ----------
        feature_key : str
            _description_
        func : Callable
            _description_
        """
        for sc in iter(self.timeframe_to_single_cell.values()):
            sc.add_feature(feature_key, func(sc))

    def add_single_cell(self, timeframe, cell: SingleCellStatic):
        self.timeframe_to_single_cell[timeframe] = cell
        self.timeframe_set.add(timeframe)

    def get_img(self, timeframe):
        return self.timeframe_to_single_cell[timeframe].get_img()

    def get_mask(self, timeframe):
        return self.timeframe_to_single_cell[timeframe].get_mask()

    def get_timeframe_span(self):
        assert len(self.timeframe_set) > 0, "sct: timeframe set is empty."
        return (min(self.timeframe_set), max(self.timeframe_set))

    def get_timeframe_span_length(self):
        min_t, max_t = self.get_timeframe_span()
        return max_t - min_t

    def get_single_cell(self, timeframe: int) -> SingleCellStatic:
        return self.timeframe_to_single_cell[timeframe]

    def pop_single_cell(self, timeframe: int):
        self.timeframe_set.remove(timeframe)
        return self.timeframe_to_single_cell.pop(timeframe)

    def to_dict(self):
        res = {
            "track_id": int(self.track_id),
            "timeframe_to_single_cell": {
                int(float(timeframe)): sc.to_json_dict() for timeframe, sc in self.timeframe_to_single_cell.items()
            },
            "dataset_info": self.img_dataset.to_json_dict(),
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
            self.img_dataset = img_dataset
        else:
            self.img_dataset = LiveCellImageDataset().load_from_json_dict(json_dict["dataset_info"])
        self.img_total_timeframe = len(self.img_dataset)
        self.timeframe_to_single_cell = {}
        for timeframe, sc in json_dict["timeframe_to_single_cell"].items():
            self.timeframe_to_single_cell[int(timeframe)] = SingleCellStatic(
                int(timeframe), img_dataset=self.img_dataset
            ).load_from_json_dict(sc, img_dataset=self.img_dataset)
            if img_dataset is None and share_img_dataset:
                img_dataset = self.img_dataset
        self.timeframe_set = set(self.timeframe_to_single_cell.keys())
        return self

    def get_sc_feature_table(self):
        feature_table = None
        for timeframe, sc in self:
            assert timeframe == sc.timeframe, "timeframe mismatch"
            feature_series = sc.get_feature_pd_series()
            row_idx = "_".join([str(self.track_id), str(sc.timeframe)])
            if feature_table is None:
                feature_table = pd.DataFrame({row_idx: feature_series})
            else:
                feature_table[row_idx] = feature_series
        feature_table = feature_table.transpose()
        return feature_table

    def get_sc_bboxes(self):
        bbox_list = []
        for _, sc in self:
            bbox_list.append(sc.bbox)
        return bbox_list

    def get_scs_napari_shapes(self, bbox=False, contour_sample_num=20, return_scs=False):
        shapes_data = []
        scs = []
        for _, sc in self:
            scs.append(sc)
            if bbox:
                shapes_data.append(sc.get_napari_shape_bbox_vec())
            else:
                shapes_data.append(sc.get_napari_shape_contour_vec(contour_sample_num=contour_sample_num))
        if return_scs:
            return shapes_data, scs
        return shapes_data

    def add_nonoverlapping_sct(self, other_sct: "SingleCellTrajectory"):
        """add the other sct to this sct, but only add the non-overlapping single cells"""
        if len(self.timeframe_set.intersection(other_sct.timeframe_set)) > 0:
            raise ValueError("cannot add overlapping single cell trajectories")
        for timeframe, sc in other_sct:
            self.add_single_cell(timeframe, sc)

    def add_mother(self, mother_sct: "SingleCellTrajectory"):
        self.mother_trajectories.add(mother_sct)

    def add_daughter(self, daughter_sct: "SingleCellTrajectory"):
        self.daughter_trajectories.add(daughter_sct)

    def remove_mother(self, mother_sct: "SingleCellTrajectory"):
        self.mother_trajectories.remove(mother_sct)

    def remove_daughter(self, daughter_sct: "SingleCellTrajectory"):
        self.daughter_trajectories.remove(daughter_sct)

    def copy(self):
        import copy

        return copy.deepcopy(self)

    def subsct(self, min_time, max_time):
        """return a subtrajectory of this trajectory, with timeframes between min_time and max_time. Mother and daugher info will be copied if the min_time and max_time are the start and end of the new trajectory, respectively."""
        require_copy_mothers_info = False
        require_copy_daughters_info = False
        self_span = self.get_timeframe_span()

        # TODO: if time is float case, consider round-off errors
        if min_time == self_span[0]:
            require_copy_mothers_info = True
        if max_time == self_span[1]:
            require_copy_daughters_info = True

        sub_sct = SingleCellTrajectory()
        for timeframe, sc in self:
            if timeframe >= min_time and timeframe <= max_time:
                sub_sct.add_single_cell(timeframe, sc)
        if require_copy_mothers_info:
            sub_sct.mother_trajectories = self.mother_trajectories.copy()
        if require_copy_daughters_info:
            sub_sct.daughter_trajectories = self.daughter_trajectories.copy()
        return sub_sct

    def split(self, split_time) -> Tuple["SingleCellTrajectory", "SingleCellTrajectory"]:
        """split this trajectory into two trajectories: [start, split_time), [split_time, end], at the given split time"""
        if split_time not in self.timeframe_set:
            raise ValueError("split time not in this trajectory")
        sct1 = self.subsct(min(self.timeframe_set), split_time - 1)
        sct2 = self.subsct(split_time, max(self.timeframe_set))
        return sct1, sct2


class SingleCellTrajectoryCollection:
    def __init__(self) -> None:
        self.track_id_to_trajectory = dict()
        self._iter_index = 0

    def __contains__(self, track_id):
        return track_id in self.track_id_to_trajectory

    def __getitem__(self, track_id):
        return self.get_trajectory(track_id)

    def __setitem__(self, track_id, trajectory: SingleCellTrajectory):
        assert (
            track_id == trajectory.track_id
        ), "track_id mismatch (between [tacj_id] and [trajectory.track_id]): ({}, {})".format(
            track_id, trajectory.track_id
        )
        self.track_id_to_trajectory[track_id] = trajectory

    def __len__(self):
        return len(self.track_id_to_trajectory)

    def __iter__(self):
        return iter(self.track_id_to_trajectory.items())

    def add_trajectory(self, trajectory: SingleCellTrajectory):
        if trajectory.track_id is None:
            trajectory.track_id = self._next_track_id()
        if trajectory.track_id in self.track_id_to_trajectory:
            raise ValueError("trajectory with track_id {} already exists".format(trajectory.track_id))
        self[trajectory.track_id] = trajectory

    def get_trajectory(self, track_id) -> SingleCellTrajectory:
        return self.track_id_to_trajectory[track_id]

    def pop_trajectory(self, track_id):
        return self.track_id_to_trajectory.pop(track_id)

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

    def get_feature_table(self) -> pd.DataFrame:
        feature_table = None
        for track_id, trajectory in self:
            assert track_id == trajectory.track_id, "track_id mismatch"
            sc_feature_table = trajectory.get_sc_feature_table()
            if feature_table is None:
                feature_table = sc_feature_table
            else:
                feature_table = pd.concat([feature_table, sc_feature_table])
        return feature_table

    def pop_trajectory(self, track_id):
        return self.track_id_to_trajectory.pop(track_id)

    def get_track_ids(self):
        return sorted(list(self.track_id_to_trajectory.keys()))

    def subset(self, track_ids):
        new_sc_traj_collection = SingleCellTrajectoryCollection()
        for track_id in track_ids:
            new_sc_traj_collection.add_trajectory(self.get_trajectory(track_id))
        return new_sc_traj_collection

    def subset_random(self, n):
        import random

        track_ids = self.get_track_ids()
        random.shuffle(track_ids)
        return self.subset(track_ids[:n])

    def _next_track_id(self):
        return max(self.get_track_ids()) + 1
