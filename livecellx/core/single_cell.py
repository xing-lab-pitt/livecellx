import itertools
import json
import copy
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union
from collections import deque
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import pandas as pd
from skimage.measure._regionprops import RegionProperties
from skimage.measure import regionprops
import tqdm
import uuid

from livecellx.core.datasets import LiveCellImageDataset, SingleImageDataset
from livecellx.core.io_utils import LiveCellEncoder
from livecellx.core.parallel import parallelize
from livecellx.core.sc_key_manager import SingleCellMetaKeyManager as SCKM
from livecellx.livecell_logger import main_info, main_warning, main_exception


# TODO: possibly refactor load_from_json methods into a mixin class
class SingleCellStatic:
    """Single cell at one time frame"""

    HARALICK_FEATURE_KEY = "_haralick"
    MORPHOLOGY_FEATURE_KEY = "_morphology"
    AUTOENCODER_FEATURE_KEY = "_autoencoder"
    CACHE_IMG_CROP_KEY = "_cached_img_crop"
    id_generator = itertools.count()

    def __init__(
        self,
        timeframe: Optional[int] = None,
        bbox: Optional[np.array] = None,
        regionprops: Optional[RegionProperties] = None,
        img_dataset: Optional[LiveCellImageDataset] = None,
        mask_dataset: Optional[LiveCellImageDataset] = None,
        dataset_dict: Optional[Dict[str, LiveCellImageDataset]] = None,
        feature_dict: Optional[Dict[str, np.array]] = None,
        contour: Optional[np.array] = None,
        meta: Optional[Dict[str, object]] = None,
        uns: Optional[Dict[str, object]] = None,
        id: Optional[int] = None,  # TODO: automatically assign id (incremental or uuid),
        cache: Optional[Dict] = None,  # TODO: now only image crop is cached
        update_mask_dataset_by_contour=False,
        empty_cell=False,
        tmp=None,
        use_cache_contour_mask=False,
        use_img_crop_cache=False,
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
        empty_cell: bool
            use when intend to create an empty cell object. suppress the empty warning.
        """
        if cache is None:
            self.cache = dict()
        else:
            self.cache = cache

        self.regionprops = regionprops
        self.timeframe = timeframe
        self.img_dataset = img_dataset

        # TODO: discuss and decide whether to keep mask dataset
        self.mask_dataset = mask_dataset

        self.feature_dict = feature_dict
        if self.feature_dict is None:
            self.feature_dict = dict()

        self.bbox = bbox
        if contour is None and not empty_cell:
            main_warning(">>> [SingleCellStatic] WARNING: contour is None, please check if this is expected.")
            contour = np.array([], dtype=int)
        self.contour = contour

        # infer bbox from regionprops
        if (bbox is None) and (regionprops is not None):
            self.bbox = regionprops.bbox
        elif (bbox is None) and contour is not None:
            self.update_contour(self.contour, update_bbox=True, update_mask_dataset=update_mask_dataset_by_contour)
        # TODO: enable img_crops caching ONLY in RAM mode, otherwise caching these causes memory issues
        # self.raw_img = self.get_img()
        # self.img_crop = None
        # self.mask_crop = None
        self.meta: Dict = meta
        if self.meta is None:
            self.meta = dict()

        self.uns: dict = uns
        if self.uns is None:
            self.uns = dict()

        # TODO: [smz] add i/o method written and test comparison for dataset_dict
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
        if id is not None:
            self.id = id
        else:
            # self.id = SingleCellStatic.id_generator.__next__()
            self.id = uuid.uuid4()

        if tmp is not None:
            self.tmp = tmp
        else:
            self.tmp = dict()

        self.cache_img_shape = None
        self.enable_cache_contour_mask = use_cache_contour_mask
        self.enable_img_crop_cache = use_img_crop_cache

    def __repr__(self) -> str:
        return f"SingleCellStatic(id={self.id}, timeframe={self.timeframe}, bbox={self.bbox})"

    # TODO: [smz] implement this
    # def equals(self, other_cell: "SingleCellStatic", bbox=None, padding=0, iou_threshold=0.5):

    def compute_regionprops(self, crop=True, ignore_errors=False):
        props = regionprops(
            label_image=self.get_contour_mask(crop=crop).astype(int), intensity_image=self.get_contour_img(crop=crop)
        )

        # TODO: multiple cell parts? WARNING in the future
        if not ignore_errors:
            assert (
                len(props) == 1
            ), "contour mask should contain only one region. You can set ignore_errors=True to ignore this error/check."
        return props[0]

    # TODO: optimize compute overlap mask functions by taking union of two single cell's merged bboxes and then only operate on the union region to make the process faster
    def compute_overlap_mask(self, other_cell: "SingleCellStatic", bbox=None):
        def is_overlap_bbox(bbox1, bbox2):
            return not (bbox1[0] > bbox2[2] or bbox1[2] < bbox2[0] or bbox1[1] > bbox2[3] or bbox1[3] < bbox2[1])

        if not is_overlap_bbox(bbox, other_cell.bbox):
            return np.zeros(self.get_img().shape, dtype=bool)
        if bbox is None:
            bbox = self.bbox
        mask = self.get_contour_mask(bbox=bbox).astype(bool)
        return np.logical_and(mask, other_cell.get_contour_mask(bbox=bbox).astype(bool))

    def compute_overlap_percent(self, other_cell: "SingleCellStatic", bbox=None):
        """compute overlap defined by: overlap = intersection / self's area. It is different from IoU.

        Parameters
        ----------
        other_cell : SingleCellStatic
            _description_

        Returns
        -------
        _type_
            _description_
        """
        if bbox is None:
            bbox = self.bbox
        mask = self.get_contour_mask(bbox=bbox).astype(bool)
        overlap_mask = self.compute_overlap_mask(other_cell, bbox=bbox)
        return np.sum(overlap_mask) / np.sum(mask)

    def compute_iou(self, other_cell: "SingleCellStatic", bbox=None):
        # Compare bbox, if not overlap, return 0
        other_cell_bbox = other_cell.bbox
        if not (
            other_cell_bbox[0] <= self.bbox[2]
            and other_cell_bbox[2] >= self.bbox[0]
            and other_cell_bbox[1] <= self.bbox[3]
            and other_cell_bbox[3] >= self.bbox[1]
        ):
            return 0.0

        if bbox is None:
            # bbox = self.bbox
            # Take the merged bbox of two cells
            bbox = np.array(
                [
                    min(self.bbox[0], other_cell_bbox[0]),
                    min(self.bbox[1], other_cell_bbox[1]),
                    max(self.bbox[2], other_cell_bbox[2]),
                    max(self.bbox[3], other_cell_bbox[3]),
                ]
            )
        mask = self.get_contour_mask(bbox=bbox).astype(bool)
        overlap_mask = self.compute_overlap_mask(other_cell, bbox=bbox)
        return np.sum(overlap_mask) / (
            np.sum(mask) + np.sum(other_cell.get_contour_mask(bbox=bbox).astype(bool)) - np.sum(overlap_mask)
        )

    def compute_iomin(self, other_cell: "SingleCellStatic", bbox=None):
        # Compare bbox, if not overlap, return 0
        other_cell_bbox = other_cell.bbox
        if not (
            other_cell_bbox[0] <= self.bbox[2]
            and other_cell_bbox[2] >= self.bbox[0]
            and other_cell_bbox[1] <= self.bbox[3]
            and other_cell_bbox[3] >= self.bbox[1]
        ):
            return 0.0

        if bbox is None:
            # bbox = self.bbox
            # Take the merged bbox of two cells
            bbox = np.array(
                [
                    min(self.bbox[0], other_cell_bbox[0]),
                    min(self.bbox[1], other_cell_bbox[1]),
                    max(self.bbox[2], other_cell_bbox[2]),
                    max(self.bbox[3], other_cell_bbox[3]),
                ]
            )
        mask = self.get_contour_mask(bbox=bbox).astype(bool)
        overlap_mask = self.compute_overlap_mask(other_cell, bbox=bbox)
        return np.sum(overlap_mask) / min(
            np.sum(mask.flatten()), np.sum(other_cell.get_contour_mask(bbox=bbox).astype(bool).flatten())
        )

    def update_regionprops(self):
        self.regionprops = self.compute_regionprops()

    def get_contour(self) -> np.array:
        return np.copy(self.contour)

    def get_img(self):
        return self.img_dataset.get_img_by_time(self.timeframe)

    def get_mask(self, dtype=bool):
        if isinstance(self.mask_dataset, SingleImageDataset) or isinstance(self.mask_dataset, LiveCellImageDataset):
            try:
                return self.mask_dataset.get_img_by_time(time=self.timeframe)
            except Exception as e:
                main_exception(
                    f"Error getting mask for single cell {self.id} at timeframe {self.timeframe} from presented mask dataset. Using contour mask instead."
                )
        if self.contour is not None:
            return self.get_contour_mask(crop=False, dtype=dtype)

        raise ValueError("mask dataset and contour are both None")

    def get_label_mask(self, dtype=int):
        mask = self.get_mask(dtype=dtype)
        return mask

    def get_bbox(self) -> np.array:
        if self.bbox is None:
            self.update_bbox()
        return np.array(self.bbox)

    @staticmethod
    def gen_skimage_bbox_img_crop(bbox, img, padding=0, pad_zeros=False, preprocess_img_func=None):
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

        # pad the image if the bbox is too close to the edge
        if pad_zeros:
            if min_x == 0:
                img_crop = np.pad(img_crop, ((padding, 0), (0, 0), (0, 0)), mode="constant")
            if min_y == 0:
                img_crop = np.pad(img_crop, ((0, 0), (padding, 0), (0, 0)), mode="constant")
            if max_x + padding > img.shape[0]:
                img_crop = np.pad(img_crop, ((0, padding), (0, 0), (0, 0)), mode="constant")
            if max_y + padding > img.shape[1]:
                img_crop = np.pad(img_crop, ((0, 0), (0, padding), (0, 0)), mode="constant")

        return img_crop

    def get_center(self, crop=False):
        return np.array(self.compute_regionprops(crop=crop).centroid)

    def get_img_crop(self, padding=0, bbox=None, **kwargs):
        cache_key = (SingleCellStatic.CACHE_IMG_CROP_KEY, padding, tuple(bbox if bbox is not None else [-1]))
        if self.enable_img_crop_cache and SingleCellStatic.CACHE_IMG_CROP_KEY in self.uns:
            return self.cache[cache_key].copy()
        if bbox is None:
            bbox = self.bbox
        img_crop = SingleCellStatic.gen_skimage_bbox_img_crop(bbox=bbox, img=self.get_img(), padding=padding, **kwargs)
        # TODO: enable in RAM mode
        # if self.img_crop is None:
        #     self.img_crop = img_crop
        if self.enable_img_crop_cache:
            self.cache[cache_key] = img_crop
        return img_crop

    def get_img_shape(self, use_cache=True):
        if use_cache and self.cache_img_shape:
            return self.cache_img_shape
        else:
            self.cache_img_shape = self.get_img().shape
            return self.cache_img_shape

    def get_mask_crop(self, bbox=None, dtype=bool, **kwargs):
        # TODO: enable in RAM mode
        # if self.mask_crop is None:
        #     self.mask_crop = SingleCellStatic.gen_skimage_bbox_img_crop(self.bbox, self.get_mask())
        if bbox is None:
            bbox = self.bbox
        return SingleCellStatic.gen_skimage_bbox_img_crop(bbox, self.get_mask(), **kwargs).astype(dtype=dtype)

    def get_label_mask_crop(self, bbox=None, dtype=int, **kwargs):
        return self.get_mask_crop(bbox=bbox, dtype=dtype, **kwargs)

    def update_bbox(self, bbox=None):
        if bbox is None and self.contour is not None:
            self.bbox = self.get_bbox_from_contour(self.contour)
        else:
            self.update_contour(self.contour, update_bbox=True)

        # TODO: enable in RAM mode
        # self.img_crop = None
        # self.mask_crop = None

    def clear_contour_cache(self):
        for key in self.cache:
            if key[0] == "contour_mask":
                self.cache.pop(key, None)
            elif key[0] == "contour_label_mask":
                self.cache.pop(key, None)

    def update_contour(self, contour, update_bbox=True, update_mask_dataset=True, dtype=int):
        self.contour = np.array(contour, dtype=dtype)
        self.clear_contour_cache()
        if update_mask_dataset:
            new_mask = SingleCellStatic.gen_contour_mask(self.contour, img=self.get_img(), crop=False, dtype=bool)
            self.mask_dataset = SingleImageDataset(new_mask)

        if len(contour) == 0:
            return
        # TODO: 3D?
        if update_bbox:
            self.bbox = self.get_bbox_from_contour(self.contour)

    def update_sc_mask_by_crop(self, mask, padding_pixels=np.zeros(2, dtype=int), bbox: np.array = None):
        """
        Updates the single cell mask by cropping the input mask to the bounding box of the single cell and updating the contour.

        Parameters
        ----------
        mask : np.ndarray
            The input mask to crop.
        padding_pixels : np.ndarray, optional
            The number of pixels to pad the bounding box by, by default np.zeros(2, dtype=int).
        bbox : np.ndarray, optional
            The bounding box of the single cell, by default None.

        Raises
        ------
        AssertionError
            If the input mask has more than one contour.
        """
        from livecellx.segment.utils import find_contours_opencv

        if isinstance(padding_pixels, int):
            padding_pixels = np.array([padding_pixels, padding_pixels])
        contours = find_contours_opencv(mask)
        assert len(contours) == 1, f"Input mask does not have exactly one contour. #contours: {len(contours)}"
        if bbox is None:
            bbox = self.bbox

        self.contour = contours[0] + [bbox[0], bbox[1]] - padding_pixels
        old_bbox = self.bbox
        self.update_bbox()
        new_whole_mask = self.get_mask().copy()
        # use self bbox to update the new_whole_mask
        x_min = max(old_bbox[0] - padding_pixels[0], 0)
        x_max = min(old_bbox[2] + padding_pixels[0], new_whole_mask.shape[0])
        y_min = max(old_bbox[1] - padding_pixels[1], 0)
        y_max = min(old_bbox[3] + padding_pixels[1], new_whole_mask.shape[1])
        new_whole_mask[x_min:x_max, y_min:y_max] = mask
        self.mask_dataset = SingleImageDataset(new_whole_mask)

    def to_json_dict(self, include_dataset_json=False, dataset_json_dir=None):
        """returns a dict that can be converted to json"""
        # TODO: add arg to let users define their own json dataset paths
        if self.img_dataset is not None:
            self.meta[SCKM.JSON_IMG_DATASET_PATH] = str(
                self.img_dataset.get_default_json_path(out_dir=dataset_json_dir)
            )
            self.img_dataset.write_json(out_dir=dataset_json_dir, overwrite=False)
        if isinstance(self.mask_dataset, SingleImageDataset):
            pass
        elif self.mask_dataset is not None:
            self.meta[SCKM.JSON_MASK_DATASET_PATH] = str(
                self.mask_dataset.get_default_json_path(out_dir=dataset_json_dir)
            )
            self.mask_dataset.write_json(out_dir=dataset_json_dir, overwrite=False)

        res = {
            "timeframe": int(self.timeframe),
            "bbox": list(np.array(self.bbox, dtype=float)) if self.bbox is not None else None,
            "feature_dict": dict(self.feature_dict),
            "contour": self.contour.tolist() if self.contour is not None else None,
            "meta": dict(self.meta),
            "id": str(self.id),
            "uns": dict(self.uns),
        }

        if include_dataset_json:
            res["dataset_json"] = self.img_dataset.to_json_dict()

        if dataset_json_dir:
            # make dataset_json_dir posix
            dataset_json_dir = Path(dataset_json_dir).as_posix()
            res["dataset_json_dir"] = str(dataset_json_dir)

        return res

    def load_from_json_dict(self, json_dict, img_dataset=None, mask_dataset=None):
        """load from json dict

        Parameters
        ----------
        json_dict : _type_
            _description_
        img_dataset : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        self.timeframe = json_dict["timeframe"]
        self.bbox = np.array(json_dict["bbox"], dtype=float)
        self.feature_dict = json_dict["feature_dict"]
        # convert dictionary feature to series
        for key in self.feature_dict:
            if isinstance(self.feature_dict[key], dict):
                self.feature_dict[key] = pd.Series(self.feature_dict[key])
        self.contour = np.array(json_dict["contour"], dtype=float)
        self.id = json_dict["id"]

        if "meta" in json_dict and json_dict["meta"] is not None:
            self.meta = json_dict["meta"]

        # Set dataset paths, prioritizing those in the meta attribute. If not found, check the root of the json dictionary.
        img_dataset_path = None
        mask_dataset_path = None

        if SCKM.JSON_IMG_DATASET_PATH in self.meta:
            img_dataset_path = self.meta[SCKM.JSON_IMG_DATASET_PATH]
        elif SCKM.JSON_IMG_DATASET_PATH in json_dict:
            img_dataset_path = json_dict[SCKM.JSON_IMG_DATASET_PATH]
            self.meta[SCKM.JSON_IMG_DATASET_PATH] = img_dataset_path

        if SCKM.JSON_MASK_DATASET_PATH in self.meta:
            mask_dataset_path = self.meta[SCKM.JSON_MASK_DATASET_PATH]
        elif SCKM.JSON_MASK_DATASET_PATH in json_dict:
            mask_dataset_path = json_dict[SCKM.JSON_MASK_DATASET_PATH]
            self.meta[SCKM.JSON_MASK_DATASET_PATH] = mask_dataset_path

        self.img_dataset = img_dataset
        self.mask_dataset = mask_dataset

        if self.img_dataset is None and img_dataset_path:
            self.img_dataset = LiveCellImageDataset.load_from_json_file(path=self.meta[SCKM.JSON_IMG_DATASET_PATH])

        if self.mask_dataset is None and mask_dataset_path:
            self.mask_dataset = LiveCellImageDataset.load_from_json_file(path=self.meta[SCKM.JSON_MASK_DATASET_PATH])

        # TODO: discuss and decide whether to keep mask dataset
        # self.mask_dataset = LiveCellImageDataset(
        #     json_dict["dataset_name"] + "_mask", json_dict["dataset_path"]
        # )

        # Load uns
        if "uns" in json_dict:
            self.uns = json_dict["uns"]
        return self

    inflate_from_json_dict = load_from_json_dict

    @staticmethod
    # TODO: check forward declaration change: https://peps.python.org/pep-0484/#forward-references
    def load_single_cells_json(path: Union[Path, str], verbose=False) -> List["SingleCellStatic"]:
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
        if verbose:
            main_info("loading single cells from json file: " + str(path))
        with open(path, "r") as f:
            sc_json_dict_list = json.load(f)

        # contour = [] here to suppress warning
        single_cells = []
        for sc_json_dict in tqdm.tqdm(sc_json_dict_list, desc="constructing single cells from json dict"):
            # Load the single cell from json dict
            sc = SingleCellStatic(contour=[]).load_from_json_dict(sc_json_dict)
            single_cells.append(sc)
        if verbose:
            main_info("Done constructing single cells from json dict")
            main_info("loaded " + str(len(single_cells)) + " single cells")
        return single_cells

    @staticmethod
    # TODO: check forward declaration change: https://peps.python.org/pep-0484/#forward-references
    def load_single_cells_jsons(paths: str) -> List["SingleCellStatic"]:
        all_scs = []
        for path in paths:
            single_cells = SingleCellStatic.load_single_cells_json(path=path)
            for sc in single_cells:
                sc.meta["src_json"] = path
            all_scs.extend(single_cells)
        return all_scs

    @staticmethod
    def write_single_cells_json(
        single_cells: List["SingleCellStatic"], path: Union[str, Path], dataset_dir: str = None, return_list=False
    ):
        """Write a JSON file containing a list of single cells.

        Parameters
        ----------
        single_cells : List of SingleCellStatic
            The list of single cells to be written to the JSON file.
        path : Union[str, Path]
            The path to the JSON file.
        dataset_dir : str, optional
            The path to the dataset directory, by default None.
        return_list : bool, optional
            Whether to return the list of single cell JSON dictionaries, by default False.

        Returns
        -------
        List of dict
            The list of single cell JSON dictionaries, if `return_list` is True.

        Raises
        ------
        TypeError
            If there is an error writing the JSON file due to non-serializable attributes.

        Notes
        -----
        This function writes a JSON file containing a list of single cells. Each single cell is converted to a JSON dictionary
        using the `to_json_dict` method of the `SingleCellStatic` class. The `include_dataset_json` parameter of `to_json_dict`
        is set to False, and the `dataset_json_dir` parameter is set to the `dataset_dir` argument passed to this function.

        If `dataset_dir` is not provided, it is derived from the parent directory of the `path` argument by appending "/datasets".

        The `img_dataset` and `mask_dataset` attributes of each single cell are expected to be instances of classes that have a
        `write_json` method. The `write_json` method is called on the `img_dataset` and `mask_dataset` objects to write their
        respective JSON files to the `dataset_dir`.

        If `return_list` is True, the function returns the list of single cell JSON dictionaries.

        If there is an error writing the JSON file due to non-serializable attributes, a `TypeError` is raised.

        Example
        -------
        single_cells = [single_cell1, single_cell2, single_cell3]
        path = "/path/to/single_cells.json"
        dataset_dir = "/path/to/datasets"
        write_single_cells_json(single_cells, path, dataset_dir, return_list=True)
        """
        import json

        if dataset_dir is None:
            dataset_dir = Path(path).parent / "datasets"
        all_sc_jsons = []
        for sc in single_cells:
            sc_json = sc.to_json_dict(include_dataset_json=False, dataset_json_dir=dataset_dir)
            img_dataset = sc.img_dataset
            mask_dataset = sc.mask_dataset
            img_dataset.write_json(out_dir=dataset_dir, overwrite=False)
            if mask_dataset is not None:
                mask_dataset.write_json(out_dir=dataset_dir, overwrite=False)
            all_sc_jsons.append(sc_json)

        with open(path, "w+") as f:
            try:
                json.dump(all_sc_jsons, f, cls=LiveCellEncoder)
            except TypeError as e:
                main_exception("sample sc:" + str(all_sc_jsons[0]))
                main_exception("Error writing json file. Check that all attributes are serializable.")
                print(e)
            except Exception as e:
                print(e)

        if return_list:
            return all_sc_jsons

    def write_json(self, path=None, dataset_json_dir=None):
        # TODO: discuss with the team
        # if dataset_json_dir is None:
        #     dataset_json_dir = os.path.dirname(path)

        json_dict = self.to_json_dict(dataset_json_dir=dataset_json_dir)

        if dataset_json_dir is not None:
            self.img_dataset.write_json(out_dir=dataset_json_dir, overwrite=False)
            self.mask_dataset.write_json(out_dir=dataset_json_dir, overwrite=False)
        if self.img_dataset is not None and SCKM.JSON_IMG_DATASET_PATH in json_dict:
            img_dataset_dir = os.path.dirname(json_dict[SCKM.JSON_IMG_DATASET_PATH])
            self.img_dataset.write_json(out_dir=img_dataset_dir, overwrite=False)
        if self.mask_dataset is not None and SCKM.JSON_MASK_DATASET_PATH in json_dict:
            mask_dataset_dir = os.path.dirname(json_dict[SCKM.JSON_MASK_DATASET_PATH])
            self.mask_dataset.write_json(out_dir=mask_dataset_dir, overwrite=False)

        if path is None:
            return json.dumps(json_dict, cls=LiveCellEncoder)
        else:
            with open(path, "w+") as f:
                json.dump(json_dict, f, cls=LiveCellEncoder)

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
        """
        A utility function to calculate pixel coord in image crop's coordinate system
        to draw contours on an image crop.

        Parameters
        ----------
        padding : int, optional
            Padding value to be used in the calculations, by default 0

        Returns
        -------
        np.array
            Returns contour coordinates in the cropped image's coordinate system
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
    def gen_contour_mask(
        contour, img=None, shape=None, bbox=None, padding=0, crop=True, mask_val=255, dtype=bool
    ) -> np.array:  #
        # TODO: optimize: we do not need img here but shape of img.
        from skimage.draw import line, polygon

        assert img is not None or shape is not None, "either img or shape must be provided"
        if shape is None:
            res_shape = img.shape
        else:
            res_shape = shape

        if bbox is None:
            if crop:
                bbox = SingleCellStatic.get_bbox_from_contour(contour)
            else:
                bbox = [0, 0, res_shape[0], res_shape[1]]

        res_mask = np.zeros(res_shape, dtype=dtype)
        rows, cols = polygon(contour[:, 0], contour[:, 1])
        res_mask[rows, cols] = mask_val
        res_mask = SingleCellStatic.gen_skimage_bbox_img_crop(bbox, res_mask, padding=padding)
        return res_mask

    def get_contour_mask(self, padding=0, crop=True, bbox=None, dtype=bool) -> np.array:
        hash_key: Tuple = ("contour_mask", padding, crop, tuple(bbox if bbox is not None else [-1]))
        if self.enable_cache_contour_mask and hash_key in self.cache:
            return self.cache[hash_key]
        contour = self.contour
        res = SingleCellStatic.gen_contour_mask(
            contour, shape=self.get_img_shape(), bbox=bbox, padding=padding, crop=crop, dtype=dtype
        )
        if self.enable_cache_contour_mask:
            self.cache[hash_key] = res
        return res

    def get_contour_label_mask(self, padding=0, crop=True, bbox=None, dtype=int) -> np.array:
        hash_key: Tuple = ("contour_mask", padding, crop, tuple(bbox if bbox is not None else [-1]))
        if self.enable_cache_contour_mask and hash_key in self.cache:
            return self.cache[hash_key]
        contour = self.contour
        res = SingleCellStatic.gen_contour_mask(
            contour, shape=self.get_img_shape(), bbox=bbox, padding=padding, crop=crop, dtype=dtype
        )
        if self.enable_cache_contour_mask:
            self.cache[hash_key] = res
        return res

    def get_contour_img(self, crop=True, bg_val=0, **kwargs) -> np.array:
        """return a contour image with out of self cell region set to background_val"""

        # TODO: filter kwargs for contour mask case. (currently using the same kwargs as self.gen_skimage_bbox_img_crop)
        # Do not preprocess the mask when generating the sc image
        mask_kwargs = kwargs.copy()
        if "preprocess_img_func" in mask_kwargs:
            mask_kwargs.pop("preprocess_img_func")
        contour_mask = self.get_contour_mask(crop=crop, **mask_kwargs).astype(bool)

        contour_img = self.get_img_crop(**kwargs) if crop else self.get_img()
        contour_img[np.logical_not(contour_mask)] = bg_val
        return contour_img

    get_sc_img = get_contour_img
    get_sc_mask = get_contour_mask
    get_sc_label_mask = get_contour_label_mask

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
            elif isinstance(features, list):
                tmp_series = pd.Series(features)
            else:
                raise TypeError(
                    f"feature type:{type(features)} not supported, must be list, numpy array or pandas series"
                )
            tmp_series = tmp_series.add_prefix(feature_name + "_")
            if res_series is None:
                res_series = tmp_series
            else:
                res_series = pd.concat([res_series, tmp_series])
        # add time frame information
        if res_series is None:
            main_info("WARNING: no features found for single cell")
            res_series = pd.Series()
        res_series["t"] = self.timeframe
        res_series["sc_id"] = self.id
        return res_series

    def get_napari_shape_vec(self, coords):
        # TODO: napari shapes layer convention discussion...looks weird
        napari_shape_vec = [[self.timeframe] + list(coord) for coord in coords]
        return napari_shape_vec

    def get_napari_shape_bbox_vec(self):
        x1, y1, x2, y2 = self.bbox
        coords = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
        return self.get_napari_shape_vec(coords)

    def get_napari_shape_contour_vec(self, contour_sample_num: float = np.inf):
        if contour_sample_num is None:
            contour_sample_num = np.inf
        contour = self.contour
        if len(contour) == 0:
            return []
        slice_step = int(len(contour) / contour_sample_num)
        slice_step = max(slice_step, 1)  # make sure slice_step is at least 1
        if contour_sample_num is not None:
            contour = contour[::slice_step]
        return self.get_napari_shape_vec(contour)

    def sample_contour_point(self, max_contour_num):
        """sample contour points from the trajectory and modify contour attr"""
        contour = self.contour
        if len(contour) > max_contour_num:
            contour = contour[:: int(len(contour) / max_contour_num)]
        self.update_contour(contour)

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

    def show_label_mask(self, padding=0, ax: plt.Axes = None, crop=True, **kwargs):
        if ax is None:
            ax = plt.gca()
        if crop:
            ax.imshow(self.get_label_mask_crop(padding=padding), **kwargs)
        else:
            ax.imshow(self.get_label_mask(crop=crop, padding=padding), **kwargs)
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
        fig, axes = plt.subplots(1, 6, figsize=figsize)
        self.show(ax=axes[0], crop=False, padding=padding, **kwargs)
        axes[0].set_title("img")
        self.show_mask(ax=axes[1], crop=False, padding=padding, **kwargs)
        axes[1].set_title("mask")
        self.show_contour_img(ax=axes[2], crop=crop, padding=padding, **kwargs)
        axes[2].set_title("contour_img")
        self.show_contour_mask(ax=axes[3], crop=crop, padding=padding, **kwargs)
        axes[3].set_title("contour_mask")
        self.show_label_mask(ax=axes[4], crop=True, padding=padding, **kwargs)
        axes[4].set_title("label_crop")
        self.show(ax=axes[5], crop=True, padding=padding, **kwargs)
        axes[5].set_title("img_crop")
        return axes

    def copy(self):
        import copy

        return copy.copy(self)

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

    META_MOTHER_IDS = "mother_trajectory_ids"
    META_DAUGHTER_IDS = "daughter_trajectory_ids"

    def __init__(
        self,
        track_id: int = None,
        timeframe_to_single_cell: Dict[int, SingleCellStatic] = None,
        img_dataset: LiveCellImageDataset = None,
        mask_dataset: LiveCellImageDataset = None,
        extra_datasets: Dict[str, LiveCellImageDataset] = None,
        mother_trajectories=None,
        daughter_trajectories=None,
        meta: Dict[str, Any] = None,
        tmp: Dict[str, Any] = None,
    ) -> None:
        if timeframe_to_single_cell is None:
            self.timeframe_to_single_cell = dict()
        else:
            self.timeframe_to_single_cell = timeframe_to_single_cell
        self.timeframe_set = set(self.timeframe_to_single_cell.keys())
        self.times = sorted(self.timeframe_set)

        self.img_dataset = img_dataset
        self.img_total_timeframe = len(img_dataset) if img_dataset is not None else None
        self.track_id = track_id
        if track_id is None:
            self.track_id = uuid.uuid4()
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

        if meta is not None:
            self.meta = meta
        else:
            self.meta = {SingleCellTrajectory.META_MOTHER_IDS: [], SingleCellTrajectory.META_DAUGHTER_IDS: []}
            self.meta[SingleCellTrajectory.META_MOTHER_IDS] = [mother.track_id for mother in self.mother_trajectories]
            self.meta[SingleCellTrajectory.META_DAUGHTER_IDS] = [
                daughter.track_id for daughter in self.daughter_trajectories
            ]

        if tmp is not None:
            self.tmp = tmp
        else:
            self.tmp = {}

    def __repr__(self) -> str:
        return f"SingleCellTrajectory(track_id={self.track_id}, #timeframe set={len(self)})"

    def __len__(self):
        # return self.get_timeframe_span_length()
        return len(self.timeframe_to_single_cell)

    def __getitem__(self, timeframe: int) -> SingleCellStatic:
        if timeframe not in self.timeframe_set:
            raise KeyError(f"single cell at timeframe {timeframe} does not exist in the trajectory")
        return self.get_single_cell(timeframe)

    def __iter__(self):
        return iter(self.timeframe_to_single_cell.items())

    def _update_meta(self):
        self.meta[SingleCellTrajectory.META_MOTHER_IDS] = [mother.track_id for mother in self.mother_trajectories]
        self.meta[SingleCellTrajectory.META_DAUGHTER_IDS] = [
            daughter.track_id for daughter in self.daughter_trajectories
        ]

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

    def add_single_cell(self, timeframe, sc: SingleCellStatic):
        self.timeframe_to_single_cell[timeframe] = sc
        self.timeframe_set.add(timeframe)
        self.times = sorted(self.timeframe_set)

    add_sc = add_single_cell

    def get_img(self, timeframe):
        return self.timeframe_to_single_cell[timeframe].get_img()

    def get_mask(self, timeframe):
        return self.timeframe_to_single_cell[timeframe].get_mask()

    def get_timeframe_span(self):
        assert len(self.timeframe_set) > 0, "sct: timeframe set is empty."
        return (min(self.timeframe_set), max(self.timeframe_set))

    get_time_span = get_timeframe_span

    def get_timeframe_span_length(self):
        min_t, max_t = self.get_timeframe_span()
        return max_t - min_t + 1

    get_time_span_length = get_timeframe_span_length

    def get_single_cell(self, timeframe: int) -> SingleCellStatic:
        return self.timeframe_to_single_cell[timeframe]

    def get_all_scs(self) -> List[SingleCellStatic]:
        scs = self.timeframe_to_single_cell.values()
        sorted_scs = sorted(scs, key=lambda sc: sc.timeframe)
        return list(sorted_scs)

    get_sorted_scs = get_all_scs

    def num_scs(self) -> int:
        return len(self.timeframe_to_single_cell)

    def pop_single_cell(self, timeframe: int):
        self.timeframe_set.remove(timeframe)
        return self.timeframe_to_single_cell.pop(timeframe)

    def to_json_dict(self, dataset_json_dir=None):
        # Check if mother and daughter trajectories exist in metadata. If not, add them
        self._update_meta()
        # Update metadata with img and mask datasets json paths
        self.meta.update(
            {
                "img_dataset_json_path": str(self.img_dataset.get_default_json_path(out_dir=dataset_json_dir))
                if self.img_dataset is not None
                else None,
                "mask_dataset_json_path": str(self.mask_dataset.get_default_json_path(out_dir=dataset_json_dir))
                if self.mask_dataset is not None
                else None,
            }
        )

        res = {
            "track_id": int(self.track_id),
            "timeframe_to_single_cell": {
                int(float(timeframe)): sc.to_json_dict(dataset_json_dir=dataset_json_dir)
                for timeframe, sc in self.timeframe_to_single_cell.items()
            },
            # Store mother and daughter trajectories, and dataset json path in metadata
            "meta": self.meta,
        }

        if self.img_dataset is not None and res["meta"].get("img_dataset_json_path") is not None:
            img_dataset_dir = os.path.dirname(res["meta"].get("img_dataset_json_path"))
            self.img_dataset.write_json(out_dir=img_dataset_dir, overwrite=False)
        if self.mask_dataset is not None and res["meta"].get("mask_dataset_json_path") is not None:
            mask_dataset_dir = os.path.dirname(res["meta"].get("mask_dataset_json_path"))
            self.mask_dataset.write_json(out_dir=mask_dataset_dir, overwrite=False)

        return res

    # TODO: [smz] add log to input and output functions
    def write_json(self, path=None, dataset_json_dir=None):
        json_dict = self.to_json_dict(dataset_json_dir=dataset_json_dir)

        # Write img and mask datasets to JSON file
        if self.img_dataset is not None and json_dict["meta"].get("img_dataset_json_path") is not None:
            img_dataset_dir = os.path.dirname(json_dict["meta"].get("img_dataset_json_path"))
            self.img_dataset.write_json(out_dir=img_dataset_dir, overwrite=False)
        if self.mask_dataset is not None and json_dict["meta"].get("mask_dataset_json_path") is not None:
            mask_dataset_dir = os.path.dirname(json_dict["meta"].get("mask_dataset_json_path"))
            self.mask_dataset.write_json(out_dir=mask_dataset_dir, overwrite=False)

        if path is None:
            return json.dumps(json_dict, cls=LiveCellEncoder)
        else:
            with open(path, "w+") as f:
                json.dump(json_dict, f, cls=LiveCellEncoder)

    def load_from_json_dict(self, json_dict, img_dataset=None, share_img_dataset=True):
        self.track_id = json_dict["track_id"]
        if "meta" in json_dict:
            self.meta = json_dict["meta"]

        # Load img dataset from input
        if img_dataset:
            self.img_dataset = img_dataset

        shared_img_dataset = None
        if share_img_dataset:
            shared_img_dataset = self.img_dataset

        # Load img dataset and mask dataset from json
        # Backward compatibility: check dataset json file path from meta first.
        # If they're not found, look in `json_dict`.
        img_dataset_json_path = self.meta.get("img_dataset_json_path", json_dict.get("img_dataset_json_path"))
        if self.img_dataset is None and img_dataset_json_path is not None:
            if os.path.exists(img_dataset_json_path):
                self.img_dataset = LiveCellImageDataset.load_from_json_file(path=img_dataset_json_path)
            else:
                raise Warning(f"img_dataset_json_path {img_dataset_json_path} does not exist")

        mask_dataset_json_path = self.meta.get("mask_dataset_json_path", json_dict.get("mask_dataset_json_path"))
        if self.mask_dataset is None and mask_dataset_json_path is not None:
            if os.path.exists(mask_dataset_json_path):
                self.mask_dataset = LiveCellImageDataset.load_from_json_file(path=mask_dataset_json_path)
            else:
                raise Warning(f"mask_dataset_json_path {mask_dataset_json_path} does not exist")

        if self.img_dataset is None:
            main_warning("[SCT loading] img_dataset is None after attempting to load it")
            # raise ValueError("img_dataset is None after attempting to load it")

        self.img_total_timeframe = len(self.img_dataset) if self.img_dataset is not None else 0
        self.timeframe_to_single_cell = {}
        for timeframe, sc in json_dict["timeframe_to_single_cell"].items():
            self.timeframe_to_single_cell[int(timeframe)] = SingleCellStatic(
                timeframe=int(timeframe), img_dataset=shared_img_dataset, empty_cell=True
            ).load_from_json_dict(sc, img_dataset=shared_img_dataset)

        self.timeframe_set = set(self.timeframe_to_single_cell.keys())
        self.times = sorted(self.timeframe_set)
        return self

    def inflate_other_trajectories(self, sctc: "SingleCellTrajectoryCollection"):
        """inflate the other trajectories in this trajectory's mother and daughter trajectories"""
        self.mother_trajectories = {sctc.get_trajectory(id) for id in self.meta[SingleCellTrajectory.META_MOTHER_IDS]}
        self.daughter_trajectories = {
            sctc.get_trajectory(id) for id in self.meta[SingleCellTrajectory.META_DAUGHTER_IDS]
        }

    def inflate_other_trajectories_by_dict(self, track_id_to_trajectory: Dict[int, "SingleCellTrajectory"]):
        """inflate the other trajectories in this trajectory's mother and daughter trajectories"""
        self.mother_trajectories = {
            track_id_to_trajectory[id] for id in self.meta[SingleCellTrajectory.META_MOTHER_IDS]
        }
        self.daughter_trajectories = {
            track_id_to_trajectory[id] for id in self.meta[SingleCellTrajectory.META_DAUGHTER_IDS]
        }

    @staticmethod
    def load_from_json_file(path):
        with open(path, "r") as file:
            json_dict = json.load(file)
        return SingleCellTrajectory().load_from_json_dict(json_dict)

    def get_sc_feature_table(self):
        feature_table = None
        for timeframe, sc in self:
            assert timeframe == sc.timeframe, "timeframe mismatch"
            feature_series = sc.get_feature_pd_series()
            feature_series["track_id"] = self.track_id
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

    def is_empty(self):
        return len(self.timeframe_set) == 0

    def subsct(self, min_time, max_time, track_id=None, keep_track_id=False):
        """return a subtrajectory of this trajectory, with timeframes between [min_time, max_time]. Mother and daugher info will be copied if the min_time and max_time are the start and end of the new trajectory, respectively."""
        require_copy_mothers_info = False
        require_copy_daughters_info = False
        if self.is_empty():
            return SingleCellTrajectory(track_id=track_id)

        self_span = self.get_timeframe_span()

        # TODO: if time is float case, consider round-off errors
        if min_time == self_span[0]:
            require_copy_mothers_info = True
        if max_time == self_span[1]:
            require_copy_daughters_info = True
        if keep_track_id:
            track_id = self.track_id
        sub_sct = SingleCellTrajectory(img_dataset=self.img_dataset, mask_dataset=self.mask_dataset, track_id=track_id)
        for timeframe, sc in self:
            if timeframe >= min_time and timeframe <= max_time:
                sub_sct.add_single_cell(timeframe, sc)
        if require_copy_mothers_info:
            sub_sct.mother_trajectories = self.mother_trajectories.copy()
        if require_copy_daughters_info:
            sub_sct.daughter_trajectories = self.daughter_trajectories.copy()
        return sub_sct

    def split(self, split_time, tid_1=None, tid_2=None) -> Tuple["SingleCellTrajectory", "SingleCellTrajectory"]:
        """split this trajectory into two trajectories: [start, split_time), [split_time, end], at the given split time"""
        if split_time not in self.timeframe_set:
            raise ValueError("split time not in this trajectory")
        sct1 = self.subsct(min(self.timeframe_set), split_time - 1, track_id=tid_1)
        sct2 = self.subsct(split_time, max(self.timeframe_set), track_id=tid_2)
        return sct1, sct2

    def next_time(self, time: Union[int, float]) -> Union[int, float, None]:
        """return the next time in this trajectory after the given time. If the given time is the last time in this trajectory, return None

        Parameters
        ----------
        time : Union[int, float]
            _description_

        Returns
        -------
        Union[int, float, None]
            _description_
        """
        # TODO: we may save a linked list of timeframes, so that we can do this in O(1) time
        import bisect

        def get_next_largest_element(sorted_list, elem):
            index = bisect.bisect(sorted_list, elem)
            if index == len(sorted_list):
                return None  # No larger element found
            else:
                return sorted_list[index]

        return get_next_largest_element(self.times, time)

    @staticmethod
    def show_trajectory_on_grid(**kwargs):
        return show_sct_on_grid(**kwargs)

    def show_on_grid(self, **kwargs):
        return show_sct_on_grid(self, **kwargs)


def load_from_json_dict_parallel_wrapper(track_id, trajectory_dict):
    return track_id, SingleCellTrajectory().load_from_json_dict(trajectory_dict)


class SingleCellTrajectoryCollection:
    """
    Represents a collection of single-cell trajectories.

    Attributes:
    - track_id_to_trajectory: A dictionary mapping track IDs to SingleCellTrajectory objects.
    - _iter_index: An index used for iterating over the track ID to trajectory mapping.
    """

    def __init__(self) -> None:
        self.track_id_to_trajectory: Dict[float, SingleCellTrajectory] = dict()
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

    def __iter__(self) -> Iterator[Tuple[float, SingleCellTrajectory]]:
        return iter(self.track_id_to_trajectory.items())

    def add_trajectory(self, trajectory: SingleCellTrajectory):
        if trajectory.track_id is None:
            trajectory.track_id = self._next_track_id()
        if trajectory.track_id in self.track_id_to_trajectory:
            raise ValueError("trajectory with track_id {} already exists".format(trajectory.track_id))
        self[trajectory.track_id] = trajectory

    def filter_trajectories(self, filter_func: Callable[[SingleCellTrajectory], bool], inplace=False):
        if inplace:
            self.track_id_to_trajectory = {
                track_id: trajectory for track_id, trajectory in self if filter_func(trajectory)
            }
            return self
        else:
            new_sctc = SingleCellTrajectoryCollection()
            for track_id, trajectory in self:
                if filter_func(trajectory):
                    new_sctc.add_trajectory(trajectory)
            return new_sctc

    def filter_trajectories_by_length(self, min_length=None, max_length=None, inplace=False):
        def filter_func(trajectory):
            length = len(trajectory)
            if min_length is not None and length < min_length:
                return False
            if max_length is not None and length > max_length:
                return False
            return True

        return self.filter_trajectories(filter_func, inplace=inplace)

    def get_trajectory(self, track_id) -> SingleCellTrajectory:
        return self.track_id_to_trajectory[track_id]

    def get_all_scs(self) -> List[SingleCellStatic]:
        all_scts = self.get_all_trajectories()
        all_scs = []
        for sct in all_scts:
            all_scs.extend(sct.get_all_scs())
        return all_scs

    def get_all_trajectories(self) -> List[SingleCellTrajectory]:
        return list(self.track_id_to_trajectory.values())

    # TODO refactor get_all_tids and get_all_track_ids
    def get_all_tids(self) -> List[float]:
        return list(self.track_id_to_trajectory.keys())

    def get_track_ids(self):
        return sorted(list(self.track_id_to_trajectory.keys()))

    def get_max_tid(self):
        return max(self.get_track_ids())

    get_all_track_ids = get_all_tids

    def pop_trajectory(self, track_id):
        return self.track_id_to_trajectory.pop(track_id)

    def to_json_dict(self, dataset_json_dir=None):
        return {
            "track_id_to_trajectory": {
                int(track_id): trajectory.to_json_dict(dataset_json_dir=dataset_json_dir)
                for track_id, trajectory in self.track_id_to_trajectory.items()
            }
        }

    def _post_load_trajectories(self):
        for _, trajectory in self:
            trajectory.inflate_other_trajectories(self)

    def load_from_json_dict(self, json_dict):
        self.track_id_to_trajectory = {}
        for track_id, trajectory_dict in json_dict["track_id_to_trajectory"].items():
            # TODO: track_id = int(float(track_id)) remove extra float conversion in the future
            self.track_id_to_trajectory[int(float(track_id))] = SingleCellTrajectory().load_from_json_dict(
                trajectory_dict
            )
        self._post_load_trajectories()
        return self

    def load_from_json_dict_parallel(self, json_dict):
        self.track_id_to_trajectory = {}
        inputs = []
        for track_id, trajectory_dict in json_dict["track_id_to_trajectory"].items():
            # TODO: track_id = int(float(track_id)) remove extra float conversion in the future
            inputs.append((int(float(track_id)), trajectory_dict))
        outputs = parallelize(load_from_json_dict_parallel_wrapper, inputs)
        for track_id, traj in outputs:
            self.track_id_to_trajectory[int(float(track_id))] = traj
        self._post_load_trajectories()
        return self

    def write_json(self, path, dataset_json_dir=None, filter_empty=True):
        if filter_empty:
            self.remove_empty_sct(inplace=True)

        if dataset_json_dir is None:
            dataset_json_dir = Path(os.path.dirname(path)) / "datasets"
            # Create the directory if it doesn't exist
            dataset_json_dir.mkdir(parents=True, exist_ok=True)

        with open(path, "w+") as f:
            json.dump(self.to_json_dict(dataset_json_dir=dataset_json_dir), f, cls=LiveCellEncoder)

    @staticmethod
    def load_from_json_file(path, parallel=False):
        with open(path, "r") as f:
            json_dict = json.load(f)
        main_info(f"json loaded from {path}")

        main_info("Creating SingleCellTrajectoryCollection from json_dict...")
        if parallel:
            return SingleCellTrajectoryCollection().load_from_json_dict_parallel(json_dict)
        else:
            return SingleCellTrajectoryCollection().load_from_json_dict(json_dict)

    def histogram_traj_length(self, ax=None, **kwargs):
        import seaborn as sns

        id_to_sc_trajs = self.track_id_to_trajectory
        all_traj_lengths = np.array([_traj.get_timeframe_span_length() for _traj in id_to_sc_trajs.values()])
        if ax is None:
            ax = sns.countplot(x=all_traj_lengths, **kwargs)
        else:
            ax = sns.countplot(x=all_traj_lengths, ax=ax, **kwargs)
        for container in ax.containers:
            ax.bar_label(container)
        ax.set(xlabel="Trajectory Length")
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

    def get_time_span(self) -> [int, int]:
        res_time_span = (None, None)
        for track_id, trajectory in self:
            _tmp_time_span = trajectory.get_time_span()
            if res_time_span[0] is None:
                res_time_span = _tmp_time_span
            else:
                res_time_span = (min(res_time_span[0], _tmp_time_span[0]), max(res_time_span[1], _tmp_time_span[1]))
        if res_time_span[0] is None:
            return (-np.inf, -np.inf)
        return res_time_span

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
        if len(self.get_track_ids()) == 0:
            return 0
        return max(self.get_track_ids()) + 1

    def remove_empty_sct(self, inplace=True):
        remove_tids = []
        remove_scs = []
        for tid, sct in self:
            _tmp_scs = sct.get_all_scs()
            to_be_removed = True
            for sc in _tmp_scs:
                if len(sc.contour) > 0:
                    to_be_removed = False
                    break
            if to_be_removed:
                remove_tids.append(tid)
                remove_scs.extend(_tmp_scs)
        if inplace:
            for tid in remove_tids:
                self.pop_trajectory(tid)
            return self
        else:
            new_sctc = SingleCellTrajectoryCollection()
            for tid, sct in self:
                if tid not in remove_tids:
                    new_sctc.add_trajectory(sct)
            return new_sctc


def create_sctc_from_scs(scs: List[SingleCellStatic]) -> SingleCellTrajectoryCollection:
    temp_sc_trajs = SingleCellTrajectoryCollection()
    for idx, sc in enumerate(scs):
        sct = SingleCellTrajectory(track_id=idx, timeframe_to_single_cell={sc.timeframe: sc})
        temp_sc_trajs.add_trajectory(sct)
    return temp_sc_trajs


def filter_sctc_by_time_span(sctc: SingleCellTrajectoryCollection = None, time_span=(0, np.inf), keep_track_id=True):
    new_sctc = SingleCellTrajectoryCollection()
    track_id_counter = 0
    for _, sct in sctc:
        if keep_track_id:
            track_id = sct.track_id
        else:
            track_id = track_id_counter
        subsct = sct.subsct(time_span[0], time_span[1], track_id=track_id)
        track_id_counter += 1
        if subsct.num_scs() > 0:
            new_sctc.add_trajectory(subsct)
    return new_sctc


def create_sc_table(
    scs: List[SingleCellStatic], normalize_features=True, add_time=False, add_sc_id=False, meta_keys=[]
):
    import pandas as pd
    import numpy as np

    df = pd.DataFrame([sc.get_feature_pd_series() for sc in scs])
    if normalize_features:
        for col in df.columns:
            if col == "sc_id":
                continue
            df[col] = df[col] - df[col].mean()
            col_std = df[col].std()
            if col_std != 0 and not np.isnan(col_std):
                df[col] /= col_std
    # remove column t from df
    if not add_time:
        df.drop("t", axis=1, inplace=True)
    if not add_sc_id:
        df.drop("sc_id", axis=1, inplace=True)

    # add meta information
    for key in meta_keys:
        if key in df.columns:
            main_warning(f"meta key {key} conflicts with feature key {key}, skipping")
            continue
        df[key] = [sc.meta[key] for sc in scs]
    return df


def show_sct_on_grid(
    trajectory: "SingleCellTrajectory",
    nr=4,
    nc=4,
    start=0,
    interval=1,
    padding=20,
    dims: Tuple[int, int] = None,
    dims_offset: Tuple[int, int] = (0, 0),
    pad_dims=True,
    ax_width=4,
    ax_height=4,
    ax_title_fontsize=8,
    cmap="viridis",
    ax_contour_polygon_kwargs=dict(fill=None, edgecolor="r"),
    dpi=300,
    show_mask=False,
    fig=None,
    axes=None,
    crop_from_center=True,
    show_contour=True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Display a grid of single cell images with contours overlaid.

    Parameters:
    -----------
    trajectory : SingleCellTrajectory
        The trajectory object containing the single cell images.
    nr : int, optional
        Number of rows in the grid, by default 4.
    nc : int, optional
        Number of columns in the grid, by default 4.
    start : int, optional
        The starting timeframe, and will be replaced by the first timeframe of the trajectory if it is smaller than the first timeframe. by default 0.
    interval : int, optional
        The interval between timeframes, by default 5.
    padding : int, optional
        The padding around the single cell image, by default 20.
    dims : Tuple[int, int], optional
        The dimensions to crop the single cell image to, by default None.
    dims_offset : Tuple[int, int], optional
        The offset to apply to the cropped image, by default (0, 0).
    pad_dims : bool, optional
        Whether to pad the cropped image to match the specified dimensions, by default True.
    ax_width : int, optional
        The width of each subplot, by default 4.
    ax_height : int, optional
        The height of each subplot, by default 4.
    ax_title_fontsize : int, optional
        The fontsize of the subplot titles, by default 8.
    cmap : str, optional
        The colormap to use for displaying the single cell images, by default "viridis".
    ax_contour_polygon_kwargs : dict, optional
        The keyword arguments to pass to the Polygon object for drawing the contour, by default dict(fill=None, edgecolor='r').

    Returns:
    --------
    matplotlib.axes.Axes
        The axes object containing the grid of subplots.
    """
    if axes is None:
        fig, axes = plt.subplots(nr, nc, figsize=(nc * ax_width, nr * ax_height), dpi=dpi)
        if nr == 1:
            axes = np.array([axes])
    else:
        assert np.array(axes).shape == (nr, nc), "axes shape mismatch"

    span_range = trajectory.get_timeframe_span()
    traj_start, traj_end = span_range
    if start < traj_start:
        start = span_range[0]
        print(
            "start timeframe larger than the first timeframe of the trajectory, replace start_timeframe with the first timeframe={}".format(
                int(start)
            )
        )

    if isinstance(ax_contour_polygon_kwargs, dict):
        ax_contour_polygon_kwargs_list = [ax_contour_polygon_kwargs] * nr * nc
    else:
        ax_contour_polygon_kwargs_list = ax_contour_polygon_kwargs

    for r in range(nr):
        for c in range(nc):
            ax = axes[r, c]
            ax.axis("off")
            timeframe = start + interval * (r * nc + c)
            if timeframe > traj_end:
                break
            if timeframe not in trajectory.timeframe_set:
                continue
            sc = trajectory.get_single_cell(timeframe)
            if show_mask:
                sc_img = sc.get_mask_crop(padding=padding)
            else:
                sc_img = sc.get_img_crop(padding=padding)
            contour_coords = sc.get_contour_coords_on_crop(padding=padding)

            if dims is not None:
                center_coord = [sc_img.shape[i] // 2 for i in range(2)]
                if crop_from_center:
                    xs, ys, xe, ye = (
                        center_coord[0] - dims[0] // 2,
                        center_coord[1] - dims[1] // 2,
                        center_coord[0] + dims[0] // 2,
                        center_coord[1] + dims[1] // 2,
                    )

                    # Center padding
                    _center_padding = padding // 2
                    xs, ys, xe, ye = (
                        xs - _center_padding,
                        ys - _center_padding,
                        xe + _center_padding,
                        ye + _center_padding,
                    )
                    # Fit to boundary of img shape [0, boundary]
                    xs, ys, xe, ye = max(0, xs), max(0, ys), min(sc_img.shape[0], xe), min(sc_img.shape[1], ye)
                    sc_img = sc_img[xs:xe, ys:ye]
                    contour_coords[:, 0] = contour_coords[:, 0] - xs
                    contour_coords[:, 1] = contour_coords[:, 1] - ys
                    if pad_dims:
                        _pad_pixels = [dims[0] - (xe - xs), dims[1] - (ye - ys)]
                        # Ensure non-neg and //2 for center
                        _pad_pixels = [max(0, val) // 2 for val in _pad_pixels]
                        sc_img = np.pad(sc_img, _pad_pixels, mode="constant", constant_values=0)
                        contour_coords[:, 0] += _pad_pixels[0]
                        contour_coords[:, 1] += _pad_pixels[1]
                else:
                    sc_img = sc_img[
                        dims_offset[0] : dims_offset[0] + dims[0], dims_offset[1] : dims_offset[1] + dims[1]
                    ]
                    contour_coords[:, 0] -= dims_offset[0]
                    contour_coords[:, 1] -= dims_offset[1]
                    if pad_dims:
                        _pad_pixels = [max(0, dims[i] - sc_img.shape[i]) for i in range(len(dims))]
                        sc_img = np.pad(sc_img, _pad_pixels, mode="constant", constant_values=0)
                        contour_coords[:, 0] += _pad_pixels[0]
                        contour_coords[:, 1] += _pad_pixels[1]

            ax.imshow(sc_img, cmap=cmap)

            if show_contour:
                # draw a polygon based on contour coordinates
                from matplotlib.patches import Polygon

                polygon = Polygon(
                    np.array([contour_coords[:, 1], contour_coords[:, 0]]).transpose(),
                    **ax_contour_polygon_kwargs_list[r * nc + c],
                )
                ax.add_patch(polygon)
                ax.set_title(f"time: {timeframe}", fontsize=ax_title_fontsize)

    if fig is not None:
        main_info(f"tighting figure layout...")
        fig.tight_layout(pad=0.5, h_pad=0.4, w_pad=0.4)
    return fig, axes


def combine_scs_label_masks(scs: List[SingleCellStatic], scs_labels: list = None, original_meta_label_key=None):
    """Generate a label mask from a list of single cell objects."""
    label_mask = np.zeros(scs[0].get_mask().shape, dtype=np.int32)
    if scs_labels is None and original_meta_label_key is None:
        scs_labels = list(range(1, len(scs) + 1))
    elif scs_labels is None and original_meta_label_key is not None:
        scs_labels = [sc.meta[original_meta_label_key] for sc in scs]

    for sc_idx in range(len(scs)):
        sc = scs[sc_idx]
        label = scs_labels[sc_idx]
        label_mask[sc.get_mask()] = label
    return label_mask


def get_time2scs(scs: List[SingleCellStatic]):
    time2scs = {}
    for sc in scs:
        if sc.timeframe not in time2scs:
            time2scs[sc.timeframe] = []
        time2scs[sc.timeframe].append(sc)
    return time2scs


def sample_samples_from_sctc(
    sctc: SingleCellTrajectoryCollection,
    objective_sample_num=10000,
    exclude_scs_ids=set(),
    seed=0,
    length_range=(6, 10),
    max_trial_counter=1000,
    check_nonoverlap=True,
):
    def _check_in_visited_range(visited_range, track_id, start_time, end_time):
        """check if the given time range of a track is in the visited range"""
        if track_id not in visited_range:
            return False
        for _start, _end in visited_range[track_id]:
            # Check if there is any overlap between [_start, _end] and [start_time, end_time]
            if _start <= start_time:
                ls, le = _start, _end
                rs, re = start_time, end_time
            else:
                ls, le = start_time, end_time
                rs, re = _start, _end
            if rs <= le:
                return True
        return False

    # set numpy seed
    np.random.seed(seed)

    normal_frame_len_range = length_range
    counter = 0
    normal_samples = []
    normal_samples_extra_info = []
    skipped_sample_num = 0
    visited_range = {}
    while counter < objective_sample_num and max_trial_counter > 0:
        # randomly select a sct from sctc
        # generate a list of scs
        track_id = np.random.choice(list(sctc.track_id_to_trajectory.keys()))
        sct = sctc.get_trajectory(track_id)
        # randomly select a length
        frame_len = np.random.randint(*normal_frame_len_range)
        # generate a sample
        times = list(sct.timeframe_to_single_cell.keys())
        times = sorted(times)
        if len(times) <= frame_len:
            max_trial_counter -= 1
            continue
        start_idx = np.random.randint(0, len(times) - frame_len)
        start_time = times[start_idx]
        end_time = times[start_idx + frame_len - 1]

        if check_nonoverlap and _check_in_visited_range(visited_range, track_id, start_time, end_time):
            max_trial_counter -= 1
            continue
        if track_id not in visited_range:
            visited_range[track_id] = []
        visited_range[track_id].append((start_time, end_time))

        sub_sct = sct.subsct(start_time, end_time)

        is_some_sc_in_exclude_scs = False
        for time, sc in sub_sct.timeframe_to_single_cell.items():
            # print("sc.id:", sc.id, type(sc.id))
            if str(sc.id) in exclude_scs_ids:
                is_some_sc_in_exclude_scs = True
                break
        if is_some_sc_in_exclude_scs:
            # print("some sc in the exclude scs list")
            skipped_sample_num += 1
            continue

        new_sample = []
        for time, sc in sub_sct.timeframe_to_single_cell.items():
            new_sample.append(sc)
        normal_samples.append(new_sample)
        normal_samples_extra_info.append(
            {
                "src_dir": sub_sct.get_all_scs()[0].meta["src_dir"],
                "track_id": track_id,
                "start_time": start_time,
                "end_time": end_time,
            }
        )
        counter += 1

    if check_nonoverlap:
        print("visited range:", visited_range)
        # Check if the generated samples are non-overlapping
        main_info("Checking if the generated samples are non-overlapping...")
        for i in range(len(normal_samples)):
            for j in range(i + 1, len(normal_samples)):
                # Retrieve track id, start time, and end time for each sample
                track_id_i = normal_samples_extra_info[i]["track_id"]
                start_time_i = normal_samples_extra_info[i]["start_time"]
                end_time_i = normal_samples_extra_info[i]["end_time"]
                track_id_j = normal_samples_extra_info[j]["track_id"]
                start_time_j = normal_samples_extra_info[j]["start_time"]
                end_time_j = normal_samples_extra_info[j]["end_time"]

                # Check if the two samples overlap
                if track_id_i == track_id_j:
                    if start_time_i <= end_time_j and start_time_j <= end_time_i:
                        print(f"Overlap found between samples {i} and {j}")
                        print(f"Sample {i}: track_id={track_id_i}, start_time={start_time_i}, end_time={end_time_i}")
                        print(f"Sample {j}: track_id={track_id_j}, start_time={start_time_j}, end_time={end_time_j}")
                        assert False, "Overlap found between samples while checking non-overlapping samples"
        main_info("Success: No overlap found between the generated samples")
    print("# of skipped samples based on the excluded scs list:", skipped_sample_num)
    print("# of generated samples:", len(normal_samples))
    print("# of generated samples extra info:", len(normal_samples_extra_info))
    return normal_samples, normal_samples_extra_info


def filter_boundary_cells(scs: List[SingleCellStatic], dist_to_boundary=30, bbox_bounds=None):
    not_boundary_scs = []
    if bbox_bounds is None:
        dim = scs[0].get_img().shape[:2]
        bbox_bounds = [0, 0, dim[0], dim[1]]
    for sc in scs:
        bbox = sc.get_bbox()
        if (
            bbox[0] > bbox_bounds[0] + dist_to_boundary
            and bbox[1] > bbox_bounds[1] + dist_to_boundary
            and bbox[2] < bbox_bounds[2] - dist_to_boundary
            and bbox[3] < bbox_bounds[3] - dist_to_boundary
        ):
            not_boundary_scs.append(sc)
    return not_boundary_scs


def create_label_mask_from_scs(scs: List[SingleCellStatic], labels=None, dtype=np.int32, bbox=None, padding=None):
    label_mask = np.zeros(scs[0].get_mask_crop(bbox=bbox, padding=padding).shape, dtype=dtype)
    if len(scs) == 0:
        return label_mask
    if labels is None:
        labels = list(range(1, len(scs) + 1))  # Bg label is 0
    if bbox is None:
        shape = scs[0].get_img_shape()
        bbox = [0, 0, shape[0], shape[1]]

    for idx, sc in enumerate(scs):
        label_mask[sc.get_mask_crop(bbox=bbox, padding=padding)] = labels[idx]
    return label_mask


def largest_bbox(scs):
    if len(scs) == 0:
        return [0, 0, 0, 0]
    largest_bbox = [np.inf, np.inf, -np.inf, -np.inf]
    for sc in scs:
        bbox = sc.bbox
        if bbox[0] < largest_bbox[0]:
            largest_bbox[0] = bbox[0]
        if bbox[1] < largest_bbox[1]:
            largest_bbox[1] = bbox[1]
        if bbox[2] > largest_bbox[2]:
            largest_bbox[2] = bbox[2]
        if bbox[3] > largest_bbox[3]:
            largest_bbox[3] = bbox[3]
    return largest_bbox
