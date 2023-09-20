import os
from pathlib import Path
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pycocotools.coco import COCO

from livecellx.core.datasets import LiveCellImageDataset
from livecellx.core.single_cell import SingleCellStatic


def convert_coco_category_to_mask(coco_annotation: COCO, cat_id: int, output_dir: str, label_mask: bool = True):
    """Converts a coco category to a binary mask. For each image, collect all of its annotations and aggregate them into a single mask.

    Parameters
    ----------
    coco_annotation :
        COCO annotation object
    cat_id :
        category IDs specified in COCO json files
    output_dir :
        Output directory to save the masks
    """
    coco_img_ids = list(coco_annotation.imgs.keys())
    id2anns = coco_annotation.imgToAnns
    for img_id, anns in id2anns.items():
        mask = None
        cur_label = 1
        for ann in anns:
            if not (ann["category_id"] == cat_id):
                continue
            tmp_mask = coco_annotation.annToMask(ann)
            if mask is None:
                mask = tmp_mask
            if label_mask:
                tmp_mask = tmp_mask * cur_label
                cur_label += 1
                mask[tmp_mask > 0] = tmp_mask[tmp_mask > 0]
            else:
                mask = np.logical_or(mask, tmp_mask)

        # print(type(mask))
        # plt.imshow(mask)
        # plt.show()
        img = Image.fromarray(mask)
        img.save(Path(output_dir) / f"{img_id}.png")


def coco_to_sc(coco_data: COCO, extract_bbox=False) -> List[SingleCellStatic]:
    """Converts COCO annotation to SingleCellStatic objects. img_id is stored in the meta data and used as the timeframe of each sc.

    Parameters
    ----------
    coco_data : COCO
        _description_
    extract_bbox : bool, optional
        _description_, by default False

    Returns
    -------
    List[SingleCellStatic]
        _description_
    """
    img_metas = coco_data.imgs
    sc_list = []
    # constrcut dataset
    img_id_to_img_path = {}
    for img_id, img_meta in coco_data.imgs.items():
        img_path = img_meta["file_name"]
        img_id_to_img_path[img_id] = img_path

    # TODO: add image meta to LiveCellImageDataset
    dataset = LiveCellImageDataset(time2url=img_id_to_img_path, max_cache_size=0)
    for ann_key in coco_data.anns:
        ann = coco_data.anns[ann_key]

        # TODO: use the first contour instead of raising an error here?
        # TODO: or create multiple contours for a single annotation?
        assert (
            len(ann["segmentation"]) == 1
        ), "more than 1 contour contained in the segmentation list of some annotation"
        segmentation_flattened = ann["segmentation"][0]
        tmp_contour = np.array(segmentation_flattened).reshape(-1, 2)
        contour = tmp_contour.copy()
        contour[:, 0], contour[:, 1] = tmp_contour[:, 1], tmp_contour[:, 0]  # change to row-column format
        img_id = ann["image_id"]
        img_path = img_metas[img_id]["file_name"]
        meta = dict(ann)
        meta.pop("segmentation")
        meta["img_id"] = img_id
        meta["path"] = img_path
        if extract_bbox and "bbox" in ann:
            # https://github.com/cocodataset/cocoapi/issues/102
            # [x,y,width,height]
            # x, y: the upper-left coordinates of the bounding box
            # width, height: the dimensions of your bounding box
            bbox = list(ann["bbox"])
            x, y, width, height = bbox
            bbox = [y, x, y + height, x + width]  # change to row-column format
            if contour[:, 0].max() >= bbox[2]:
                bbox[2] = contour[:, 0].max() + 1
            if contour[:, 1].max() >= bbox[3]:
                bbox[3] = contour[:, 1].max() + 1

            # fix the bounding box
            # TODO: add warnings for two branches below
            assert contour[:, 0].min() >= 0, "negative row contour index"
            assert contour[:, 1].min() >= 0, "negative column contour index"
            assert contour[:, 0].max() < bbox[2], "row index exceeds the bounding box"
            assert contour[:, 1].max() < bbox[3], "column index exceeds the bounding box"
        else:
            bbox = None
        sc = SingleCellStatic(timeframe=img_id, bbox=bbox, contour=contour, meta=meta, img_dataset=dataset)
        if not extract_bbox:
            sc.update_bbox()  # update according to contours
        sc_list.append(sc)
    return sc_list
