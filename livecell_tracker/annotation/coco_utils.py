from matplotlib import pyplot as plt
import numpy as np
import os
from pathlib import Path
from pycocotools.coco import COCO
from PIL import Image


def convert_coco_category_to_mask(coco_annotation: COCO, cat_id: int, output_dir: str):
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
        for ann in anns:
            if not (ann["category_id"] == cat_id):
                continue
            tmp_mask = coco_annotation.annToMask(ann)
            if mask is None:
                mask = tmp_mask
            mask = np.logical_or(mask, tmp_mask)
        # print(type(mask))
        # plt.imshow(mask)
        # plt.show()
        img = Image.fromarray(mask)
        img.save(Path(output_dir) / f"{img_id}.png")
