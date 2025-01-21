import os
import argparse
from typing import Any, Dict, List, Union

import cv2
import numpy as np

import torch
from torch.nn import functional as F
from torchvision import transforms

from livecellx.core.single_cell import SingleCellStatic
from livecellx.model_zoo.segmentation.eval_csn import compute_watershed
from livecellx.model_zoo.segmentation.sc_correction_dataset import CorrectSegNetDataset
from livecellx.segment.ou_simulator import find_contours_opencv, find_label_mask_contours


def correct_sc_mask(_sc, model, padding, input_transforms=None, gpu=True, h_threshold=1):
    raw_data = CorrectSegNetDataset._prepare_sc_inference_data(
        _sc, padding_pixels=padding, bbox=None, normalize_crop=True
    )
    original_shape = raw_data["augmented_raw_img"].shape[-2:]
    sample = CorrectSegNetDataset.prepare_and_augment_data(
        raw_data,
        input_type=model.input_type,
        bg_val=0,
        normalize_uint8=model.normalize_uint8,
        exclude_raw_input_bg=model.exclude_raw_input_bg,
        force_no_edt_aug=False,
        apply_gt_seg_edt=model.apply_gt_seg_edt,
        transform=input_transforms,
    )
    if gpu:
        outputs = model(sample["input"].unsqueeze(0).cuda())
    else:
        outputs = model(sample["input"].unsqueeze(0))
    label_out = outputs[1].cpu().detach().numpy().squeeze()
    label_str = CorrectSegNetDataset.label_onehot_to_str(label_out)
    back_transforms = transforms.Compose(
        [
            transforms.Resize(size=(original_shape[0], original_shape[1])),
        ]
    )
    out_mask_transformed = back_transforms(outputs[0]).cpu().detach().numpy().squeeze()
    watershed_mask = compute_watershed(out_mask_transformed[0], h_threshold=h_threshold)
    return out_mask_transformed, watershed_mask, label_str


def contours_to_scs(contours, ref_sc: SingleCellStatic, padding=None, min_area=None):
    new_scs = []
    sc_bbox = ref_sc.get_bbox(padding=padding)
    for contour in contours:
        if min_area is not None and cv2.contourArea(contour) < min_area:
            continue
        _contour = np.array([(sc_bbox[0] + x, sc_bbox[1] + y) for x, y in contour])
        new_sc = SingleCellStatic(
            ref_sc.timeframe,
            contour=_contour,
            img_dataset=ref_sc.img_dataset,
        )
        new_scs.append(new_sc)
    return new_scs


def correct_sc(
    _sc, model, padding, input_transforms=None, gpu=True, min_area=100, return_outputs=False, h_threshold=1
) -> Union[List[SingleCellStatic], Dict[str, Any]]:
    out_mask, watershed_mask, label_str = correct_sc_mask(
        _sc, model=model, padding=padding, input_transforms=input_transforms, gpu=gpu, h_threshold=h_threshold
    )
    contours = find_label_mask_contours(watershed_mask)
    _scs = contours_to_scs(contours, ref_sc=_sc, padding=padding, min_area=min_area)
    for sc_ in _scs:
        sc_.meta["csn_gen"] = True
        sc_.meta["orig_sc_id"] = str(_sc.id)
    if not return_outputs:
        return _scs
    else:
        return {
            "scs": _scs,
            "out_mask": out_mask,
            "watershed_mask": watershed_mask,
            "label_str": label_str,
        }
