import os

import matplotlib.pyplot as plt
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from skimage import measure
from livecell_tracker.core import SingleCellTrajectory, SingleCellStatic
from pathlib import Path
from livecell_tracker.segment.datasets import LiveCellImageDataset
from PIL import Image, ImageSequence
from tqdm import tqdm
import json


def gen_cfg(model_path=None, output_dir="./detectron_training_output"):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("deepfashion_val",)
    cfg.DATASETS.TEST = ()
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )  # Let training initialize from model zoo

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 3000
    cfg.SOLVER.STEPS = (1000, 1500)
    cfg.SOLVER.GAMMA = 0.05
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13

    cfg.TEST.EVAL_PERIOD = 500
    cfg.OUTPUT_DIR = output_dir
    if model_path is not None:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55  # set the testing threshold for this model
    cfg.DATASETS.TEST = ("deepfashion_val",)
    return cfg


def detectron_visualize_img(img, cfg, detectron_outputs):
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(detectron_outputs["instances"].to("cpu"))
    figure, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=80)
    axis_img = ax.imshow(v.get_image()[:, :, ::-1])
    return axis_img.get_array()


def convert_detectron_instances_to_label_masks(instance_pred_masks):
    res_mask = np.zeros(instance_pred_masks.shape[1:])
    for idx in range(instance_pred_masks.shape[0]):
        res_mask[instance_pred_masks[idx, :, :]] = idx + 1
    return res_mask


def convert_detectron_instances_to_binary_masks(instance_pred_masks):
    label_mask = convert_detectron_instances_to_label_masks(instance_pred_masks)
    label_mask[label_mask > 0] = 1
    return label_mask


def detectron_segment_imgs(imgs: LiveCellImageDataset, out_dir: Path):
    segmentation_results = {}
    print(type(imgs))
    for idx in tqdm(range(len(imgs))):
        img_path = imgs.get_img_path(idx)
        img = imgs[idx]
        original_img_filename = os.path.basename(img_path).split(".")[0]
        output_filename = original_img_filename + ".png"  # change extension to PNG

        # save binary mask
        mask, predictor_results = segment_raw_img_by_detectron_wrapper(img, return_detectron_results=True)
        # convert mask to 8-bit binary mask
        assert mask.max() < 2 ** 8, "more than 256 instances predicted?"
        mask = mask.astype(np.uint8)
        binary_mask_img = Image.fromarray(mask)
        binary_mask_img.save(out_dir / output_filename)

        # save overlayed image
        overlay_output_filename = "overlay_" + original_img_filename + ".png"  # change extension to PNG
        # overlayed_img = overlay(img, mask, mask_channel_rgb_val=100, img_channel_rgb_val_factor=2)
        overlayed_arr = detectron_visualize_img(img[:, :, np.newaxis], DETECTRON_CFG, predictor_results)
        overlayed_img = Image.fromarray(overlayed_arr)
        overlayed_img.save(out_dir / overlay_output_filename)
        del overlayed_img, overlayed_arr, mask, binary_mask_img

        def _save_instance_masks():
            # save predicted instance masks
            pred_binary_masks = predictor_results["instances"].to("cpu").pred_masks.numpy()
            for idx in range(pred_binary_masks.shape[0]):
                pred_binary_mask = pred_binary_masks[idx, :, :]
                pred_binary_mask_img = Image.fromarray(pred_binary_mask)
                pred_binary_mask_img.save(out_dir / f"{original_img_filename}_instance_{idx}.png")
                del pred_binary_mask, pred_binary_mask_img
            del predictor_results, pred_binary_masks

        # _save_instance_masks()

        # generate contours and save to json
        contours = []
        for instance_mask in predictor_results["instances"].to("cpu").pred_masks.numpy():
            tmp_contours = measure.find_contours(
                instance_mask, level=0.5, fully_connected="low", positive_orientation="low"
            )
            if len(tmp_contours) != 1:
                print("[WARN] more than 1 contour found in instance mask")
            # convert to list for saving into json
            contours.extend([[list(coords) for coords in coord_arr] for coord_arr in tmp_contours])
        assert original_img_filename not in segmentation_results, "duplicate image filename?"
        segmentation_results[img_path] = {}
        segmentation_results[img_path]["contours"] = contours
    return segmentation_results
