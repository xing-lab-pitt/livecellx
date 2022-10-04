import os

import matplotlib.pyplot as plt
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer


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
