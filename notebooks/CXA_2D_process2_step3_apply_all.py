import livecellx
from livecellx.model_zoo.segmentation.custom_transforms import CustomTransformEdtV9
from livecellx.preprocess.utils import overlay


import numpy as np
import json
from livecellx.core import (
    SingleCellTrajectory,
    SingleCellStatic,
    SingleCellTrajectoryCollection,
)
from livecellx.core.single_cell import get_time2scs
from livecellx.core.datasets import LiveCellImageDataset
from livecellx.preprocess.utils import (
    overlay,
    enhance_contrast,
    normalize_img_to_uint8,
)
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
from typing import List


import glob
from PIL import Image, ImageSequence
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import json


import torch
import torch
import torch.utils.data
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar

from livecellx.model_zoo.segmentation.sc_correction import CorrectSegNet
from livecellx.model_zoo.segmentation.sc_correction_aux import CorrectSegNetAux
from livecellx.core.utils import label_mask_to_edt_mask
from livecellx.core.single_cell import combine_scs_label_masks
import livecellx.model_zoo.segmentation.csn_configs as csn_configs
import numpy as np
import warnings
import tqdm
from livecellx.core.io_sc import prep_scs_from_mask_dataset
from livecellx.core.sc_filters import filter_boundary_cells, filter_scs_by_size
import random

import tqdm
from datetime import datetime
import torch
from livecellx.core.utils import label_mask_to_edt_mask
from livecellx.segment.ou_viz import viz_ou_outputs
from livecellx.segment.ou_utils import create_ou_input_from_sc
from livecellx.preprocess.utils import normalize_edt
from livecellx.core.io_utils import save_png
from livecellx.model_zoo.segmentation.sc_correction_dataset import CorrectSegNetDataset
from livecellx.model_zoo.segmentation.csn_sc_utils import correct_sc, correct_sc_mask


from livecellx.segment.ou_utils import dilate_or_erode_label_mask


from livecellx.model_zoo.segmentation.sc_correction_dataset import CorrectSegNetDataset

torch.manual_seed(237)
# PADDING_PIXELS = 50
# PADDING_PIXELS = 10
PADDING_PIXELS = 20
OUT_THRESHOLD = 1


model_ckpt = "/home/ken67/livecellx/notebooks/lightning_logs/version_v18_02-inEDTv1-augEdtV9-scaleV2-lr-0.0001-aux-seed-404/checkpoints/last.ckpt"

# model_ckpt = (
#     "lightning_logs/version_v17_02-inEDTv1-augEdtV8-scaleV2-lr=0.00001-aux/checkpoints/last.ckpt"
# )
model = CorrectSegNetAux.load_from_checkpoint(model_ckpt)
# input_transforms = csn_configs.gen_train_transform_edt_v8(degrees=0, shear=0, flip_p=0)

model.cuda()
model.eval()


input_transforms = CustomTransformEdtV9(degrees=0, shear=0, flip_p=0, use_gaussian_blur=True, gaussian_blur_sigma=15)


lcx_out_dir = Path("./notebook_results/CXA_process2_7_19/")
lcx_out_dir.mkdir(parents=True, exist_ok=True)
# mapping_path = lcx_out_dir / "iomin_all_sci2sci2metric.json"
mapping_path = lcx_out_dir / "iou_all_sci2sci2metric.json"


sci2sci2metric = json.load(open(mapping_path, "r"))


# all_scs = prep_scs_from_mask_dataset(d2_mask_dataset, d2_mask_dataset)
all_tracked_scs = SingleCellTrajectoryCollection.load_from_json_file(
    lcx_out_dir / "sctc_filled_SORT_bbox_max_age_3_min_hits_1.json"
).get_all_scs()

filtered_tracked_scs = filter_boundary_cells(all_tracked_scs, dist_to_boundary=30, use_box_center=False)


print("# of cp_scs", len(all_tracked_scs))
print("# of filtered_cp_scs", len(filtered_tracked_scs))
print("# of filtered cells", len(all_tracked_scs) - len(filtered_tracked_scs))

# ## Apply CSN to all the cells


id2sc = {sc.id: sc for sc in all_tracked_scs}

# Use only the date in the filename
csn_out = (
    lcx_out_dir
    / "csn_apply_all"
    / f"v18-404-correct-ALL-scs-{datetime.now().strftime('%Y%m%d')}_PAD={PADDING_PIXELS}_OUTTH={OUT_THRESHOLD}"
)


csn_viz_out = csn_out / "viz"
csn_mask_out = csn_out / "mask"
csn_out.mkdir(parents=True, exist_ok=True)
csn_viz_out.mkdir(parents=True, exist_ok=True)
csn_mask_out.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame(columns=["sc_id", "label_str", "#ws-scs"])
save_df_interval = 1


for idx, _sc in tqdm.tqdm(enumerate(filtered_tracked_scs)):
    # out_mask_transformed, watershed_mask, label_str = correct_sc_mask(sc_from, model, PADDING_PIXELS, input_transforms, gpu=True)
    res_dict = correct_sc(_sc, model, PADDING_PIXELS, input_transforms, gpu=True, return_outputs=True)
    out_mask_transformed = res_dict["out_mask"]
    watershed_mask = res_dict["watershed_mask"]
    label_str = res_dict["label_str"]

    _sc.meta["csn_out_aux_label"] = label_str

    results_df = results_df.append(
        {
            "sc_id": _sc.id,
            "label_str": label_str,
            "#ws-scs": len(np.unique(watershed_mask)) - 1,
        },
        ignore_index=True,
    )
    if idx % save_df_interval == 0:
        results_df.to_csv(csn_out / "results.csv", index=False)
    # # # Save resaults
    # save_png(out_mask_transformed[0], csn_mask_out / f"raw-seg-{_sc.id}_from.png", mode="RGB") # 3 channel, (SEG, US, OS)
    # save_png(watershed_mask, csn_mask_out / f"watershed-sc-{_sc.id}_from.png")

    # np.save(csn_mask_out / f"npy-raw-seg-{_sc.id}_from.npy", out_mask_transformed[0], allow_pickle=False)
    # np.save(csn_mask_out / f"npy-watershed_sc-{_sc.id}_from.npy", watershed_mask, allow_pickle=False)


SingleCellStatic.write_single_cells_json(filtered_tracked_scs, csn_out / "csn_applied_scs.json")
