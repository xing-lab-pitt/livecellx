from typing import List, Union, Tuple
from typing import Dict
import skimage
import skimage.measure
from pandas import Series
import numpy as np
import pandas as pd

from livecellx.core.single_cell import SingleCellStatic
from livecellx.livecell_logger import main_info
from livecellx.preprocess.utils import normalize_img_to_uint8
from livecellx.core.parallel import parallelize


def compute_dims_corr(cell_features_1: pd.DataFrame, cell_features_2: pd.DataFrame, sort_by_abs=True):
    feature_cols = cell_features_2.columns
    feature_corr_df = pd.DataFrame()
    for feature in feature_cols:
        for dim in cell_features_1.columns:
            _embedding = cell_features_1[dim]
            # suffer from NAN
            # corr = np.corrcoef(np.array(sc_feature_table[feature]), _embedding)[0, 1]

            # avoid and exclude NA values
            _tmp_df = pd.DataFrame({"embedding": _embedding, "feature": cell_features_2[feature]})
            all_corrs = _tmp_df.corr()
            corr = all_corrs["feature"].loc["embedding"]
            new_df = pd.DataFrame({"feature": [feature], "corr": [corr], "dim": [dim]})
            feature_corr_df = pd.concat([feature_corr_df, new_df], ignore_index=True)

    # sort feature_corr_df by corr
    dim2feature_corr_dict = {}
    for dim in cell_features_1.columns:
        dim2feature_corr_dict[dim] = feature_corr_df[feature_corr_df["dim"] == dim].sort_values(
            by="corr", ascending=False, key=lambda x: abs(x)
        )
        # add a rank column
        dim2feature_corr_dict[dim]["rank"] = np.arange(len(dim2feature_corr_dict[dim]))
    return dim2feature_corr_dict
