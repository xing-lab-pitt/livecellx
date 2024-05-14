import json

import numpy as np

# def load_multimaps(sctc_scs_path, all_scs):
#     time2multimaps__id = json.load(open(time2multi_maps_path, "r"))

#     all_scs_ids = set([sc.id for sc in all_scs])
#     id2sc = {sc.id: sc for sc in all_scs}
#     time2multimaps = {}
#     for time in time2multimaps__id:
#         time2multimaps[time] = []
#         for _map in time2multimaps__id[time]:
#             map_id = _map["map_from"]
#             _scs_ids = _map["map_to"]
#             assert map_id in all_scs_ids, f"sc1_id {map_id} not in sctc_scs_ids"
#             for sc2_id in _scs_ids:
#                 assert sc2_id in all_scs_ids, f"sc2_id {sc2_id} not in sctc_scs_ids"
#             time2multimaps[time].append(
#                 {
#                     "map_from": id2sc[map_id],
#                     "map_to": [id2sc[sc2_id] for sc2_id in _scs_ids],
#                 }
#             )
#     return time2multimaps


def standardize_label_name(label):
    """od->overseg_dropout; o->overseg; u->underseg; c->correct"""
    if label == "od":
        return "overseg_dropout"
    elif label == "o":
        return "overseg"
    elif label == "u":
        return "underseg"
    elif label == "c":
        return "correct"
    elif label == "d":
        return "discard"
    elif np.isnan(label):
        return "unknown"
    else:
        assert False, f"unknown label {label}"


def load_multimaps(merged_annotation_df, id2sc):
    """
    Load multimaps from a merged annotation DataFrame.

    Args:
        merged_annotation_df (pandas.DataFrame): The merged annotation DataFrame. Required columns: "map_id", "class", "map". "map" should be a python string representation of a dictionary with keys "map_from" and "map_to".
        id2sc (dict): A dictionary mapping IDs to SC values.

    Returns:
        list: A list of multimaps, where each multimaps is a dictionary with the following keys:
            - "map_from": The source SC value.
            - "map_to": A list of target SC values.
            - "label": The standardized label name.
            - "map_id": The map ID.
    """
    multimaps = []
    for i, row in merged_annotation_df.iterrows():
        map_id = row["map_id"]
        label = row["class"]
        _map = eval(row["map"])
        sc1_id = _map["map_from"]

        sc2_ids = _map["map_to"]
        sc1 = id2sc[sc1_id]
        sc2_ids = [sc2_id for sc2_id in sc2_ids]
        sc2s = [id2sc[sc2_id] for sc2_id in sc2_ids]
        multimaps.append(
            {
                "map_from": sc1,
                "map_to": sc2s,
                "label": standardize_label_name(label),
                "map_id": map_id,
            }
        )
    return multimaps
