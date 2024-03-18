import argparse
from pathlib import Path
from typing import List
from matplotlib import pyplot as plt

import numpy as np
import tqdm
from livecellx.core.parallel import parallelize
from livecellx.core.single_cell import (
    SingleCellStatic,
    SingleCellTrajectory,
    SingleCellTrajectoryCollection,
    sample_samples_from_sctc,
)
from livecellx.track.utils_CMU_C2C12 import *
from livecellx.core.datasets import LiveCellImageDatasetManager, LiveCellImageDataset


argparser = argparse.ArgumentParser(description="Process annotation for CMU C2C12 dataset")
argparser.add_argument("--xml_path", type=str, help="Path to the xml file")
argparser.add_argument("--out_dir", type=str, help="Output directory", required=True)
argparser.add_argument("--img_dir", type=str, help="Path to the image directory", required=True)
argparser.add_argument("--classes", type=list, help="List of classes", default=["mitosis", "normal"])
argparser.add_argument("--force_recalculate", action="store_true", help="Force recalculate")
args = argparser.parse_args()


def scs_from_CMU_frame_cell_data(cell_data, dataset: LiveCellImageDataset):
    scs = []
    for i, timeframe in enumerate(cell_data["timepoints"]):
        y, x = cell_data["xcoords"][i], cell_data["ycoords"][i]
        # Exclude NaN values
        if np.isnan(x) or np.isnan(y):
            continue
        timeframe = int(timeframe)
        # Testing purpose
        # if timeframe > 10:
        #     break
        sc = SingleCellStatic(timeframe=timeframe, id=cell_data["cellID"], empty_cell=True, img_dataset=dataset)
        sc.meta["cell_status"] = cell_data["cellStatus"][i]
        sc.meta["cmu_x"] = x
        sc.meta["cmu_y"] = y
        sc.meta["cmu_cell_id"] = cell_data["cellID"]
        sc.meta["src_dir"] = str(dataset.data_dir_path)
        scs.append(sc)
    return scs


def make_pseudo_square_contour(sc, bbox_size=20, dim_thresholds=None):
    x, y = sc.meta["cmu_x"], sc.meta["cmu_y"]
    # contour = np.array([[x-5, y-5], [x+5, y-5], [x+5, y+5], [x-5, y+5]], dtype=int)
    contour = np.array(
        [
            [x - bbox_size, y - bbox_size],
            [x + bbox_size, y - bbox_size],
            [x + bbox_size, y + bbox_size],
            [x - bbox_size, y + bbox_size],
        ],
        dtype=int,
    )
    if dim_thresholds is not None:
        contour[:, 0] = np.clip(contour[:, 0], 0, dim_thresholds[0] - 1)
        contour[:, 1] = np.clip(contour[:, 1], 0, dim_thresholds[1] - 1)
    sc.update_contour(contour)
    return sc


def make_pseudo_square_contour_sct(sct: SingleCellTrajectory, bbox_size=20):
    for t, sc in sct:
        make_pseudo_square_contour(sc, bbox_size=bbox_size)
    return sct


def make_pseudo_square_contour_wrapper(sc, dim_thresholds=None):
    make_pseudo_square_contour(sc, dim_thresholds=dim_thresholds)
    return sc


def is_last_timeframe_mitotic(sct: SingleCellTrajectory) -> bool:
    span = sct.get_time_span()
    last_sc = sct.get_single_cell(span[1])
    return last_sc.meta["cell_status"] == MITOTIC_STATUS_CODE


def any_mitosis_in_sct(sct: SingleCellTrajectory) -> bool:
    for t, sc in sct:
        if sc.meta["cell_status"] == MITOTIC_STATUS_CODE or sc.meta["cell_status"] == MITOTIC_OR_APOPTOTIC_CODE:
            return True
    return False


def main():
    xml_path = Path(args.xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"{xml_path} does not exist")
    out_dir = args.out_dir
    out_dir = Path(out_dir) / xml_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    import xml.etree.ElementTree as ET

    # xml_data = xml_path.read_text()
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Extracting all 'fs' nodes
    fs_nodes = root.findall("fs")

    # Extracting lineage-centric and frame-centric data
    lineage_centric_data = extract_lineage_centric_info(fs_nodes)
    frame_centric_data = extract_frame_centric_info(fs_nodes)

    # Construct image dataset
    img_dir_path = args.img_dir
    # img_dir_path = dataset_dir_path / "images"
    dataset = LiveCellImageDataset(img_dir_path, ext="tif")

    inputs = []
    for cell_data in tqdm.tqdm(frame_centric_data):
        inputs.append(tuple([cell_data, dataset]))

    outputs = parallelize(scs_from_CMU_frame_cell_data, inputs, cores=32)
    scs = [sc for sublist in outputs for sc in sublist]

    print("# of total scs:", len(scs))
    print("# unique cell IDs:", len(set([sc.meta["cmu_cell_id"] for sc in scs])))

    num_mitotic = len([sc for sc in scs if sc.meta["cell_status"] == MITOTIC_STATUS_CODE])
    print("# of mitotic scs:", num_mitotic)
    unique_mitotic_cell_ids = set(
        [sc.meta["cmu_cell_id"] for sc in scs if sc.meta["cell_status"] == MITOTIC_STATUS_CODE]
    )
    print("# of unique mitotic cell IDs:", len(unique_mitotic_cell_ids))

    num_mitotic_or_apoptotic = len([sc for sc in scs if sc.meta["cell_status"] == MITOTIC_OR_APOPTOTIC_CODE])
    print("# of mitotic or apoptotic cells:", num_mitotic_or_apoptotic)
    unique_mitotic_or_apoptotic_cell_ids = set(
        [sc.meta["cmu_cell_id"] for sc in scs if sc.meta["cell_status"] == MITOTIC_OR_APOPTOTIC_CODE]
    )
    print("# of unique mitotic or apoptotic cell IDs:", len(unique_mitotic_or_apoptotic_cell_ids))

    out_dir.mkdir(parents=True, exist_ok=True)
    filename = xml_path.stem
    scs_out_path = out_dir / f"{filename}_scs.json"

    dims = scs[0].get_img().shape[:2]
    if not scs_out_path.exists() or args.force_recalculate:
        print("Making pseudo square contours...")
        inputs = [(sc, dims) for sc in scs]
        outputs = parallelize(make_pseudo_square_contour_wrapper, inputs, cores=32)
        scs = outputs
        print(f"Writing single cells to {scs_out_path}")
        SingleCellStatic.write_single_cells_json(scs, scs_out_path)
    else:
        scs = SingleCellStatic.load_single_cells_json(scs_out_path)

    # Check and build trajectories
    print("Building trajectories...")
    cellId_to_scs = {}
    for sc in scs:
        if sc.id not in cellId_to_scs:
            cellId_to_scs[sc.id] = []
        cellId_to_scs[sc.id].append(sc)

    sctc = SingleCellTrajectoryCollection()
    for cell_id in cellId_to_scs:
        cellId_to_scs[cell_id] = sorted(cellId_to_scs[cell_id], key=lambda x: x.timeframe)
        _scs = cellId_to_scs[cell_id]
        for i, sc in enumerate(_scs):
            if sc == _scs[-1]:
                continue
            if sc.timeframe + 1 != _scs[i + 1].timeframe:
                print(f"Error: {sc.timeframe} and {_scs[i+1].timeframe} are not consecutive timeframes")

            # assert _scs[i+1].timeframe == _scs[i].timeframe + 1, f"Error: {_scs[i].timeframe} and {_scs[i+1].timeframe} are not consecutive timeframes"
        t2sc = {}
        for sc in _scs:
            t2sc[int(sc.timeframe)] = sc
        sct = SingleCellTrajectory(track_id=cell_id, timeframe_to_single_cell=t2sc)
        assert len(sct) > 0
        sctc.add_trajectory(sct)

    # Visualize last 8 frames of mitosis trajectories
    mitosis_preview_out_dir = out_dir / "all_mitosis_preview"
    mitosis_preview_out_dir.mkdir(parents=True, exist_ok=True)
    nc = 5
    for tid, full_sct in tqdm.tqdm(sctc):
        full_span = full_sct.get_time_span()
        sct = full_sct.subsct(full_span[1] - 8, full_span[1])
        scs = sct.get_all_scs()
        if scs[0].meta["cell_status"] == MITOTIC_STATUS_CODE:
            nr = len(sct) // nc + 1
            fig, axes = sct.show_on_grid(padding=50, show_mask=False, nr=nr, nc=nc)
            plt.savefig(mitosis_preview_out_dir / f"{tid}.png")

    mitosis_sctc = SingleCellTrajectoryCollection()
    window = 8
    for tid, full_sct in tqdm.tqdm(sctc):
        full_span = full_sct.get_time_span()
        if not is_last_timeframe_mitotic(full_sct):
            continue
        scs = sct.get_all_scs()
        sct = full_sct.subsct(full_span[1] - window, full_span[1])
        mitosis_sctc.add_trajectory(sct)

    mitosis_sctc_out_path = out_dir / f"mitosis_sctc.json"
    print(f"Writing mitosis trajectories to {mitosis_sctc_out_path}")
    mitosis_sctc.write_json(mitosis_sctc_out_path)

    normal_sctc = SingleCellTrajectoryCollection()
    for tid, full_sct in tqdm.tqdm(sctc):
        if any_mitosis_in_sct(full_sct):
            continue
        normal_sctc.add_trajectory(full_sct)
    normal_sctc_out_path = out_dir / f"normal_sctc.json"
    print(f"Writing normal trajectories to {normal_sctc_out_path}")
    normal_sctc.write_json(normal_sctc_out_path)

    print("# of mitosis trajectories:", len(mitosis_sctc))
    print("# of normal trajectories:", len(normal_sctc))

    def write_samples(samples: List[List[SingleCellStatic]], class_name, out_dir):
        out_dir = out_dir / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, sample in enumerate(samples):
            SingleCellStatic.write_single_cells_json(sample, out_dir / f"{i}.json")

    mitosis_samples = [sct.get_all_scs() for tid, sct in mitosis_sctc]
    normal_samples, normal_samples_extra_info = sample_samples_from_sctc(
        normal_sctc, objective_sample_num=len(mitosis_samples) * 10
    )

    print("# mitosis samples:", len(mitosis_samples))
    print("# normal samples:", len(normal_samples))

    write_samples(mitosis_samples, "mitosis", out_dir)
    write_samples(normal_samples, "normal", out_dir)


if __name__ == "__main__":
    main()
