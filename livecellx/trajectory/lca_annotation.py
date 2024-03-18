from pathlib import Path
from livecellx.livecell_logger import main_info, main_warning
from livecellx.core.single_cell import SingleCellStatic
from livecellx.track.classify_utils import load_class2samples_from_json_dir, load_all_json_dirs


def load_hela_data(
    hela_json_dirs=[
        Path(r"../datasets/mitosis-annotations-2023/DIC-C2DH-HeLa/01_annotation"),
        Path(r"../datasets/mitosis-annotations-2023/DIC-C2DH-HeLa/02_annotation"),
    ]
):

    hela_all_class2samples, hela_all_class2sample_extra_info = load_all_json_dirs(hela_json_dirs)
    hela_scs = []
    mitosis_hela_cells = []
    for cls in hela_all_class2samples:
        main_info(f"Loading {cls} samples, #samples: {len(hela_all_class2samples[cls])}")
        for sample in hela_all_class2samples[cls]:
            hela_scs.extend(sample)
            if cls == "mitosis":
                mitosis_hela_cells.extend(sample)

    all_hela_cells_paths = [hela_dir / "single_cells.json" for hela_dir in hela_json_dirs]
    all_hela_scs = []

    for hela_cells_path in all_hela_cells_paths:
        _scs = SingleCellStatic.load_single_cells_json(hela_cells_path)
        all_hela_scs.extend(_scs)

    mitosis_hela_cell_ids = [sc.id for sc in mitosis_hela_cells]
    main_info(f"{len(all_hela_scs)} loaded")
    all_non_mitosis_hela_scs = [sc for sc in all_hela_scs if sc.id not in mitosis_hela_cell_ids]

    main_info(f"Loaded {len(mitosis_hela_cells)} mitosis static single cells in total")
    main_info(f"Loaded {len(all_non_mitosis_hela_scs)} interphase cells")
    main_info(f"Loaded {len(all_hela_scs)} single cells in total")

    for sc in mitosis_hela_cells:
        sc.meta["is_mitosis"] = True
    for sc in all_non_mitosis_hela_scs:
        sc.meta["is_mitosis"] = False

    return all_hela_scs, mitosis_hela_cells, all_non_mitosis_hela_scs


if __name__ == "__main__":
    load_hela_data()
