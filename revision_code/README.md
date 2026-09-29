# Final-submission revision code

This directory contains the selected revision sources for the final manuscript.
The selection is based on the main and supplementary submission PDFs inspected
on 2026-09-29. It is not a new evaluation or a rerun of tracking.

- [Figure-to-code map](FIGURE_CODE_MAP.md): adopted panels, inputs, outputs,
  dependencies, and the revision-only scope.
- [GitHub upload guide](GITHUB_UPLOAD.md): explicit upload scope and package changes.
- [Upload file list](GITHUB_UPLOAD_FILES.txt): repository-root-relative paths.
- [Complete data/model release](REVISION_RELEASE.md): assets packaged alongside code.

## Scope

The directory-level .gitignore is an explicit allowlist. Unused scripts,
notebooks, backup files, manuscript PDFs, model weights, images, masks, trajectory
JSONs, logs, cached features, and generated figures are excluded from normal Git
history. Required data, weights and adopted outputs are included in separate
revision release archives. Nothing is deleted.
Historical directory names are retained so existing relative paths still work.

The final revision figure entry points are:

| Panels | Entry point |
| --- | --- |
| Fig. 4g; Extended Data Fig. 4k | process2_trajectory_level_metrics.py |
| Fig. 4h-i, two annotated cases | results_wrongmask_altanative1_downsampling/plot_final_figure_4_trajectory_cases.py |
| Fig. 5d, selected GT/SORT/LiveCellX lineages | process2_downstream_analysis.py: plot_gt_lineages_4f_original_by_mother |
| Extended Data Fig. 3i, 1,014 cases | generate_process2_all_corrected_csn_metrics.py |

Several older-named scripts are retained **as dependencies**, not as alternative
final figure entry points. In particular, the CTC-style, trajectory-statistics,
and downstream-analysis modules provide functions used by the adopted scripts.
Do not run all Python files in this folder as a batch.

## Environment

Use a compatible LiveCellX environment and install this checkout in editable
mode from the repository root (`python -m pip install -e .`). Figure scripts
use NumPy, pandas, SciPy, Matplotlib, scikit-image, and scikit-learn in addition
to the package dependencies. The tracking pipeline also needs its compatible
PyTorch/torchvision/Lightning stack; TimeSformer inference needs MMAction2,
MMEngine/MMCV and video decoding dependencies.

No new dependency versions were inferred or installed during curation.
The existing scientific environment must be reproduced separately before a
fresh-machine end-to-end run can be claimed. Only load trusted checkpoint files.

## Regenerate Adopted Panels

Run from the repository root after restoring the external input data listed in
FIGURE_CODE_MAP.md. These commands overwrite their existing figure outputs.

```bash
python revision_code/process2_trajectory_level_metrics.py
python revision_code/results_wrongmask_altanative1_downsampling/plot_final_figure_4_trajectory_cases.py
python revision_code/generate_process2_all_corrected_csn_metrics.py
```

Fig. 5d uses the selected-lineage function and full method collections; its
separate command is in FIGURE_CODE_MAP.md. Extended Data Table 1 evaluation
materials are explicitly outside this release.

The first command uses the GT-region collections and frames 0-99. It generates
the SORT/LiveCellX figure and the SORT/LiveCellX/Ultrack figure together, plus
source/debug CSV tables. It does not rerun model inference.

The second command uses the edited GT1 collection and explicitly paired
LiveCellX/SORT case collections. It does not rematch IDs. It also generates
morphology/PCA/overlay diagnostics that were not adopted in Fig. 4h-i; those
additional outputs are ignored by Git and are not identified as final panels.

The third command evaluates temporal-reference correction cases across its
input mask sequence. This is not the manually annotated GT trajectory benchmark
and does not inherit the first command's 0-99 evaluation window.

The first two commands support `--output-dir` and input path overrides;
see `--help`. The third uses the repository-relative constants in
generate_process2_csn_cell_match_rate.py.

## Optional Tracking / Data Preparation

These are reusable tools, not substitutes for the fixed, manually annotated
collections underlying the published benchmark:

- run_sort_tracking.py: initial SORT from paired image and label-mask folders.
- run_livecellx_pipeline.py: post-SORT patches, CS-Net, corrected-mask export,
  corrected SORT, post-patches, identity repair, and TimeSformer lineage.
- livecellx_post_sort_patches.py: shared patch/repair implementations.
- filter_after_livecellx_sctc_to_gt_region.py: spatial GT-region restriction.
- rebuild_ultrack_sctc_from_tracks_df_with_lineage.py: reconstruct Ultrack
  tracks and lineage from matching label images and exported track tables.
- data/downsample_process2_alternative1_frames.py: paired every-fifth-frame
  extraction, using helpers in downsample_process1_process2_frames.py.

Example for a NEW dataset/output directory, not a reproduction of every
historical benchmark configuration:

```bash
python revision_code/run_sort_tracking.py \
  --image-dir /path/to/images --mask-dir /path/to/label_masks \
  --output-dir /path/to/new_run --max-age 5 --min-hits 1

python revision_code/run_livecellx_pipeline.py \
  --image-dir /path/to/images --mask-dir /path/to/label_masks \
  --output-dir /path/to/new_run --max-age 5 --min-hits 1 \
  --csnet-model /path/to/last.ckpt \
  --timesformer-model /path/to/best_acc_top1_epoch_65.pth
```

The TimeSformer default config is now configs/timesformer_divst_v15.py, a
byte-identical copy of the former work-directory config. No inference settings
were changed. The loader's existing test-pipeline overrides still apply.
Inference checkpoint weights are bundled in separate release archives. Original
training datasets for retraining the pretrained networks are not included.

Trajectory dataset references and image/mask paths use the checkout root as
their relative base. For portability, preserve the data's relative folder
layout and include referenced dataset-description files in a separate data
release. Copying trajectory JSONs alone is insufficient.
