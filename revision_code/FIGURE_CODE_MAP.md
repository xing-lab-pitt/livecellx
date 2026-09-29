# Final Figure Source Map

Scope: the two PDFs in final_submission, inspected on 2026-09-29.
Panel mapping below is based on captions, script outputs, and saved source
tables. Figures were not regenerated or numerically revalidated during this
curation. Main Fig. 4 and Extended Data Fig. 4 are different figure numbers.

## Adopted Revision Analyses

### Fig. 4g and Extended Data Fig. 4k

Entry: `process2_trajectory_level_metrics.py`.

Outputs under `latest_results/trajectory_level_metrics/`:

- Fig. 4g: `fig_process2_five_tracking_metrics_sort_livecellx.{png,pdf}`.
- Extended Data Fig. 4k: `fig_process2_five_tracking_metrics.{png,pdf}`.
- The LiveCellX/Ultrack-only variant is an extra output, not assigned a final panel.
- Source values: `process2_five_tracking_metrics_summary.csv`; per-track,
  division and detection-match tables are saved alongside it.

Default inputs under `results_process2_csnet/final_results_traj_collections/`:

| Role | File |
| --- | --- |
| GT | sctc-final-2026-06-10_100_corrected_motherdaughter_corrected_gt_region_only_missing_fixed.json |
| SORT | sctc-final-2026-0610_before_livecellx_gt_region_only.json |
| LiveCellX | sctc-final-2026-06-02_before_correction_timesformer_lineage_corrected_gt_region_only.json |
| Ultrack | ultrack_from_cellpose_process2_beforecsnet_exp4_gt_region_only.json |

These are GT-region collections, evaluated over frames 0-99. Metrics are ATR,
ATP, ATA, AssA and BC(5), not the old DET/SEG/LNK/TF/CT figure. BC requires
explicit lineage; missing lineage is not silently converted to a zero score.

Helper: `process2_ctc_style_tracking_evaluation.py` supplies collection loading,
GT tables, spatial matching, and relationship utilities. Its standalone old
CTC-style figure is not an adopted final revision panel.

The saved five-metric summary is the source for this mapping. Do not reuse the
older `process2_trajectory_level_summary.csv` (SoftDiv/LCR analysis).

### Fig. 4h-i

Entry: `results_wrongmask_altanative1_downsampling/plot_final_figure_4_trajectory_cases.py`.

Input folder, relative to that script:
`traj_collections/final_figure_4_collections/`.

Inputs: `livecellx_traj_gt1.json` (edited GT),
`sort_traj71_107.json`, and `livecellx_traj102_168.json`.

| GT / LiveCellX | SORT | Inclusive downsampled frame window |
| --- | --- | --- |
| 168 | 107 | 0-10 |
| 102 | 71 | 20-30 |

Output folders under the same script directory:

- `trajectory_based_results/gt_traj_0168_annotated_traj/`.
- `trajectory_based_results/gt_traj_0102_annotated_traj/`.

Fig. 4h uses `center_distance_to_gt.png`; Fig. 4i uses
`contour_center.png`. Their respective source tables are
`center_distance_to_gt_series.csv` and `contour_center_series.csv`.

The plot uses contour-mask centroids, not bounding-box centers. The edited
GT1 input still has the legend label GT. Keep the annotated collections and
their referenced dataset/image files together in the external data release.

Dependencies: `process2_downstream_analysis.py`,
`process2_trajectory_statistics.py`, and
`process2_ctc_style_tracking_evaluation.py`. These large modules are retained
unchanged apart from the packaged model-config path; helper extraction would
risk altering previously generated results and is not part of this curation.

### Extended Data Fig. 3i

Entry: `generate_process2_all_corrected_csn_metrics.py`.

Helper: `generate_process2_csn_cell_match_rate.py` supplies masks, overlap
functions, and IoU thresholds (0.5, 0.6, 0.7).

Inputs:
`data/process2/2D_mask/*_cp_masks.png` and
`results_process2_csnet/livecellx_corrected_masks/frame_*.png`.

Adopted output:
`latest_results/figures/fig3i_process2_all_corrected_csn_cell_match_rate.{png,pdf}`.

Saved source:
`results_process2_csnet/process2_all_corrected_csn_cell_match_rate.csv`.
Its six rows each report 1,014 cases, agreeing with the final caption.
The script uses adjacent-frame masks as temporal references, not independently
annotated GT. It averages per-case cell-match rates. Do not describe this as
a GT segmentation benchmark or substitute the all-cells/underseg-only variant.

## Fig. 5d: Selected Mother-Daughter Lineages

Entry function: `process2_downstream_analysis.plot_gt_lineages_4f_original_by_mother`.
GT mothers are 1528 and 1534; the GT-region GT collection and FULL SORT/LiveCellX
collections listed above are used, restricted to frames 0-99. No Ultrack is shown.

Under `latest_results/downstream_analysis/gt_lineage_4f_original/`, retain:
`fig_gt_lineages_4f_original_selected`,
`fig_sort_lineages_4f_original_selected`,
`fig_livecellx_lineages_4f_original_selected` (PNG/PDF/SVG and source CSV).
The all-lineages output is not the adopted panel and is not bundled.

To regenerate into a NEW output location from the checkout root:

```bash
PYTHONPATH=revision_code python - <<'PY'
from pathlib import Path
import process2_downstream_analysis as d
gt = d.load_sctc(d.GT_PATH)
sort = d.load_sctc(d.DOWNSTREAM_SORT_BEFORE_FULL_PATH)
livecellx = d.load_sctc(d.DOWNSTREAM_SORT_AFTER_FULL_PATH)
d.plot_gt_lineages_4f_original_by_mother(
    gt, Path("revision_code/reproduced_fig5d"), min_frame=0, max_frame=99,
    before_sctc=sort, after_sctc=livecellx)
PY
```

The helper also generates an all-GT diagnostic in that output location; only
the three selected-lineage files belong to the paper panel. Titles, axis wording,
legend placement and panel assembly in the final PDF were edited separately;
the standalone outputs are source plots, not pixel-identical final layouts.

## Old-to-Final Scope Crosscheck

Local comparison: `ms_11-23.pdf` and `supplement.pdf` versus the final PDFs.

- Old main Fig. 3a-f -> final Fig. 4a-f: existing schematic/example panels.
- Old main Fig. 3g,h,i,j,k -> final Extended Data Fig. 4g,j,l,m,n:
  pre-existing correction-round, ROC and gap analyses, not new revision runs.
- Old main Fig. 4a-c,d,f -> final Fig. 5a-c,e,f: existing lineage panels.
  Final Fig. 5d is the new selected GT/SORT/LiveCellX comparison.
- Old main Fig. 5 -> final Fig. 6: existing downstream analyses.
- Old Extended Data Fig. 3a-h,i,j -> final Extended Data Fig. 3a-h,j,k:
  existing CS-Net evaluations; final Fig. 3i is the new temporal-reference result.
  The final caption of panel j says IoU 0.8 where the old caption said 0.9;
  this legacy-caption discrepancy is not validated by the new panel-i code.
- Old Extended Data Fig. 4g,h -> final Fig. 4h,i: relocated existing analyses.
- Old Extended Data Fig. 6c,d -> final Fig. 5g,h: existing PyTorch/diffusion panels.
- Extended Data Table 1 evaluation materials: explicitly excluded from this
  release by author instruction. Pipeline checkpoints do not constitute a
  release of that table's experiments.

This comparison identifies the new numerical panels covered here; it does not
audit ownership or reproduce every pre-existing panel in the paper.

## Retained Processing Dependencies

| File | Reason retained |
| --- | --- |
| run_livecellx.py | Dataset loading and corrected-mask export imported by the generic pipelines |
| result_process12_newtracking_fig4/run_process12_newtracking_fig4.py | Coarse-to-fine TimeSformer and lineage functions imported by run_livecellx_pipeline.py |
| result_process12_newtracking_fig4/reindex_sctc_zero_based.py | Referenced by that helper's standalone workflow |
| configs/timesformer_divst_v15.py | Exact config formerly available only in an untracked training work directory |
| tests/test_livecellx_post_sort_patches.py | Existing focused patch regression tests |
| ../notebooks/CXA_2D_multiround_correction_with_tracking_benchmark.py | CS-Net multiround implementation imported at runtime |

GT-region filtering and Ultrack lineage reconstruction scripts are included for
input preparation. They do not replace manual GT annotation. Ultrack exp4
conversion requires the matching label sequence, tracks_df.csv and graph.json;
do not mix files from different experiments. The historical exp1-exp3 converters
and ad hoc lineage-patching scripts are excluded.

## Scope Boundary

Only revision analyses are being released. Original manuscript panels and their
historical code provenance are outside this task. Existing tracked package code
is included as a dependency, not as a new audit of old figures.

## Excluded Local Material

Excluded from new uploads: old CTC/SoftDiv/LCR entry variants, generic rebuttal
plotters, runtime tables, synthetic demonstration plots, candidate-search and
padding experiments, duplicated notebooks/launchers, backups, unadopted generated
outputs and manuscript PDFs. The adopted outputs, checkpoint weights and required
raw/annotated datasets are included separately in the revision release archives
(see REVISION_RELEASE.md), not in ordinary Git history.
Dependencies listed above are explicit exceptions. Exclusion is via Git rules;
all local files remain available.
