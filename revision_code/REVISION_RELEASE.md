# Selected Revision Release

## Scope

Main Fig. 4g-i and Fig. 5d; Extended Data Fig. 3i and Fig. 4k.
The bundle contains selected sources, required shared code, annotated and
method trajectory collections, image/mask inputs, adopted output/source files,
Ultrack exp4 exports and the two pipeline inference checkpoints.

Extended Data Table 1 evaluation materials, old manuscript panels, unused
experimental outputs and model-training datasets are not included.
Shared modules and full upstream collections are retained for dependencies
and provenance, not presented as additional paper panels.

## Final Folder

`revision_code/revision_release/` is the upload folder.
`UPLOAD_ASSETS.txt` inside it lists the attachments; `SHA256SUMS.txt` checks
their bytes. Do not upload temporary or backup directories.

`revision_code.tar.gz` contains sources and the package snapshot.
`revision_data.tar.gz` contains inputs, annotations and adopted outputs.
`revision_csnet_model.tar.gz` contains `revision_code/model/last.ckpt`.
`revision_timesformer_model.tar.gz` contains
`revision_code/model/best_acc_top1_epoch_65.pth`.

The model paths are recorded by the case pipeline manifest. Their inclusion
does not publish the deferred table's experiments or establish permission
for third-party data; authors must confirm public-sharing rights.

## Build or Verify Without Recomputing

For a future complete rebuild into a NEW directory:

```bash
python revision_code/prepare_revision_release.py --output-dir /path/to/new/release
```

The builder copies and packages existing files only. It does not rerun tracking,
inference, metrics or plotting, and never overwrites source data.
Selected Fig. 5d PNG/PDF/SVG outputs and source CSVs are included; the unadopted
all-lineages figure is excluded.

```bash
python revision_code/prepare_revision_release.py --verify revision_code/revision_release
```

This verifies archive contents, checksums and recorded reference closure.
After changing only sources, `--refresh-code PATH` refreshes the code archive
and file manifests without changing data/model bytes. Regenerate attachment
checksums afterward; the companion publication notes are separate files.

## Restore

Extract the four archives into the SAME new empty directory. Alternatively,
use the reviewed Git version and extract just data and model archives into
that clean checkout. Never extract over working annotations without a backup.

Dataset JSONs in the bundle use checkout-relative paths. Original local
collections were not edited. Preserve the directory hierarchy; copying only
a trajectory JSON is insufficient.

Use README.md and FIGURE_CODE_MAP.md for the four plotting entry points.
The standalone source figures may differ from final manuscript assembly in
titles, legend placement and panel layout. Manual annotations cannot be
recreated by simply rerunning inference; they are provided explicitly.

The bundle is not a claim of retraining reproducibility, independent numerical
validation, deterministic tracking IDs or complete GUI stability.
