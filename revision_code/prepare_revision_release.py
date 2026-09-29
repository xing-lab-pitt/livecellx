#!/usr/bin/env python3
"""Build the revision-only code, data, source-results and model release.

No tracking or inference is run. Source files are never modified. Referenced
dataset JSONs, images and masks are discovered recursively from the selected
trajectory collections. Only exported JSON path fields are made repository-
relative; contours, IDs, features, times and lineage are preserved. The output
directory must not exist. Failed builds are left for inspection, not overwritten.

Usage: python revision_code/prepare_revision_release.py
       python revision_code/prepare_revision_release.py --verify RELEASE_DIR
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
REV = Path("revision_code")
FINAL = REV / "results_process2_csnet/final_results_traj_collections"
CASE = REV / "results_wrongmask_altanative1_downsampling"
ANNOTATED = CASE / "traj_collections/final_figure_4_collections"
PATH_KEYS = {"img_dataset_json_path", "mask_dataset_json_path", "dataset_json_dir", "data_dir_path"}
REFERENCE_KEYS = {"img_dataset_json_path", "mask_dataset_json_path"}

COLLECTIONS = [
    FINAL / "sctc-final-2026-06-10_100_corrected_motherdaughter_corrected_gt_region_only_missing_fixed.json",
    FINAL / "sctc-final-2026-0610_before_livecellx_gt_region_only.json",
    FINAL / "sctc-final-2026-06-02_before_correction_timesformer_lineage_corrected_gt_region_only.json",
    FINAL / "ultrack_from_cellpose_process2_beforecsnet_exp4_gt_region_only.json",
    FINAL / "sctc-final-2026-0610_before_livecellx_full.json",
    FINAL / "sctc-final-2026-06-02_before_correction_timesformer_lineage_corrected_full.json",
    FINAL / "ultrack_from_cellpose_process2_beforecsnet_exp4_full.json",
    ANNOTATED / "livecellx_traj_gt1.json",
    ANNOTATED / "sort_traj71_107.json",
    ANNOTATED / "livecellx_traj102_168.json",
    CASE / "traj_collections/sort.json",
    CASE / "traj_collections/livecellx_csnet_timesformer_lineage.json",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path, rows, columns):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


class Release:
    def __init__(self, out):
        self.out = out
        self.files = {}
        self.edges = set()
        self.json_queue = []
        self.processed = set()
        self.path_changes = 0
        self.stage = out / "_staging"
        self.stage.mkdir()

    def relative(self, value):
        p = Path(value).expanduser()
        p = p if p.is_absolute() else ROOT / p
        p = p.resolve()
        if not p.is_relative_to(ROOT):
            raise ValueError(f"Reference outside repository; review before publishing: {value}")
        return p.relative_to(ROOT).as_posix()

    def add(self, path, group, role, owner=None, scan_json=False):
        name = self.relative(path)
        source = ROOT / name
        if not source.is_file():
            raise FileNotFoundError(f"Required {role}: {source} (referenced by {owner})")
        if owner:
            self.edges.add((owner, name))
        if name not in self.files:
            self.files[name] = {"source": source, "export": source, "group": group, "roles": set()}
        elif self.files[name]["group"] != group:
            raise ValueError(f"Conflicting release groups for {name}")
        self.files[name]["roles"].add(role)
        if scan_json and name not in self.processed:
            self.json_queue.append(name)
        return name

    def glob(self, directory, pattern, group, role):
        paths = sorted((ROOT / directory).glob(pattern))
        paths = [p for p in paths if p.is_file()]
        if not paths:
            raise FileNotFoundError(f"No {role}: {directory}/{pattern}")
        for path in paths:
            self.add(path, group, role)

    def normalize(self, obj, owner):
        if isinstance(obj, list):
            for value in obj:
                self.normalize(value, owner)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                if key in PATH_KEYS and isinstance(value, str) and value not in {"", "None"}:
                    relative = self.relative(value)
                    if key in REFERENCE_KEYS:
                        self.add(relative, "data", "dataset_descriptor", owner, scan_json=True)
                    self.path_changes += int(relative != value)
                    obj[key] = relative
                elif key == "time2url" and isinstance(value, dict):
                    for time, url in value.items():
                        if not isinstance(url, str):
                            raise TypeError(f"Non-string image URL: {owner}, frame {time}")
                        relative = self.relative(url)
                        self.add(relative, "data", "referenced_image_or_mask", owner)
                        self.path_changes += int(relative != url)
                        value[time] = relative
                else:
                    self.normalize(value, owner)

    def scan(self):
        while self.json_queue:
            name = self.json_queue.pop()
            if name in self.processed:
                continue
            self.processed.add(name)
            print(f"[references] {name}", flush=True)
            record = self.files[name]
            with record["source"].open() as handle:
                data = json.load(handle)
            if "time2url" in data and not data["time2url"] and data.get("data_dir_path") not in {None, "", "None"}:
                raise ValueError(f"Dataset has only a directory, not an explicit frame mapping: {name}")
            self.normalize(data, name)
            target = self.stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w") as handle:
                json.dump(data, handle, ensure_ascii=True, separators=(",", ":"))
            record["export"] = target
            del data

    def select(self):
        code_paths = (ROOT / REV / "GITHUB_UPLOAD_FILES.txt").read_text().splitlines()
        tracked = subprocess.check_output(["git", "ls-files", "-z", "livecellx"], cwd=ROOT).decode().split("\0")
        for name in sorted(set(code_paths + [p for p in tracked if p] + ["pyproject.toml", "requirements.txt", "readme.md", "LICENSE"])):
            self.add(name, "code", "source_code")
        for name in COLLECTIONS:
            self.add(name, "data", "trajectory_collection", scan_json=True)
        for directory, pattern, role in [
            (REV / "data/process2/2D_mask", "*_cp_masks.png", "cell_match_input_masks"),
            (REV / "data/process2/2D_mask_renamed", "*.png", "initial_tracking_masks"),
            (REV / "data/process2/2D_mask_alternative1_renamed", "*.png", "downsampling_source_masks"),
            (REV / "data/process2/2D_8bit", "*.tiff", "input_images"),
            (REV / "data/process2_alternative1_downsample5/2D_8bit", "*.tiff", "case_input_images"),
            (REV / "data/process2_alternative1_downsample5/2D_mask_renamed", "*.png", "case_input_masks"),
            (REV / "results_process2_csnet/livecellx_corrected_masks", "frame_*.png", "cell_match_corrected_masks"),
            (CASE / "livecellx_corrected_masks", "*.png", "case_corrected_masks"),
        ]:
            self.glob(directory, pattern, "data", role)
        ultrack = REV / "tracking_from_cellpose_process2_beforecsnet_exp4"
        self.glob(ultrack / "c", "**/*", "data", "ultrack_label_chunk")
        for name in ("zarr.json", "tracks_df.csv", "graph.json"):
            self.add(ultrack / name, "data", "ultrack_exp4_export")
        for name in ("sort_manifest.json", "livecellx_pipeline_manifest.json"):
            self.add(CASE / "traj_collections" / name, "data", "run_provenance")
        self.add(REV / "data/process2_alternative1_downsample5/frame_mapping.csv", "data", "original_to_downsampled_frame_mapping")
        self.add(REV / "results_process2_csnet/livecellx_correction_rounds.csv", "data", "correction_provenance")
        metrics = REV / "latest_results/trajectory_level_metrics"
        self.glob(metrics, "process2_five_tracking_metrics_*.csv", "data", "metric_source_table")
        for stem in ("fig_process2_five_tracking_metrics", "fig_process2_five_tracking_metrics_sort_livecellx"):
            for ext in ("png", "pdf"):
                self.add(metrics / f"{stem}.{ext}", "data", "adopted_metric_figure")
        for name in ("process2_all_corrected_csn_cell_match_rate.csv", "process2_all_corrected_csn_cell_match_case_summary.csv", "process2_all_corrected_csn_case_type_counts.csv"):
            self.add(REV / "results_process2_csnet" / name, "data", "cell_match_source_table")
        for ext in ("png", "pdf"):
            self.add(REV / "latest_results/figures" / f"fig3i_process2_all_corrected_csn_cell_match_rate.{ext}", "data", "adopted_cell_match_figure")
        for tid in (168, 102):
            folder = CASE / "trajectory_based_results" / f"gt_traj_{tid:04d}_annotated_traj"
            for name in ("contour_center.png", "center_distance_to_gt.png", "contour_center_series.csv", "center_distance_to_gt_series.csv"):
                self.add(folder / name, "data", "adopted_case_figure_or_source_table")
        lineage = REV / "latest_results/downstream_analysis/gt_lineage_4f_original"
        for method in ("gt", "sort", "livecellx"):
            stem = f"fig_{method}_lineages_4f_original_selected"
            for ext in ("png", "pdf", "svg"):
                self.add(lineage / f"{stem}.{ext}", "data", "adopted_fig5d_lineage_figure")
            self.add(lineage / f"{stem}_source_data.csv", "data", "fig5d_lineage_source_table")

        self.add(REV / "model/last.ckpt", "csnet_model", "csnet_inference_checkpoint")
        self.add(REV / "model/best_acc_top1_epoch_65.pth", "timesformer_model", "timesformer_inference_checkpoint")
        self.scan()

    def archive(self, groups=("code", "data", "csnet_model", "timesformer_model")):
        rows = []
        artifacts = []
        for group in groups:
            archive = self.out / f"revision_{group}.tar.gz"
            names = sorted(n for n, r in self.files.items() if r["group"] == group)
            print(f"[archive] {group}: {len(names)} files", flush=True)
            with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
                for index, name in enumerate(names):
                    record = self.files[name]
                    path = record["export"]
                    before = record["source"].stat()
                    source_hash = sha256(record["source"])
                    export_hash = source_hash if path == record["source"] else sha256(path)
                    info = tar.gettarinfo(str(path), arcname=name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mode = 0o755 if os.access(record["source"], os.X_OK) else 0o644
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
                    after = record["source"].stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise RuntimeError(f"Source changed during packaging: {name}")
                    rows.append({"archive": archive.name, "path": name, "roles": ";".join(sorted(record["roles"])), "source_bytes": before.st_size, "release_bytes": path.stat().st_size, "source_sha256": source_hash, "release_sha256": export_hash})
                    if index % 200 == 0:
                        print(f"  {index + 1}/{len(names)}", flush=True)
            artifacts.append({"archive": archive.name, "files": len(names), "bytes": archive.stat().st_size, "sha256": sha256(archive)})
        write_tsv(self.out / "FILES.tsv", rows, ["archive", "path", "roles", "source_bytes", "release_bytes", "source_sha256", "release_sha256"])
        write_tsv(self.out / "ARTIFACTS.tsv", artifacts, ["archive", "files", "bytes", "sha256"])
        write_tsv(self.out / "DEPENDENCIES.tsv", [{"owner": owner, "dependency": dep} for owner, dep in sorted(self.edges)], ["owner", "dependency"])
        shutil.rmtree(self.stage)
        return artifacts


def verify(directory):
    with (directory / "FILES.tsv").open() as handle:
        expected = {row["path"]: row for row in csv.DictReader(handle, delimiter="\t")}
    seen = set()
    with (directory / "ARTIFACTS.tsv").open() as handle:
        artifacts = list(csv.DictReader(handle, delimiter="\t"))
    for row in artifacts:
        archive = directory / row["archive"]
        if sha256(archive) != row["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {archive}")
        with tarfile.open(archive, "r|gz") as tar:
            for member in tar:
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise ValueError(f"Unexpected or duplicate archive member: {member.name}")
                record = expected[member.name]
                if record["archive"] != archive.name or member.size != int(record["release_bytes"]):
                    raise ValueError(f"Archive membership/size mismatch: {member.name}")
                digest = hashlib.sha256()
                with tar.extractfile(member) as handle:
                    for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != record["release_sha256"]:
                    raise ValueError(f"File checksum mismatch: {member.name}")
                seen.add(member.name)
        print(f"[verified] {archive.name}", flush=True)
    if seen != set(expected):
        raise ValueError("Missing files in release archives")
    with (directory / "DEPENDENCIES.tsv").open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["owner"] not in seen or row["dependency"] not in seen:
                raise ValueError(f"Missing referenced dataset/image: {row}")
    print(f"[verified] {len(seen)} files and all recorded dataset dependencies", flush=True)



def refresh_code(directory):
    """Refresh only code assets; retain byte-identical data/model archives."""
    with (directory / "FILES.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    with (directory / "ARTIFACTS.tsv").open() as handle:
        artifacts = list(csv.DictReader(handle, delimiter="\t"))
    with tempfile.TemporaryDirectory(prefix="code-refresh-", dir=directory) as temp:
        work = Path(temp)
        builder = Release(work)
        names = (ROOT / REV / "GITHUB_UPLOAD_FILES.txt").read_text().splitlines()
        tracked = subprocess.check_output(["git", "ls-files", "-z", "livecellx"], cwd=ROOT).decode().split("\0")
        for name in sorted(set(names + [p for p in tracked if p] + ["pyproject.toml", "requirements.txt", "readme.md", "LICENSE"])):
            builder.add(name, "code", "source_code")
        replacement = builder.archive(groups=("code",))
        with (work / "FILES.tsv").open() as handle:
            new_rows = list(csv.DictReader(handle, delimiter="\t"))
        os.replace(work / "revision_code.tar.gz", directory / "revision_code.tar.gz")
        rows = [r for r in rows if r["archive"] != "revision_code.tar.gz"] + new_rows
        artifacts = [r for r in artifacts if r["archive"] != "revision_code.tar.gz"] + replacement
        write_tsv(directory / "FILES.tsv", rows, ["archive", "path", "roles", "source_bytes", "release_bytes", "source_sha256", "release_sha256"])
        write_tsv(directory / "ARTIFACTS.tsv", artifacts, ["archive", "files", "bytes", "sha256"])
    verify(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / REV / "revision_release")
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--refresh-code", type=Path, help="Refresh source archive only; data/model bytes stay unchanged.")
    args = parser.parse_args()
    if args.refresh_code:
        refresh_code(args.refresh_code.resolve())
        return
    if args.verify:
        verify(args.verify.resolve())
        return
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    release = Release(out)
    release.select()
    artifacts = release.archive()
    readme = """# Selected revision release

This release covers Fig. 4g-i, Fig. 5d, Extended Data Fig. 3i and 4k,
their inputs and source results, full upstream tracking collections, and the
CS-Net / TimeSformer pipeline inference checkpoints. Extended Data Table 1
evaluation materials are excluded by author instruction. Old manuscript panels
and unused experimental outputs are excluded; shared code dependencies remain.
Training datasets for retraining these pretrained networks are not bundled.
This is not a release of every experiment reported in the paper.

## Contents
- revision_code.tar.gz: selected revision sources plus the existing package,
  installation metadata, license, and required benchmark implementation.
- revision_data.tar.gz: edited GT, method collections, recursively referenced
  dataset descriptors/images/masks, Ultrack exp4 exports, frame mapping,
  selected source tables and final adopted figure outputs.
- revision_csnet_model.tar.gz: model/last.ckpt.
- revision_timesformer_model.tar.gz: model/best_acc_top1_epoch_65.pth.
- FILES.tsv: original/released hashes and sizes, file roles, and archive paths.
- ARTIFACTS.tsv: archive hashes and sizes.
- DEPENDENCIES.tsv: every discovered dataset and image/mask reference.

## Restore
Create a NEW empty directory. Extract all four archives into that same directory
with `tar -xzf ARCHIVE -C NEW_DIRECTORY`. The directory becomes the checkout root.
Do not extract into an existing working dataset: some filenames may coincide.
In an existing Git clone, restore data/model archives only after backing up any
local files. Code/data use the same repository-relative layout.

Trajectory/dataset JSONs are compacted and their path fields made relative only
in these archives. Original files are unchanged. Contours, times, track IDs,
features and lineage were not edited. Some run-provenance records retain original
machine paths as historical evidence; they are not live dataset references.

Use a compatible scientific Python environment, install the checkout, then follow
revision_code/README.md and FIGURE_CODE_MAP.md for the adopted figure entry
points, including the selected-lineage function for Fig. 5d. Model weights
are unnecessary for redrawing from the archived collections, but are included
for tracking/inference. Load checkpoints only if you trust this release.

The collections contain manual annotation/editing; rerunning inference alone
does not recreate hand edits or guarantee identical trajectory IDs. This release
is not a claim of fresh-machine, deterministic, raw-data-to-paper reproduction.

Verification: `python revision_code/prepare_revision_release.py --verify PATH`.
All archives were checked by streaming their contents and comparing file hashes.
No training, tracking or figure regeneration is performed by the release builder.

Upload code changes to GitHub using revision_code/GITHUB_UPLOAD_FILES.txt. Publish
the four archives and accompanying manifests together as release/data assets;
do not add large data and checkpoint binaries to normal Git history. This script
does not publish anything and no external repository URL has been invented.
"""
    (out / "RELEASE_README.md").write_text(readme)
    (out / "BUILD_INFO.json").write_text(json.dumps({"python": platform.python_version(), "path_fields_normalized": release.path_changes, "code_commit_base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(), "note": "Working-tree source snapshot, including listed local modifications; not the clean commit."}, indent=2))
    packages = sorted({(dist.metadata.get("Name", "unknown"), dist.version) for dist in metadata.distributions()})
    write_tsv(out / "BUILD_ENVIRONMENT.tsv", [{"package": name, "version": version} for name, version in packages], ["package", "version"])
    verify(out)
    for artifact in artifacts:
        print(f"{artifact['archive']}: {artifact['bytes'] / 1024**3:.3f} GiB", flush=True)
    print(f"Release saved to {out}", flush=True)


if __name__ == "__main__":
    main()
