#!/usr/bin/env python
"""Reindex large LiveCellX collections for zero-based Napari display.

The established final-results convention maps the first existing source image
to timeframe 0. Missing physical frame numbers are collapsed consistently in
the image datasets and trajectory cells. Collections are streamed one
trajectory at a time to avoid loading multi-gigabyte JSON files into memory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple


TRACK_MAP_HEADER = '"track_id_to_trajectory"'
FRAME_RE = re.compile(r"(\d+)(?=[^\d]*$)")
TIMEFRAME_ZERO_RE = re.compile(rb'"timeframe"\s*:\s*0(?:\D|$)')


def log(message: str) -> None:
    print(message, flush=True)


def frame_number(path: str) -> int:
    match = FRAME_RE.search(Path(path).stem)
    if match is None:
        raise ValueError(f"Cannot extract a frame number from {path}")
    return int(match.group(1))


def load_dataset_time_map(dataset_path: Path) -> Tuple[Dict[int, int], dict]:
    with dataset_path.open() as handle:
        dataset = json.load(handle)
    time2url = dataset.get("time2url", {})
    if not time2url:
        raise ValueError(f"Dataset has no time2url entries: {dataset_path}")
    physical_times = sorted({frame_number(url) for url in time2url.values()})
    if len(physical_times) != len(time2url):
        raise ValueError(f"Duplicate physical frame numbers in {dataset_path}")
    return {old: new for new, old in enumerate(physical_times)}, dataset


def collection_has_timeframe_zero(path: Path, chunk_size: int = 8 * 1024 * 1024) -> bool:
    overlap = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return False
            data = overlap + chunk
            if TIMEFRAME_ZERO_RE.search(data):
                return True
            overlap = data[-64:]


def iter_track_entries(handle) -> Iterator[Tuple[str, dict]]:
    """Yield top-level track ID and trajectory objects from a pretty JSON file."""
    while True:
        line = handle.readline()
        if not line:
            raise ValueError("Could not find track_id_to_trajectory")
        if TRACK_MAP_HEADER in line:
            break

    entry_start = re.compile(r'^\s{8}"([^"]+)"\s*:\s*\{')
    lines = []
    track_id = None
    depth = 0
    in_string = False
    escaped = False

    for line in handle:
        if track_id is None:
            match = entry_start.match(line)
            if match is None:
                if line.startswith("    }"):
                    return
                continue
            track_id = match.group(1)
            lines = [line[line.index("{") :]]
        else:
            lines.append(line)

        for char in lines[-1]:
            if escaped:
                escaped = False
            elif char == "\\" and in_string:
                escaped = True
            elif char == '"':
                in_string = not in_string
            elif not in_string:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1

        if depth == 0:
            text = "".join(lines).rstrip()
            if text.endswith(","):
                text = text[:-1]
            yield track_id, json.loads(text)
            lines = []
            track_id = None

    if track_id is not None:
        raise ValueError(f"Unterminated trajectory object for track {track_id}")


def reindex_trajectory(trajectory: dict, time_map: Dict[int, int]) -> dict:
    new_cells = {}
    for old_key, cell in trajectory.get("timeframe_to_single_cell", {}).items():
        old_time = int(old_key)
        if old_time not in time_map:
            raise KeyError(f"Trajectory references absent source timeframe {old_time}")
        new_time = time_map[old_time]
        cell["timeframe"] = new_time
        new_cells[str(new_time)] = cell
    trajectory["timeframe_to_single_cell"] = new_cells
    return trajectory


def make_hardlink_backup(path: Path, suffix: str) -> Path:
    backup = path.with_name(path.name + suffix)
    if not backup.exists():
        os.link(path, backup)
        log(f"[backup] {backup}")
    return backup


def rewrite_collection(path: Path, time_map: Dict[int, int], backup_suffix: str) -> bool:
    if collection_has_timeframe_zero(path):
        log(f"[skip] already zero-based: {path}")
        return False
    make_hardlink_backup(path, backup_suffix)
    temporary = path.with_name(path.name + ".zero_based.tmp")
    track_count = 0
    cell_count = 0
    with path.open() as source, temporary.open("w") as target:
        target.write('{"track_id_to_trajectory":{')
        first = True
        for track_id, trajectory in iter_track_entries(source):
            trajectory = reindex_trajectory(trajectory, time_map)
            if not first:
                target.write(",")
            target.write(json.dumps(str(track_id)))
            target.write(":")
            json.dump(trajectory, target, separators=(",", ":"))
            first = False
            track_count += 1
            cell_count += len(trajectory.get("timeframe_to_single_cell", {}))
            if track_count % 100 == 0:
                log(f"[reindex] {path.name}: {track_count} trajectories")
        target.write("}}\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)
    log(f"[done] {path}: {track_count} trajectories, {cell_count} cells")
    return True


def canonical_dataset_url(dataset_path: Path, source_url: str, physical_time: int) -> str:
    """Use the same canonical image folders as the established final results."""
    process_name = dataset_path.parent.parent.name
    revision_dir = dataset_path.parents[3]
    data_name = "process1_30den" if process_name == "process1" else "process2"
    data_dir = revision_dir / "data" / data_name
    if "mask" in dataset_path.name.lower():
        candidate = data_dir / "2D_mask_renamed" / f"Fused_{physical_time:04d}_cp_masks.png"
    else:
        candidate = data_dir / "2D_8bit" / f"Fused_{physical_time:04d}.tiff"
    return str(candidate.resolve()) if candidate.exists() else source_url


def rewrite_dataset(
    dataset_path: Path,
    backup_suffix: str,
    preserve_dataset_urls: bool = False,
) -> bool:
    time_map, dataset = load_dataset_time_map(dataset_path)
    old_time2url = dataset["time2url"]
    url_by_physical = {frame_number(url): url for url in old_time2url.values()}
    if preserve_dataset_urls:
        new_time2url = {
            str(new): url_by_physical[old]
            for old, new in sorted(time_map.items(), key=lambda item: item[1])
        }
    else:
        new_time2url = {
            str(new): canonical_dataset_url(
                dataset_path, url_by_physical[old], old
            )
            for old, new in sorted(time_map.items(), key=lambda item: item[1])
        }
    current_keys = sorted(map(int, old_time2url))
    expected_keys = list(range(len(old_time2url)))
    if current_keys == expected_keys and old_time2url == new_time2url:
        log(f"[skip] dataset already zero-based: {dataset_path}")
        return False
    make_hardlink_backup(dataset_path, backup_suffix)
    dataset["time2url"] = new_time2url
    dataset["times"] = expected_keys
    dataset["time_indexing"] = "sequential_zero_based"
    temporary = dataset_path.with_name(dataset_path.name + ".zero_based.tmp")
    with temporary.open("w") as handle:
        json.dump(dataset, handle, indent=4)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, dataset_path)
    log(f"[done] dataset: {dataset_path}")
    return True


def reindex_outputs(
    collection_paths: Iterable[Path],
    dataset_dir: Path,
    backup_suffix: str = ".before_zero_based_fix",
    preserve_dataset_urls: bool = False,
) -> None:
    dataset_paths = sorted(dataset_dir.glob("*.json"))
    if not dataset_paths:
        raise FileNotFoundError(f"No dataset JSON files found in {dataset_dir}")
    raw = [path for path in dataset_paths if "raw" in path.name]
    reference = raw[0] if raw else dataset_paths[0]
    time_map, _ = load_dataset_time_map(reference)
    log(
        f"[mapping] {len(time_map)} source frames -> 0..{len(time_map) - 1}; "
        f"physical range {min(time_map)}..{max(time_map)}"
    )
    for path in collection_paths:
        rewrite_collection(Path(path), time_map, backup_suffix)
    for path in dataset_paths:
        rewrite_dataset(
            path,
            backup_suffix,
            preserve_dataset_urls=preserve_dataset_urls,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--collections", required=True, nargs="+", type=Path)
    parser.add_argument("--backup-suffix", default=".before_zero_based_fix")
    parser.add_argument(
        "--preserve-dataset-urls",
        action="store_true",
        help="Reindex time keys without replacing the existing image/mask URLs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reindex_outputs(
        args.collections,
        args.dataset_dir,
        args.backup_suffix,
        preserve_dataset_urls=bool(args.preserve_dataset_urls),
    )


if __name__ == "__main__":
    main()
