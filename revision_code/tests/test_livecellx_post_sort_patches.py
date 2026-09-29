import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from livecellx.core import SingleCellStatic, SingleCellTrajectory, SingleCellTrajectoryCollection
from livecellx_post_sort_patches import repair_identity_switches, reconnect_short_trajectories


def make_cell(cell_id, timeframe, y, x, size=10):
    contour = np.asarray(
        [
            [y, x],
            [y, x + size - 1],
            [y + size - 1, x + size - 1],
            [y + size - 1, x],
        ],
        dtype=int,
    )
    return SingleCellStatic(
        id=cell_id,
        timeframe=timeframe,
        bbox=np.asarray([y, x, y + size, x + size], dtype=int),
        contour=contour,
        cached_img_shape=(100, 100),
    )


def make_traj(track_id, cells):
    return SingleCellTrajectory(
        track_id=track_id,
        timeframe_to_single_cell={int(cell.timeframe): cell for cell in cells},
    )


def test_repairs_new_track_identity_switch_without_losing_cells():
    primary = make_traj(
        1,
        [
            make_cell("b0", 0, 20, 10),
            make_cell("a1", 1, 20, 55),
            make_cell("a2", 2, 20, 56),
        ],
    )
    new_track = make_traj(
        2,
        [
            make_cell("b1", 1, 20, 11),
            make_cell("b2", 2, 20, 12),
        ],
    )
    sctc = SingleCellTrajectoryCollection([primary, new_track])
    before_ids = sorted(str(cell.id) for cell in sctc.get_all_scs())

    report = repair_identity_switches(sctc)

    assert report["identity_switches_repaired"] == 1
    assert str(sctc.get_trajectory(1).get_sc(1).id) == "b1"
    assert str(sctc.get_trajectory(1).get_sc(2).id) == "b2"
    assert str(sctc.get_trajectory(2).get_sc(1).id) == "a1"
    assert str(sctc.get_trajectory(2).get_sc(2).id) == "a2"
    assert sorted(str(cell.id) for cell in sctc.get_all_scs()) == before_ids


def test_does_not_swap_a_spatially_continuous_track():
    primary = make_traj(
        1,
        [make_cell("b0", 0, 20, 10), make_cell("b1", 1, 20, 11)],
    )
    unrelated = make_traj(2, [make_cell("a1", 1, 20, 55)])
    sctc = SingleCellTrajectoryCollection([primary, unrelated])

    report = repair_identity_switches(sctc)

    assert report["identity_switches_repaired"] == 0
    assert str(sctc.get_trajectory(1).get_sc(1).id) == "b1"


def test_strict_reconnect_accepts_gap_two_but_rejects_gap_three():
    source = make_traj(1, [make_cell("s0", 0, 20, 10)])
    target = make_traj(2, [make_cell("t2", 2, 20, 10)])
    sctc = SingleCellTrajectoryCollection([source, target])
    report = reconnect_short_trajectories(sctc)
    assert report["total_merges"] == 1
    assert len(sctc) == 1
    assert len(sctc.get_all_scs()) == 2

    source = make_traj(1, [make_cell("s0", 0, 20, 10)])
    target = make_traj(2, [make_cell("t3", 3, 20, 10)])
    sctc = SingleCellTrajectoryCollection([source, target])
    report = reconnect_short_trajectories(sctc)
    assert report["total_merges"] == 0
    assert len(sctc) == 2
