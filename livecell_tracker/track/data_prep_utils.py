from typing import Dict, List
from livecell_tracker.core.single_cell import SingleCellStatic


def drop_sample_div(sample: List[SingleCellStatic]):
    """
    Remove SingleCellStatic objects in samples where at the same timepoint, there are >=2 SingleCellStatic objects.
    This function is used to drop cell division events from the mitosis samples.

    Args:
    - sample: a list of SingleCellStatic objects representing a sample.

    Returns:
    - A new list of SingleCellStatic objects with cell division events removed.
    """
    sc_by_time = {}
    for sc in sample:
        if sc.timeframe not in sc_by_time:
            sc_by_time[sc.timeframe] = []
        sc_by_time[sc.timeframe].append(sc)

    new_sample = []
    for time, scs in sc_by_time.items():
        if len(scs) == 1:
            new_sample.append(scs[0])
    return new_sample


def check_one_sc_at_time(sample: List[SingleCellStatic]):
    """check if there is only one sc at each timepoint"""
    times = set()
    for sc in sample:
        if sc.timeframe in times:
            return False
        times.add(sc.timeframe)
    return True


from typing import Dict, List


def drop_multiple_cell_frames_in_samples(class2samples: Dict, tar_keys: List[str] = ["mitosis"]) -> Dict:
    """
    Remove SingleCellStatic objects in samples where at the same timepoint, there are >=2 SingleCellStatic objects.
    This function is used to drop cell division events from the mitosis samples.

    Args:
    - class2samples: a dictionary containing samples for each class
    - tar_keys: a list of keys of the classes to drop cell division events from. Default is ["mitosis"].

    Returns:
    - A dictionary containing samples for each class with cell division events removed from the specified classes.
    """
    class2samples = class2samples.copy()
    for key in tar_keys:
        tmp_samples = []
        key_samples = class2samples[key]
        for sample in key_samples:
            tmp_samples.append(drop_sample_div(sample))
        class2samples[key] = tmp_samples
        assert all(
            [check_one_sc_at_time(sample) for sample in class2samples[key]]
        ), "there is more than one sc at the same timepoint"
    return class2samples
