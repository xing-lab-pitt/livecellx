"""Core functionality for :mod:`livecellx`.

This package previously imported a number of submodules on import, which pulled
in heavy optional dependencies such as :mod:`pytorch_lightning`.  The test suite
and lightweight consumers only require access to the classes and functions
defined in these submodules and can import them individually.  To avoid
mandatory installation of the heavy dependencies, the eager imports have been
removed.

The public submodules are still available and listed in ``__all__`` so that
``from livecellx.core import single_cell`` continues to work if the
dependencies are installed.
"""

from importlib import import_module

__all__ = [
    "single_cell",
    "utils",
    "datasets",
    "parallel",
    "sc_mapping",
    "SingleCellStatic",
    "SingleCellTrajectory",
    "SingleCellTrajectoryCollection",
]


def __getattr__(name):
    if name in {"SingleCellStatic", "SingleCellTrajectory", "SingleCellTrajectoryCollection"}:
        module = import_module("livecellx.core.single_cell")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in {"single_cell", "utils", "datasets", "parallel", "sc_mapping"}:
        module = import_module(f"livecellx.core.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
