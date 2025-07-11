"""Top level package for :mod:`livecellx`.

This package previously imported a large collection of submodules on import.
While convenient, doing so caused the package import to fail when optional
dependencies (for example ``numpy`` or ``torch``) were missing.  The test
suite only relies on a small subset of the code base and should therefore be
able to import ``livecellx`` without pulling in heavy dependencies.  To avoid
import errors during test collection we no longer eagerly import all
submodules.  Users can still access the subpackages directly, e.g.::

    from livecellx.preprocess import utils

or

    from livecellx.core import sc_mapping

The ``__all__`` attribute lists the main subpackages that are available.
"""

__all__ = [
    "annotation",
    "preprocess",
    "plot",
    "segment",
    "core",
    "track",
    "trajectory",
    "sample_data",
]

from importlib import import_module


def __getattr__(name):
    if name in __all__:
        module = import_module(f"livecellx.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
