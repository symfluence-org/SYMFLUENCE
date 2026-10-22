"""Back-compat shim: moved to ``symfluence.core.modeling.utilities.elevation_forcing``."""
from __future__ import annotations

import symfluence.core.modeling.utilities.elevation_forcing as _impl
from symfluence.core.modeling.utilities.elevation_forcing import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)
