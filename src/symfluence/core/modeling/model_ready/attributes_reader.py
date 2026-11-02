# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Reader for the model-ready attributes store.

Provides ``open_canonical_attributes`` — the single entry point model
preprocessors call to read per-HRU catchment attributes (terrain, soil,
landcover, climate, ...) from the grouped NetCDF written by
``AttributesNetCDFBuilder``.  Returns ``None`` when the store is absent so
callers transparently fall back to reading the source shapefiles, mirroring
the "open returns None ⇒ use legacy" contract used by ``forcing_reader`` and
``path_resolver``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

from .path_resolver import resolve_model_ready_path

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

logger = logging.getLogger(__name__)


def open_canonical_attributes(
    project_dir: Path,
    domain_name: str,
) -> Optional["AttributesReader"]:
    """Open the model-ready attributes store for a domain.

    Args:
        project_dir: Root of the SYMFLUENCE domain directory.
        domain_name: Name of the hydrological domain.

    Returns:
        An :class:`AttributesReader`, or ``None`` if no attributes store
        exists (so callers keep their legacy shapefile path).
    """
    attrs_dir = resolve_model_ready_path(project_dir, 'attributes')
    if not attrs_dir.exists():
        return None

    candidate = attrs_dir / f'{domain_name}_attributes.nc'
    if not candidate.exists():
        matches = sorted(attrs_dir.glob('*_attributes.nc'))
        if not matches:
            return None
        candidate = matches[0]

    try:
        return AttributesReader(candidate)
    except Exception as e:  # noqa: BLE001 — absent/corrupt store ⇒ legacy fallback
        logger.debug("Could not open attributes store %s: %s", candidate, e)
        return None


class AttributesReader:
    """Thin reader over the grouped attributes NetCDF.

    The store keeps each attribute family in its own group (``terrain``,
    ``soil``, ``landcover``, ``climate``, ``hydrogeology``, ``hru_identity``,
    ``topology``, plus ``*_extended`` CSV groups).  Per-HRU groups share the
    ``hru`` dimension and carry an ``hru_id`` coordinate for safe joins.
    """

    def __init__(self, store_path: Path) -> None:
        """Open the store and cache its group list.

        Args:
            store_path: Path to ``{domain}_attributes.nc``.
        """
        import netCDF4

        self.store_path = store_path
        with netCDF4.Dataset(str(store_path), 'r') as ds:
            self._groups: List[str] = list(ds.groups.keys())

    @property
    def groups(self) -> List[str]:
        """List of groups present in the store."""
        return list(self._groups)

    def has_group(self, group: str) -> bool:
        """Return True if *group* exists in the store."""
        return group in self._groups

    def group(self, name: str) -> "xr.Dataset":
        """Open a single group as an xarray Dataset (caller closes it)."""
        import xarray as xr

        return xr.open_dataset(self.store_path, group=name)

    def has_variable(self, group: str, var: str) -> bool:
        """Return True if *var* exists in *group*."""
        if not self.has_group(group):
            return False
        with self.group(group) as ds:
            return var in ds.variables

    def to_dataframe(self, group: str, index: Optional[str] = None) -> "pd.DataFrame":
        """Return a group as a DataFrame, optionally indexed by *index*."""
        with self.group(group) as ds:
            df = ds.to_dataframe().reset_index()
        if index is not None and index in df.columns:
            df = df.set_index(index)
        return df

    def variable(
        self,
        group: str,
        var: str,
        *,
        reduce: Optional[str] = None,
    ):
        """Read a single variable from a group.

        Args:
            group: Group name.
            var: Variable name.
            reduce: Optional reduction — ``'mean'``, ``'min'``, ``'max'`` —
                applied with NaN-skipping to return a scalar.

        Returns:
            The variable's numpy array, a reduced float, or ``None`` if the
            group/variable is missing.
        """
        import numpy as np

        if not self.has_group(group):
            return None
        with self.group(group) as ds:
            if var not in ds.variables:
                return None
            values = ds[var].values
        if reduce is None:
            return values
        reducers = {'mean': np.nanmean, 'min': np.nanmin, 'max': np.nanmax}
        if reduce not in reducers:
            raise ValueError(f"Unknown reduce '{reduce}'; expected one of {list(reducers)}")
        return float(reducers[reduce](values))

    def find_variable(
        self,
        group: str,
        keywords: Sequence[str],
        *,
        reduce: Optional[str] = None,
    ):
        """Read the first variable in *group* whose name contains any keyword.

        Case-insensitive substring match, useful when the exact column name
        varies by source (e.g. ``sand_0-5cm_mean`` vs ``sand_frac``).
        """
        if not self.has_group(group):
            return None
        with self.group(group) as ds:
            for name in ds.variables:
                lname = str(name).lower()
                if any(kw.lower() in lname for kw in keywords):
                    return self.variable(group, str(name), reduce=reduce)
        return None

    def hru_ids(self, group: str = 'hru_identity') -> Optional[List[str]]:
        """Return the ordered ``hru_id`` list from a group, or None."""
        ids = self._read_ids(group, 'hru_id')
        return ids

    def per_hru_values(self, group: str, var: str) -> Optional[Dict[str, float]]:
        """Return a ``{hru_id: value}`` mapping for a per-HRU variable.

        Joins *var* against the group's own ``hru_id`` coordinate, so the
        mapping is order-independent and safe across groups.  Returns ``None``
        if the group, the variable, or the ``hru_id`` coordinate is missing.
        """
        ids = self._read_ids(group, 'hru_id')
        if ids is None:
            return None
        values = self.variable(group, var)
        if values is None or len(values) != len(ids):
            return None
        return {hid: float(v) for hid, v in zip(ids, values)}

    def _read_ids(self, group: str, id_var: str) -> Optional[List[str]]:
        """Read a string id coordinate from a group as a list of str."""
        import numpy as np

        if not self.has_group(group):
            return None
        with self.group(group) as ds:
            if id_var not in ds.variables:
                return None
            return [str(x) for x in np.atleast_1d(ds[id_var].values)]
