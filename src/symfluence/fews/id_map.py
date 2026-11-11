# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Variable ID mapper for FEWS <-> SYMFLUENCE translation.

Import (FEWS -> SYMFLUENCE) supports three tiers:

1. Explicit entries (inline config or YAML file)
2. Auto-detected common forcing ids (``P.obs`` -> ``pptrate`` ...), accepted
   only when the incoming units already are the SYMFLUENCE units -- the
   auto-map carries no conversion, so ``mm/h`` precipitation or ``degC``
   temperature needs an explicit entry with ``conversion_factor`` /
   ``conversion_offset``
3. Pass-through (names match)

Export (SYMFLUENCE -> FEWS) uses explicit entries only. Whether simulated
``discharge`` is ``Q.sim``, ``Q.fcst`` or something else is a decision of the
FEWS configuration; guessing it from the import table used to export
simulations under the observation id ``Q.obs``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import xarray as xr

from .config import FEWSConfig, IDMapEntry
from .exceptions import IDMappingError

logger = logging.getLogger(__name__)

# Common FEWS forcing parameterId -> (SYMFLUENCE name, SYMFLUENCE units).
# Units of None cannot be verified and are passed through with a warning.
_AUTO_MAP: Dict[str, Tuple[str, Optional[str]]] = {
    "P.obs": ("pptrate", "mm/s"),
    "P.forecast": ("pptrate", "mm/s"),
    "P": ("pptrate", "mm/s"),
    "T.obs": ("airtemp", "K"),
    "T.forecast": ("airtemp", "K"),
    "T": ("airtemp", "K"),
    "RH": ("relhum", None),
    "SW": ("SWRadAtm", "W/m2"),
    "LW": ("LWRadAtm", "W/m2"),
    "WS": ("windspd", "m/s"),
    "SP": ("airpres", "Pa"),
    "Q.obs": ("discharge", "m3/s"),
    "Q.sim": ("discharge", "m3/s"),
    "H.obs": ("water_level", "m"),
    "H.sim": ("water_level", "m"),
    "ET": ("pet", None),
    "E.act": ("aet", None),
}

# Spellings treated as the same unit (compared lower-case, without spaces).
_UNIT_SYNONYMS = (
    {"mm/s", "mms-1", "kgm-2s-1", "kg/m2/s", "kgm^-2s^-1"},
    {"k", "kelvin"},
    {"w/m2", "wm-2", "w/m^2", "wm^-2"},
    {"m/s", "ms-1", "ms^-1"},
    {"pa"},
    {"m3/s", "m^3/s", "m3s-1", "m^3s^-1", "cms", "m\u00b3/s"},
    {"m"},
)


def _normalise_unit(unit: str) -> str:
    return "".join(str(unit).lower().split())


def units_equivalent(a: Optional[str], b: Optional[str]) -> bool:
    """True when two unit strings denote the same unit (known spellings only)."""
    if a is None or b is None:
        return False
    na, nb = _normalise_unit(a), _normalise_unit(b)
    if na == nb:
        return True
    return any(na in group and nb in group for group in _UNIT_SYNONYMS)


class IDMapper:
    """Bidirectional variable name/unit translator between FEWS and SYMFLUENCE.

    Args:
        config: FEWSConfig with inline id_map and id_map_file
    """

    def __init__(self, config: FEWSConfig) -> None:
        self._fews_to_sym: Dict[str, IDMapEntry] = {}
        self._sym_to_fews: Dict[str, IDMapEntry] = {}
        # fews ids that came from the auto-map (no explicit conversion)
        self._auto_ids: Set[str] = set()
        # export targets declared inline, and export names with >1 candidate
        self._inline_exports: Set[str] = set()
        self._ambiguous_exports: Dict[str, List[str]] = {}

        # Tier 1: explicit inline entries
        for entry in config.id_map:
            self._add_explicit(entry, override=True)

        # Tier 1b: YAML file entries (inline entries win)
        if config.id_map_file:
            self._load_yaml(Path(config.id_map_file))

        # Tier 2: auto-detect common forcings (import only)
        if config.auto_id_map:
            self._add_auto_map()

    def _add_explicit(self, entry: IDMapEntry, override: bool) -> None:
        """Register an explicit entry in the direction(s) it declares."""
        if entry.conversion_factor == 0:
            raise IDMappingError(
                f"Conversion factor is zero for {entry.fews_id} <-> {entry.symfluence_id}"
            )
        if (
            entry.fews_unit and entry.symfluence_unit
            and not units_equivalent(entry.fews_unit, entry.symfluence_unit)
            and entry.conversion_factor == 1.0 and entry.conversion_offset == 0.0
        ):
            raise IDMappingError(
                f"ID map entry {entry.fews_id} <-> {entry.symfluence_id} declares "
                f"different units ('{entry.fews_unit}' vs '{entry.symfluence_unit}') but "
                "no conversion_factor / conversion_offset."
            )
        direction = getattr(entry, "direction", "both")
        if direction in ("import", "both"):
            if override or entry.fews_id not in self._fews_to_sym:
                self._fews_to_sym[entry.fews_id] = entry
        if direction in ("export", "both"):
            existing = self._sym_to_fews.get(entry.symfluence_id)
            if existing is None:
                self._sym_to_fews[entry.symfluence_id] = entry
                if override:
                    self._inline_exports.add(entry.symfluence_id)
            elif existing.fews_id != entry.fews_id and (
                override or entry.symfluence_id not in self._inline_exports
            ):
                # Two candidates from the same source (inline entries beat the
                # YAML file). Not an error until that variable is exported.
                self._ambiguous_exports.setdefault(
                    entry.symfluence_id, [existing.fews_id]
                ).append(entry.fews_id)

    def _load_yaml(self, path: Path) -> None:
        """Load ID mappings from a YAML file."""
        if not path.is_file():
            logger.warning("ID map file not found: %s", path)
            return

        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f)
        except Exception as exc:  # noqa: BLE001 — wrap-and-raise to domain error
            raise IDMappingError(f"Failed to load ID map YAML {path}: {exc}") from exc

        if not isinstance(data, list):
            raise IDMappingError(f"ID map YAML must be a list of mappings, got {type(data).__name__}")

        for item in data:
            self._add_explicit(IDMapEntry(**item), override=False)

    def _add_auto_map(self) -> None:
        """Add auto-detected common variable mappings (Tier 2, import only)."""
        for fews_id, (sym_id, sym_unit) in _AUTO_MAP.items():
            if fews_id not in self._fews_to_sym:
                self._fews_to_sym[fews_id] = IDMapEntry(
                    fews_id=fews_id, symfluence_id=sym_id, symfluence_unit=sym_unit,
                )
                self._auto_ids.add(fews_id)

    def fews_to_symfluence(self, fews_id: str) -> Tuple[str, float, float]:
        """Map a FEWS variable ID to SYMFLUENCE name with conversion.

        Args:
            fews_id: FEWS parameterId

        Returns:
            Tuple of (symfluence_name, conversion_factor, conversion_offset)
        """
        if fews_id in self._fews_to_sym:
            entry = self._fews_to_sym[fews_id]
            return entry.symfluence_id, entry.conversion_factor, entry.conversion_offset

        # Tier 3: pass-through
        return fews_id, 1.0, 0.0

    def has_export_mapping(self, sym_id: str) -> bool:
        """True when ``sym_id`` has an explicit SYMFLUENCE -> FEWS entry."""
        return sym_id in self._sym_to_fews

    def symfluence_to_fews(self, sym_id: str) -> Tuple[str, float, float]:
        """Map a SYMFLUENCE variable name to FEWS parameterId with reverse conversion.

        Args:
            sym_id: SYMFLUENCE variable name

        Returns:
            Tuple of (fews_id, inverse_factor, inverse_offset)

        Raises:
            IDMappingError: If there is no explicit export entry for ``sym_id``
        """
        entry = self._sym_to_fews.get(sym_id)
        if entry is None:
            raise IDMappingError(
                f"No explicit ID-map entry exports '{sym_id}' to FEWS. Add one "
                f"(e.g. fews_id: Q.sim, symfluence_id: {sym_id}, direction: export) "
                "to FEWS_ID_MAP or the --id-map file."
            )
        if sym_id in self._ambiguous_exports:
            raise IDMappingError(
                f"'{sym_id}' has several export mappings "
                f"({self._ambiguous_exports[sym_id]}). Mark the entries that only "
                "apply to incoming data with direction: import."
            )
        # Reverse conversion: fews = (sym - offset) / factor
        factor = entry.conversion_factor
        return entry.fews_id, 1.0 / factor, -entry.conversion_offset / factor

    def _check_incoming_units(self, fews_id: str, entry: IDMapEntry, da: xr.DataArray) -> None:
        """Refuse data whose units the mapping cannot account for."""
        units = da.attrs.get("units")
        if fews_id in self._auto_ids:
            expected = entry.symfluence_unit
            if expected is None or not units:
                logger.warning(
                    "Auto-mapped %s -> %s without a unit check (%s); add an explicit "
                    "ID-map entry to state the conversion.",
                    fews_id, entry.symfluence_id,
                    "no units in the FEWS data" if not units else "no reference unit",
                )
            elif not units_equivalent(units, expected):
                raise IDMappingError(
                    f"FEWS variable '{fews_id}' is in '{units}' but SYMFLUENCE "
                    f"'{entry.symfluence_id}' is in '{expected}', and the automatic "
                    "mapping applies no conversion. Add an explicit ID-map entry with "
                    "conversion_factor / conversion_offset."
                )
        elif entry.fews_unit and units and not units_equivalent(units, entry.fews_unit):
            raise IDMappingError(
                f"FEWS variable '{fews_id}' is in '{units}' but its ID-map entry "
                f"declares fews_unit '{entry.fews_unit}'."
            )

    def rename_dataset_fews_to_sym(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename dataset variables from FEWS IDs to SYMFLUENCE names and apply conversion.

        Args:
            ds: Dataset with FEWS variable names

        Returns:
            Dataset with SYMFLUENCE variable names and converted values

        Raises:
            IDMappingError: On a unit mismatch, or when two variables of the
                dataset map to the same SYMFLUENCE name (e.g. ``P.obs`` and
                ``P.forecast``): how to combine them is not the adapter's call
        """
        targets: Dict[str, List[str]] = {}
        for var in ds.data_vars:
            targets.setdefault(self.fews_to_symfluence(str(var))[0], []).append(str(var))
        clashes = {sym: ids for sym, ids in targets.items() if len(ids) > 1}
        if clashes:
            detail = "; ".join(f"{ids} -> '{sym}'" for sym, ids in clashes.items())
            raise IDMappingError(
                f"Several FEWS variables map to the same SYMFLUENCE variable: {detail}. "
                "Export a single (merged) series from FEWS, or map them to distinct names."
            )

        converted = {}
        for var in ds.data_vars:
            name = str(var)
            da = ds[name]
            entry = self._fews_to_sym.get(name)
            if entry is None:
                converted[name] = da  # Tier 3: pass-through
                continue
            self._check_incoming_units(name, entry, da)
            attrs = dict(da.attrs)
            if entry.conversion_factor != 1.0 or entry.conversion_offset != 0.0:
                da = da * entry.conversion_factor + entry.conversion_offset
                attrs.pop("units", None)
            if entry.symfluence_unit:
                attrs["units"] = entry.symfluence_unit
            da.attrs = attrs
            converted[entry.symfluence_id] = da
        return xr.Dataset(converted, coords=ds.coords, attrs=ds.attrs)

    def rename_dataset_sym_to_fews(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename dataset variables from SYMFLUENCE names to FEWS IDs and apply reverse conversion.

        Only variables with an explicit export entry are returned; the others
        are left out (and logged), never exported under a guessed id.

        Args:
            ds: Dataset with SYMFLUENCE variable names

        Returns:
            Dataset with FEWS variable names and reverse-converted values

        Raises:
            IDMappingError: If no variable of the dataset has an export entry
        """
        converted = {}
        skipped: List[str] = []
        for var in ds.data_vars:
            name = str(var)
            if not self.has_export_mapping(name):
                skipped.append(name)
                continue
            entry = self._sym_to_fews[name]
            fews_name, inv_factor, inv_offset = self.symfluence_to_fews(name)
            da = ds[name]
            attrs = dict(da.attrs)
            if inv_factor != 1.0 or inv_offset != 0.0:
                da = da * inv_factor + inv_offset
                attrs.pop("units", None)
            if entry.fews_unit:
                attrs["units"] = entry.fews_unit
            da.attrs = attrs
            converted[fews_name] = da

        if skipped:
            logger.warning(
                "Not exported to FEWS (no explicit ID-map export entry): %s", skipped
            )
        if not converted:
            raise IDMappingError(
                f"None of the model output variables {skipped} has an explicit ID-map "
                "export entry, so nothing can be exported to FEWS. Add entries such as "
                "{fews_id: Q.sim, symfluence_id: discharge, direction: export}."
            )
        return xr.Dataset(converted, coords=ds.coords, attrs=ds.attrs)
