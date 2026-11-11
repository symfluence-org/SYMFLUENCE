# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Discretization namespacing for model-ready forcing filenames (issue #339).

The naming contract that lets ONE model-ready forcing store hold several spatial
discretizations of the same domain side by side: a remapped forcing is written as
``{domain}_{forcing}_remapped_{token}_{datetag}.nc``, where ``{token}`` identifies
the HRU set it was remapped onto. Writers (the remapping/resampling builders)
stamp the token; readers (every model's forcing load path) select by it.

The token is ``{method}-h{count}-{digest8}``: the discretization method
(``grus``, ``elevation``, ``elevation-landclass``), the number of HRUs, and the
first eight hex digits of a content hash of the target catchment shapefile
(:func:`hru_geometry`). The method alone is not enough: a lumped and a delineated
``grus`` run, 200 m and 400 m elevation bands, or a re-delineated domain all share
a method but not an HRU set. Two earlier forms are still read:

- ``{method}`` only (written between #379 and the geometry token), and
- no token at all (written before #379).

This lives in ``core`` because both ends of that contract sit in different layers —
``data`` writes the names, ``core.modeling`` and the model adapters read them — and
the convention itself is pure string handling with no I/O and no upper-layer
dependency. Keeping it here is what lets the core forcing processor scope its read
without importing the data layer.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from symfluence.core.exceptions import FileOperationError

logger = logging.getLogger(__name__)

# A namespaced remapped forcing name is
#   {domain}_{forcing}_remapped_{token}_{datetag}.nc  (or ..._remapped_{token}.nc)
# where {token} is the letter-initial discretization token written by
# discretization_token() (lumped, elevation, grus, ...). Legacy date-tag-only
# names ('..._remapped_2002-01-01-...') start with a digit and carry no token.
_REMAP_TOKEN_RE = re.compile(r'_remapped_([a-z][a-z0-9-]*?)(?:_|\.nc$)')

# {method}-h{count}-{digest8}: a geometry-aware token (see module docstring).
_GEOMETRY_TOKEN_RE = re.compile(r'^(?P<method>[a-z][a-z0-9-]*?)-h(?P<count>\d+)-(?P<digest>[0-9a-f]{8})$')

HRU_GEOMETRY_VERSION = 'hru-geometry-v1'
DIGEST_CHARS = 8


@dataclass(frozen=True)
class HruGeometry:
    """Identity of the HRU set a forcing is remapped onto.

    ``count`` is the number of HRUs (the ``hru`` dimension of the remapped
    file); ``digest`` is a sha256 hex digest of the CRS, the HRU IDs and every
    geometry in file order (:func:`hru_geometry`).
    """

    count: int
    digest: str

    @property
    def short(self) -> str:
        return self.digest[:DIGEST_CHARS]


_GEOMETRY_CACHE: Dict[Tuple, HruGeometry] = {}


def _crs_identity(crs) -> str:
    """A CRS identity that does not change between pyproj versions when it can."""
    if crs is None:
        return ''
    try:
        epsg = crs.to_epsg()
    except RuntimeError:  # pyproj CRSError; fall back to WKT
        epsg = None
    return f'EPSG:{epsg}' if epsg else crs.to_wkt()


def hru_geometry(catchment_path: Union[str, Path], hru_id_field: Optional[str] = None) -> HruGeometry:
    """Hash the HRU set in *catchment_path* into an :class:`HruGeometry`.

    The digest covers the CRS, the ``hru_id_field`` values (when the column
    exists) and each geometry's WKB, in file order, because the remapped
    ``hru`` axis follows file order. It hashes content, not file bytes, so
    re-writing an identical shapefile (a DBF header carries its write date)
    keeps the same name. Results are cached per path, size and mtime.
    """
    import geopandas as gpd
    import shapely

    path = Path(catchment_path)
    parts = [path] + ([path.with_suffix('.dbf'), path.with_suffix('.prj')] if path.suffix == '.shp' else [])
    stamp = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in parts if p.exists())
    key = (str(path.resolve()), hru_id_field, stamp)
    cached = _GEOMETRY_CACHE.get(key)
    if cached is not None:
        return cached

    gdf = gpd.read_file(path)
    digest = hashlib.sha256(HRU_GEOMETRY_VERSION.encode())
    digest.update(_crs_identity(gdf.crs).encode())
    if hru_id_field and hru_id_field in gdf.columns:
        digest.update(json.dumps([str(v) for v in gdf[hru_id_field]]).encode())
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            digest.update(b'\0')
        else:
            digest.update(shapely.to_wkb(geom, output_dimension=2, byte_order=1, include_srid=False))
    geometry = HruGeometry(count=len(gdf), digest=digest.hexdigest())
    _GEOMETRY_CACHE[key] = geometry
    return geometry


def forcing_token(discretization: object, geometry: Optional[HruGeometry] = None) -> str:
    """The filename token for *discretization* remapped onto *geometry*.

    ``{method}-h{count}-{digest8}`` with a geometry; the bare method token
    (:func:`discretization_token`) without one.
    """
    method = discretization_token(discretization)
    if geometry is None:
        return method
    return f'{method}-h{geometry.count}-{geometry.short}'


def split_token(token: str) -> Tuple[str, Optional[int], Optional[str]]:
    """Split a filename token into ``(method, hru_count, digest8)``.

    A method-only token (the pre-geometry form) returns ``(token, None, None)``.
    """
    m = _GEOMETRY_TOKEN_RE.match(token)
    if not m:
        return token, None, None
    return m.group('method'), int(m.group('count')), m.group('digest')


def discretization_token(discretization: object) -> str:
    """Return a filename-safe token identifying a spatial discretization.

    Lower-cased, with every run of non-alphanumerics collapsed to ``-``. An
    empty/unknown value maps to ``'default'``. Discretization values in practice
    are always letter-initial (``lumped``, ``elevation``, ``grus``,
    ``elevation,landclass`` -> ``elevation-landclass``); this token is what makes
    a lumped (``hru=1``) remap and, say, a 12-band elevation (``hru=12``) remap of
    the SAME domain get distinct, self-describing filenames instead of colliding
    under the shared ``{domain}_{forcing}_remapped_*`` namespace (issue #339).
    """
    tok = re.sub(r'[^a-z0-9]+', '-', str(discretization or '').strip().lower()).strip('-')
    return tok or 'default'


def forcing_name_matches_discretization(name: str, token: str) -> bool:
    """Whether *name* is a remapped forcing belonging to discretization *token*."""
    stem = Path(name).name
    return (f'_remapped_{token}_' in stem) or stem.endswith(f'_remapped_{token}.nc')


def discretization_key_from_name(name: str) -> Optional[str]:
    """Extract the discretization token from a remapped forcing filename, or None.

    ``None`` means the name predates namespacing (a date-tag-only legacy remap);
    such files share the synthetic ``'default'`` group so the original #339 catch
    still fires when two legacy untokened files disagree on spatial size.
    """
    m = _REMAP_TOKEN_RE.search(Path(name).name)
    return m.group(1) if m else None


def _hru_size(path: Path) -> Optional[int]:
    """The ``hru`` dimension size of a NetCDF file, or None if unreadable/absent."""
    import xarray as xr

    try:
        with xr.open_dataset(path) as ds:
            return int(ds.sizes['hru']) if 'hru' in ds.sizes else None
    except (OSError, ValueError) as e:
        logger.debug("Could not read hru size of %s: %s", path.name, e)
        return None


def select_forcing_files(
    forcing_files: Union[str, Path, List[Path]],
    discretization: object = None,
    geometry: Optional[HruGeometry] = None,
) -> List[Path]:
    """Pick the forcing files remapped onto a run's HRU set.

    A model-ready forcing store may legitimately hold forcing for more than one
    HRU set of the SAME domain (e.g. a lumped ``hru=1`` forcing beside a 12-band
    elevation ``hru=12`` forcing HYPE built). Each model must read only the
    forcing remapped onto ITS catchment, or ``xr.open_mfdataset`` collapses them
    into a ``conflicting dimension sizes`` error, or worse, reads weather
    averaged over the wrong HRUs (issue #339).

    Given the run's *discretization* (``config.domain.discretization`` /
    ``SUB_GRID_DISCRETIZATION``) and, when known, the *geometry* of its
    catchment shapefile (:func:`hru_geometry`), the order of preference is:

    1. files whose token is exactly ``{method}-h{count}-{digest8}`` for this run;
    2. with no *geometry*: the files of the single geometry token for this
       method (two or more is ambiguous and raises);
    3. files with the bare ``{method}`` token (written before geometry tokens);
       with a *geometry*, their ``hru`` size must equal its HRU count;
    4. every file, when NO file carries any token (a store written before #379).

    Raises :class:`~symfluence.core.exceptions.FileOperationError` when the
    store is tokened but holds no forcing for this run: returning another HRU
    set's files would feed the model the wrong weather.
    """
    files = [forcing_files] if isinstance(forcing_files, (str, Path)) else list(forcing_files)
    files = [Path(f) for f in files]
    if not discretization:
        return files

    by_token: Dict[Optional[str], List[Path]] = {}
    for f in files:
        by_token.setdefault(discretization_key_from_name(f.name), []).append(f)
    tokens = sorted(t for t in by_token if t)
    if not tokens:
        return files  # legacy untokened store

    method = discretization_token(discretization)
    wanted = forcing_token(discretization, geometry) if geometry is not None else None
    if wanted is not None and wanted in by_token:
        return _log_selection(by_token[wanted], files, wanted)

    same_method = [t for t in tokens if split_token(t)[0] == method and split_token(t)[1] is not None]
    if same_method:
        if wanted is not None:
            raise FileOperationError(
                f"The forcing store holds '{method}' forcing remapped onto other HRU sets "
                f"({', '.join(same_method)}) but none onto this run's catchment ({wanted}). "
                "The catchment shapefile changed since forcing was remapped; re-run forcing "
                "remapping for it instead of reading forcing averaged over different HRUs."
            )
        if len(same_method) > 1:
            raise FileOperationError(
                f"The forcing store holds '{method}' forcing for several HRU sets "
                f"({', '.join(same_method)}) and the caller did not say which catchment it "
                "runs on. Pass the catchment geometry (see resolve_forcing_files)."
            )
        return _log_selection(by_token[same_method[0]], files, same_method[0])

    if method in by_token:
        legacy = by_token[method]
        if geometry is not None:
            size = _hru_size(legacy[0])
            if size is not None and size != geometry.count:
                raise FileOperationError(
                    f"Forcing files named '_remapped_{method}_' have hru={size}, but this run's "
                    f"catchment has {geometry.count} HRUs. Re-run forcing remapping."
                )
        return _log_selection(legacy, files, method)

    raise FileOperationError(
        f"No remapped forcing file matches discretization '{discretization}' "
        f"(token '{wanted or method}'); the forcing store only holds "
        f"{', '.join(repr(t) for t in tokens)}. Re-run forcing remapping for "
        f"this discretization instead of reading another discretization's forcing."
    )


def _log_selection(matched: List[Path], files: List[Path], token: str) -> List[Path]:
    if len(matched) != len(files):
        logger.debug("Selected %d/%d forcing file(s) for token '%s'", len(matched), len(files), token)
    return matched


def select_hru_forcing_folder(
    files: List[Path],
    discretization: object = None,
    setting: str = 'the forcing discretization',
) -> List[Path]:
    """Select from a per-HRU forcing folder the user pointed at explicitly.

    Readers that map their own HRUs onto forcing hruIds through an explicit
    table (PRMS/GSFLOW ``climate_hru``) cannot check the catchment geometry:
    their HRUs differ from the forcing's by design. With *discretization* set,
    the files are selected by it; without it, a folder holding forcing for
    more than one HRU set raises, naming *setting*, instead of merging
    overlapping hruIds from different sets.
    """
    files = [Path(f) for f in files]
    if discretization:
        return select_forcing_files(files, discretization)
    groups = sorted({discretization_key_from_name(f.name) or 'untokened' for f in files})
    if len(groups) > 1:
        raise FileOperationError(
            f"The per-HRU forcing folder holds forcing for several HRU sets ({', '.join(groups)}); "
            f"set {setting} to choose one.")
    return files


def resolve_forcing_files(
    forcing_files: Union[str, Path, List[Path]],
    discretization: object = None,
    catchment_path: Union[str, Path, None] = None,
    hru_id_field: Optional[str] = None,
    pattern: str = '*.nc',
) -> List[Path]:
    """Select the forcing files for a run from a directory or a file list.

    The one entry point model readers should use. A directory is globbed with
    *pattern* (sorted). The catchment geometry is computed from
    *catchment_path* when that file exists; otherwise selection falls back to
    the discretization alone (see :func:`select_forcing_files`).
    """
    if isinstance(forcing_files, (str, Path)) and Path(forcing_files).is_dir():
        files = sorted(Path(forcing_files).glob(pattern))
    elif isinstance(forcing_files, (str, Path)):
        files = [Path(forcing_files)]
    else:
        files = [Path(f) for f in forcing_files]
    geometry = None
    if discretization and catchment_path is not None and Path(catchment_path).exists():
        try:
            geometry = hru_geometry(catchment_path, hru_id_field)
        except (OSError, ValueError, RuntimeError) as e:  # RuntimeError: pyogrio DataSourceError
            logger.warning("Could not read catchment %s to select forcing by HRU geometry: %s",
                           catchment_path, e)
    return select_forcing_files(files, discretization, geometry)
