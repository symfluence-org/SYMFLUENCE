# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Discretization namespacing for model-ready forcing filenames (issue #339).

The naming contract that lets ONE model-ready forcing store hold several spatial
discretizations of the same domain side by side: a remapped forcing is written as
``{domain}_{forcing}_remapped_{token}_{datetag}.nc``, where ``{token}`` identifies
the discretization it was remapped onto. Writers (the remapping/resampling
builders) stamp the token; readers (every model's forcing load path) select by it.

This lives in ``core`` because both ends of that contract sit in different layers —
``data`` writes the names, ``core.modeling`` and the model adapters read them — and
the convention itself is pure string handling with no I/O and no upper-layer
dependency. Keeping it here is what lets the core forcing processor scope its read
without importing the data layer.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Union

from symfluence.core.exceptions import FileOperationError

logger = logging.getLogger(__name__)

# A namespaced remapped forcing name is
#   {domain}_{forcing}_remapped_{token}_{datetag}.nc  (or ..._remapped_{token}.nc)
# where {token} is the letter-initial discretization token written by
# discretization_token() (lumped, elevation, grus, ...). Legacy date-tag-only
# names ('..._remapped_2002-01-01-...') start with a digit and carry no token.
_REMAP_TOKEN_RE = re.compile(r'_remapped_([a-z][a-z0-9-]*?)(?:_|\.nc$)')


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


def select_forcing_files(
    forcing_files: Union[str, Path, List[Path]],
    discretization: object = None,
) -> List[Path]:
    """Pick the forcing files matching a run's spatial discretization.

    A model-ready forcing store may legitimately hold more than one
    discretization of the SAME domain (e.g. a lumped ``hru=1`` forcing the lumped
    models need beside a 12-band elevation ``hru=12`` forcing HYPE built). Each
    model must read only the forcing matching ITS own catchment discretization,
    or ``xr.open_mfdataset`` collapses them into a ``conflicting dimension sizes``
    error (issue #339).

    Given the run's *discretization* (``config.domain.discretization`` /
    ``SUB_GRID_DISCRETIZATION``), return only the files whose namespaced filename
    carries that token. Falls back to the full list unchanged when:

    - *discretization* is falsy (caller did not scope the read), or
    - NO file carries ANY discretization token — a legacy store written before
      namespacing, whose names predate this fix.

    The fallback is what lets stores predating the fix keep working without
    regeneration. If the store IS namespaced but holds only OTHER
    discretizations, there is no safe fallback: returning those files would
    feed the model forcing remapped onto a different set of HRUs. That case
    raises :class:`~symfluence.core.exceptions.FileOperationError`.
    """
    files = [forcing_files] if isinstance(forcing_files, (str, Path)) else list(forcing_files)
    files = [Path(f) for f in files]
    if not discretization:
        return files
    token = discretization_token(discretization)
    matched = [f for f in files if forcing_name_matches_discretization(f.name, token)]
    if not matched:
        tokens = sorted({k for k in (discretization_key_from_name(f.name) for f in files) if k})
        if not tokens:
            return files  # legacy untokened store
        raise FileOperationError(
            f"No remapped forcing file matches discretization '{discretization}' "
            f"(token '{token}'); the forcing store only holds "
            f"{', '.join(repr(t) for t in tokens)}. Re-run forcing remapping for "
            f"this discretization instead of reading another discretization's forcing."
        )
    if len(matched) != len(files):
        logger.debug(
            "Selected %d/%d forcing file(s) for discretization '%s' (token '%s')",
            len(matched), len(files), discretization, token,
        )
    return matched
