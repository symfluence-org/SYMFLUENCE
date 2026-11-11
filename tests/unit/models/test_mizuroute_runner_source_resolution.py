# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""mizuRoute resolves its source runoff file from the registered declarations.

``MizuRouteRunner.fix_time_precision`` used to carry a 65-line if/elif chain
duplicating every model's runoff directory and filename -- the *fourth* copy of
that table, after ``runoff_loader``, the mizuRoute control writer, and the
schema each model registers. It now delegates to ``resolve_runoff_file``.

These tests pin the two things the collapse had to preserve: the source-model
precedence, and the filename each source resolves to.

CHANGED: HYPE left both. It is not a routable source — its runoff declaration
named ``{experiment_id}_timestep.nc``, which nothing in the HYPE adapter
writes, and its ``cout`` is already routed discharge at subbasin outlets, so
converting it would route it twice. With the declaration gone,
``resolve_runoff_file('HYPE')`` raises instead of resolving a path, and HYPE had
to leave the precedence too: otherwise a ``SUMMA,HYPE`` run would select HYPE
and fail where it previously routed SUMMA.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from symfluence.core.exceptions import ConfigurationError
from symfluence.core.modeling.utilities.runoff_loader import resolve_runoff_file

pytestmark = [pytest.mark.unit]

EXPERIMENT_ID = 'mizu_src_001'
DOMAIN_NAME = 'testdom'

def _resolve_source(active_models) -> str:
    """Drive the REAL selection inside ``MizuRouteRunner.fix_time_precision``.

    This deliberately does not reimplement the precedence. An earlier version of
    this file copied the ``next((m for m in _PRECEDENCE ...))`` expression out of
    ``runner.py`` and asserted against the copy, so reversing the production
    tuple left every case green — the file proved only that it agreed with
    itself. Instead: build the runner without ``__init__`` (its constructor
    needs a full project layout), feed it the model list, and observe which
    source ``resolve_runoff_file`` is asked for.
    """
    # hydrological_model is a read-only property reading
    # config.model.hydrological_model, so the config has to be attribute-shaped:
    # a flat dict does not reach it (the property passes no dict_key, so a dict
    # config silently yields ''). Going through the property is the point — the
    # splitting, upper-casing and 'DEFAULT' filtering it feeds are under test.
    from types import SimpleNamespace

    from symfluence.models.mizuroute.runner import MizuRouteRunner

    runner = MizuRouteRunner.__new__(MizuRouteRunner)
    runner.logger = logging.getLogger('test.mizuroute.source')
    runner.config = SimpleNamespace(
        model=SimpleNamespace(
            hydrological_model=','.join(active_models),
            mizuroute=SimpleNamespace(from_model=''),
        )
    )
    runner.project_dir = Path('/nonexistent')
    runner.experiment_id = EXPERIMENT_ID
    runner.domain_name = DOMAIN_NAME

    seen: list[str] = []

    def _spy(source_model, **kwargs):
        seen.append(source_model)
        return None  # stop before any file work; selection is what we assert

    import symfluence.models.mizuroute.runner as runner_module
    original = runner_module.resolve_runoff_file
    runner_module.resolve_runoff_file = _spy
    try:
        runner.fix_time_precision()
    finally:
        runner_module.resolve_runoff_file = original

    assert seen, "fix_time_precision never attempted to resolve a runoff file"
    return seen[0]


@pytest.mark.parametrize(
    "active_models,expected",
    [
        (['SUMMA'], 'SUMMA'),
        ([], 'SUMMA'),
        (['MIZUROUTE'], 'SUMMA'),
        (['FUSE'], 'FUSE'),
        (['GR'], 'GR'),
        (['NGEN'], 'NGEN'),
        # FUSE outranks everything, as in the original chain.
        (['FUSE', 'GR', 'HYPE', 'NGEN', 'SUMMA'], 'FUSE'),
        (['GR', 'HYPE', 'NGEN'], 'GR'),
        (['HYPE', 'NGEN'], 'NGEN'),
        # A SUMMA+FUSE run routes FUSE, not SUMMA.
        (['FUSE', 'SUMMA'], 'FUSE'),
        # HYPE is not a routable source, so it never wins the selection —
        # a SUMMA+HYPE run routes SUMMA, and a HYPE-only run falls back.
        (['HYPE'], 'SUMMA'),
        (['HYPE', 'SUMMA'], 'SUMMA'),
    ],
)
def test_source_model_precedence_is_preserved(active_models, expected):
    assert _resolve_source(active_models) == expected


def test_hype_is_not_a_resolvable_source():
    """Asking for HYPE runoff fails immediately and says why."""
    with pytest.raises(ConfigurationError, match='HYPE'):
        resolve_runoff_file(
            source_model='HYPE',
            project_dir=Path('.'),
            experiment_id=EXPERIMENT_ID,
            domain_name=DOMAIN_NAME,
        )


# A second, wrongly-named ``.nc`` written alongside the expected one in every
# filename test below, so a mis-built ``output_file_pattern`` cannot be rescued
# by anything that picks "some .nc in the directory" (``resolve_runoff_file``
# used to have exactly that glob fallback; it has been removed).
_DECOY_NC = 'aaa_decoy_not_the_runoff_file.nc'


def _assert_no_glob_fallback(caplog):
    fallbacks = [r.message for r in caplog.records
                 if 'fallback output file' in r.getMessage()]
    assert not fallbacks, (
        f"resolve_runoff_file did not build the expected name and was rescued "
        f"by the glob fallback: {fallbacks}"
    )


@pytest.mark.parametrize(
    "source_model,filename",
    [
        ('SUMMA', f"{EXPERIMENT_ID}_timestep.nc"),
        ('GR', f"{DOMAIN_NAME}_{EXPERIMENT_ID}_runs_def.nc"),
        ('NGEN', f"{EXPERIMENT_ID}_runoff.nc"),
    ],
)
def test_each_source_resolves_the_filename_the_chain_hardcoded(
    tmp_path: Path, caplog, source_model, filename
):
    """The names the deleted if/elif chain built, now served by the schema."""
    out_dir = tmp_path / 'simulations' / EXPERIMENT_ID / source_model
    out_dir.mkdir(parents=True)
    (out_dir / filename).write_bytes(b'')
    (out_dir / _DECOY_NC).write_bytes(b'')

    with caplog.at_level(logging.INFO):
        resolved = resolve_runoff_file(
            source_model=source_model,
            project_dir=tmp_path,
            experiment_id=EXPERIMENT_ID,
            domain_name=DOMAIN_NAME,
        )

    assert resolved is not None
    assert resolved.name == filename
    _assert_no_glob_fallback(caplog)


def test_fuse_file_id_is_hashed_past_six_chars(tmp_path: Path, caplog):
    """FUSE's Fortran-driven 6-char truncation survives the collapse."""
    import hashlib

    file_id = hashlib.md5(
        EXPERIMENT_ID.encode(), usedforsecurity=False
    ).hexdigest()[:6]
    expected = f"{DOMAIN_NAME}_{file_id}_runs_def.nc"

    out_dir = tmp_path / 'simulations' / EXPERIMENT_ID / 'FUSE'
    out_dir.mkdir(parents=True)
    (out_dir / expected).write_bytes(b'')
    # The un-hashed name FUSE would produce without the truncation, so dropping
    # the hashing resolves to *this* rather than being handed the hashed file
    # back by the glob fallback.
    (out_dir / f"{DOMAIN_NAME}_{EXPERIMENT_ID}_runs_def.nc").write_bytes(b'')
    (out_dir / _DECOY_NC).write_bytes(b'')

    with caplog.at_level(logging.INFO):
        resolved = resolve_runoff_file(
            source_model='FUSE',
            project_dir=tmp_path,
            experiment_id=EXPERIMENT_ID,
            domain_name=DOMAIN_NAME,
        )

    assert resolved is not None
    assert resolved.name == expected
    _assert_no_glob_fallback(caplog)


def test_no_glob_fallback_to_an_arbitrary_netcdf(tmp_path: Path, caplog):
    """A differently named ``.nc`` in the output directory is not the runoff file.

    ``resolve_runoff_file`` used to return ``glob('*.nc')[0]`` when the name
    it built did not exist — whatever sorted first, e.g. a restart file or
    another experiment's output — and the time-precision fix then rewrote that
    file in place. A missing expected file now resolves to None, naming the
    expected path.
    """
    out_dir = tmp_path / 'simulations' / EXPERIMENT_ID / 'NGEN'
    out_dir.mkdir(parents=True)
    (out_dir / 'something_else_entirely.nc').write_bytes(b'')

    with caplog.at_level(logging.WARNING):
        resolved = resolve_runoff_file(
            source_model='NGEN',
            project_dir=tmp_path,
            experiment_id=EXPERIMENT_ID,
            domain_name=DOMAIN_NAME,
        )

    assert resolved is None
    assert any(f"{EXPERIMENT_ID}_runoff.nc" in r.getMessage() for r in caplog.records)


def test_missing_output_resolves_to_none_not_a_guess(tmp_path: Path, caplog):
    """Nothing to route resolves to None so the runner can report it."""
    with caplog.at_level(logging.WARNING):
        resolved = resolve_runoff_file(
            source_model='NGEN',
            project_dir=tmp_path,
            experiment_id=EXPERIMENT_ID,
            domain_name=DOMAIN_NAME,
        )

    assert resolved is None
