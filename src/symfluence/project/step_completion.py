# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Output-existence checks for workflow steps.

Each check answers "has this step's product been written for this config?"
using the same path conventions the writing code uses, so ``workflow run`` can
skip a finished step and ``workflow status`` reports it as complete. They are
deliberately read-only and cheap (no manager construction, no model imports).

The checks are the *output* half of the skip decision; the other half is the
stage marker (``core/stage_marker.py``), which is written only after a step
succeeded and records the config it ran under. A step is complete only when
both agree.

Where a step has no guaranteed product (calibration disabled, an analysis not
listed in ``ANALYSES``), the check returns True and the marker alone decides.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional

from symfluence.core.mixins.project import simulation_dir

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


def _is_default(value: object) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in ('', 'default'))


def configured_models(config: 'SymfluenceConfig') -> List[str]:
    """Upper-cased HYDROLOGICAL_MODEL entries (comma-separated or list)."""
    raw = config.model.hydrological_model or ''
    items = raw if isinstance(raw, (list, tuple)) else str(raw).split(',')
    return [str(m).strip().upper() for m in items if str(m).strip()]


def _child_casefold(parent: Path, name: str) -> Optional[Path]:
    """Return ``parent/name`` matched case-insensitively, or None."""
    exact = parent / name
    if exact.exists():
        return exact
    if not parent.is_dir():
        return None
    wanted = name.casefold()
    for child in parent.iterdir():
        if child.name.casefold() == wanted:
            return child
    return None


def _has_files(directory: Optional[Path], *, ignore: Iterable[str] = ()) -> bool:
    """True when *directory* holds at least one regular file (recursively),
    skipping files/dirs whose name contains any of *ignore* (case-insensitive)."""
    if directory is None or not directory.is_dir():
        return False
    ignore_cf = [i.casefold() for i in ignore]
    for path in directory.rglob('*'):
        rel_parts = [p.casefold() for p in path.relative_to(directory).parts]
        if any(token in part for part in rel_parts for token in ignore_cf):
            continue
        if path.is_file():
            return True
    return False


# ---------------------------------------------------------------------------
# Geospatial
# ---------------------------------------------------------------------------


def river_basins_exist(config: 'SymfluenceConfig', project_dir: Path) -> bool:
    """define_domain: ``river_basins/{domain}_riverBasins_{method}.shp`` (or the
    configured RIVER_BASINS_PATH / RIVER_BASINS_NAME)."""
    paths = config.paths
    base = (project_dir / 'shapefiles' / 'river_basins'
            if _is_default(paths.river_basins_path) else Path(paths.river_basins_path))
    name = (f"{config.domain.name}_riverBasins_{config.domain.definition_method}.shp"
            if _is_default(paths.river_basins_name) else paths.river_basins_name)
    return (base / name).exists()


def discretization_exists(config: 'SymfluenceConfig', project_dir: Path) -> bool:
    """discretize_domain: the HRU shapefile discretization writes.

    Mirrors ``PathResolverMixin._get_catchment_file_path``: the canonical
    ``catchment/{definition_method}/{experiment_id}/{domain}_HRUs_{disc}.shp``,
    or the legacy flat ``catchment/`` file, or an explicit CATCHMENT_PATH /
    CATCHMENT_SHP_NAME. The suffix is matched case-insensitively because some
    discretizers upper-case it (GRUs -> GRUS).
    """
    paths = config.paths
    domain = config.domain
    if _is_default(paths.catchment_name):
        suffix = str(domain.discretization or '').replace(',', '_')
        name = f"{domain.name}_HRUs_{suffix}.shp"
    else:
        name = paths.catchment_name

    if not _is_default(paths.catchment_path):
        return _child_casefold(Path(paths.catchment_path), name) is not None

    catchment_dir = project_dir / 'shapefiles' / 'catchment'
    candidates = [
        catchment_dir / str(domain.definition_method) / str(domain.experiment_id),
        catchment_dir,
    ]
    return any(_child_casefold(d, name) is not None for d in candidates)


# ---------------------------------------------------------------------------
# Model-specific
# ---------------------------------------------------------------------------


def model_settings_exist(config: 'SymfluenceConfig', project_dir: Path) -> bool:
    """preprocess_models: every configured model has a non-empty
    ``settings/{MODEL}`` directory (``BaseModelPreProcessor.setup_dir``)."""
    models = configured_models(config)
    if not models:
        return False
    settings_dir = project_dir / 'settings'
    return all(_has_files(_child_casefold(settings_dir, m)) for m in models)


def model_outputs_exist(config: 'SymfluenceConfig', project_dir: Path) -> bool:
    """run_models: every configured model wrote something (other than logs)
    to ``simulations/{experiment_id}/{MODEL}`` (``BaseModelRunner._get_output_dir``)."""
    models = configured_models(config)
    if not models:
        return False
    exp_dir = simulation_dir(project_dir, str(config.domain.experiment_id), '')
    return all(
        _has_files(_child_casefold(exp_dir, m), ignore=('log',))
        for m in models
    )


def postprocessed_results_exist(config: 'SymfluenceConfig', project_dir: Path) -> bool:
    """postprocess_results: ``results/{experiment_id}_results.csv``
    (``BaseModelPostProcessor.save_streamflow_to_results``)."""
    return (project_dir / 'results' / f"{config.domain.experiment_id}_results.csv").exists()


# ---------------------------------------------------------------------------
# Calibration / analyses
# ---------------------------------------------------------------------------


def calibration_enabled(config: 'SymfluenceConfig') -> bool:
    """Same gate ``OptimizationManager.calibrate_model`` applies."""
    methods = config.optimization.methods or []
    if isinstance(methods, str):
        methods = [m.strip() for m in methods.split(',')]
    return 'iteration' in [str(m).strip().lower() for m in methods]


def calibration_results_exist(config: 'SymfluenceConfig', project_dir: Path) -> bool:
    """calibrate_model: results under ``optimization/{MODEL}/{alg}_{experiment_id}/``.

    Calibration disabled → True (nothing to produce; the marker decides).
    """
    if not calibration_enabled(config):
        return True
    from symfluence.optimization.optimization_manager import find_calibration_results

    return find_calibration_results(
        project_dir,
        str(config.domain.experiment_id),
        str(config.optimization.algorithm or ''),
    ) is not None


def _analysis_requested(config: 'SymfluenceConfig', token: str) -> bool:
    analyses = config.evaluation.analyses or []
    return token in [str(a).strip().lower() for a in analyses]


def analysis_output_exists(config: 'SymfluenceConfig', token: str, output: Path) -> bool:
    """Analyses: the output file when ``token`` is in ANALYSES, else True."""
    if not _analysis_requested(config, token):
        return True
    return output.exists()
