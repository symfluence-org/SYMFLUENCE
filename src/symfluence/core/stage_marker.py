# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Configuration-hash stage markers for SYMFLUENCE workflow invalidation.

Each workflow stage writes a JSON marker after successful completion.
The marker includes a SHA-256 hash of the config sections relevant to
that stage.  On subsequent runs the orchestrator compares the stored
hash with the current config — if they differ the stage is re-executed
even when its output files already exist.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

try:
    from symfluence.symfluence_version import __version__
except ImportError:
    try:
        from importlib.metadata import PackageNotFoundError, version

        __version__ = version("symfluence")
    except (ImportError, PackageNotFoundError):
        __version__ = "0.0.0"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage → config-section mapping
# ---------------------------------------------------------------------------

STAGE_CONFIG_SECTIONS: Dict[str, List[str]] = {
    "setup_project": ["domain"],
    "create_pour_point": ["domain"],
    "acquire_attributes": ["domain", "data"],
    "define_domain": ["domain"],
    "discretize_domain": ["domain"],
    "process_observed_data": ["evaluation", "data"],
    "acquire_forcings": ["forcing", "domain"],
    "run_model_agnostic_preprocessing": ["forcing", "domain"],
    # Model-agnostic product: the store links the basin-averaged forcing and
    # builds domain-level observation/attribute NetCDFs. It must NOT hash the
    # ``model`` section — doing so makes every model in a shared domain see the
    # other models' store as stale and rebuild shared artefacts underneath them.
    "build_model_ready_store": ["data", "forcing"],
    "preprocess_models": ["model", "forcing", "domain"],
    "run_models": ["model", "domain"],
    "postprocess_results": ["model", "evaluation"],
    "calibrate_model": ["optimization", "model", "evaluation", "domain"],
    "run_benchmarking": ["evaluation"],
    "run_decision_analysis": ["evaluation", "model"],
    "run_sensitivity_analysis": ["evaluation", "optimization"],
}

# Stages whose products live in the shared, domain-level tree (shapefiles,
# data/forcing/basin_averaged_data, data/model_ready, data/observations).
# Every experiment in a domain consumes the *same* files, so these stages must
# hash to the same value across experiments — otherwise concurrent workflows on
# one domain each judge the shared product stale and regenerate it while the
# others are reading it.
DOMAIN_SHARED_STAGES: frozenset = frozenset({
    "setup_project",
    "create_pour_point",
    "acquire_attributes",
    "define_domain",
    "discretize_domain",
    "process_observed_data",
    "acquire_forcings",
    "run_model_agnostic_preprocessing",
    "build_model_ready_store",
})

# Config fields scoped to a single experiment. They select *where a run writes
# its own outputs*, never the content of a domain-shared product, so they are
# excluded from the hash of every stage in DOMAIN_SHARED_STAGES.
EXPERIMENT_SCOPED_FIELDS: Dict[str, List[str]] = {
    "domain": ["experiment_id"],
}

# Experiment sub-periods: they decide how a run is spun up, calibrated and
# scored, never the geometry, attributes or forcing a domain-shared stage
# produces (forcing follows the experiment *window*, time_start/time_end).
_PERIOD_FIELDS: List[str] = [
    "calibration_period",
    "calibration_start_date",
    "calibration_end_date",
    "evaluation_period",
    "spinup_period",
]
_WINDOW_FIELDS: List[str] = ["time_start", "time_end"]
# Geospatial-attribute sources: drive attribute acquisition and the
# delineation/discretization built from them, not the project skeleton or the
# pour point.
_ATTRIBUTE_SOURCE_FIELDS: List[str] = [
    "dem_source",
    "download_dem",
    "download_soil",
    "download_landcover",
    "land_class_source",
    "land_class_name",
    "soilgrids_layer",
    "attribute_profile",
]
# Observation-selection fields of the ``data`` section (they feed
# process_observed_data / the model-ready store, not attribute acquisition).
_OBSERVATION_DATA_FIELDS: List[str] = [
    "additional_observations",
    "streamflow_data_provider",
    "usgs_site_code",
    "download_usgs_data",
    "streamflow_station_id",
    "download_ismn",
]

# Per-stage fields left out of the hash, on top of EXPERIMENT_SCOPED_FIELDS for
# domain-shared stages. Deliberately an *exclusion* list: anything not named
# here is hashed, so an unlisted field can at worst cause an unneeded re-run,
# never a skipped one.
STAGE_EXCLUDED_FIELDS: Dict[str, Dict[str, List[str]]] = {
    "setup_project": {"domain": _PERIOD_FIELDS + _WINDOW_FIELDS + _ATTRIBUTE_SOURCE_FIELDS},
    "create_pour_point": {"domain": _PERIOD_FIELDS + _WINDOW_FIELDS + _ATTRIBUTE_SOURCE_FIELDS},
    "acquire_attributes": {
        "domain": _PERIOD_FIELDS + _WINDOW_FIELDS,
        "data": _OBSERVATION_DATA_FIELDS,
    },
    "define_domain": {"domain": _PERIOD_FIELDS + _WINDOW_FIELDS},
    "discretize_domain": {"domain": _PERIOD_FIELDS + _WINDOW_FIELDS},
    "process_observed_data": {"evaluation": ["analyses"]},
    "acquire_forcings": {"domain": _PERIOD_FIELDS},
    "run_model_agnostic_preprocessing": {"domain": _PERIOD_FIELDS},
    "postprocess_results": {"evaluation": ["analyses"]},
    "calibrate_model": {"evaluation": ["analyses"]},
}

# Hash-scheme version recorded in every marker. Markers written before the
# per-stage exclusions existed carry no version and are compared against the
# legacy whole-section hash, so upgrading never forces a re-run by itself.
HASH_VERSION = 2

MARKER_DIR_NAME = "stage_markers"

# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def compute_config_hash(
    config: Any,
    sections: Sequence[str],
    exclude: Optional[Mapping[str, Sequence[str]]] = None,
) -> str:
    """
    Compute a deterministic SHA-256 hex digest over the given config sections.

    Sections are sorted alphabetically before serialization so the hash is
    independent of the order they are listed in ``STAGE_CONFIG_SECTIONS``.

    Args:
        config: Typed ``SymfluenceConfig``.
        sections: Config sections to hash.
        exclude: Optional ``{section: [field, ...]}`` of fields to drop before
            hashing (see ``EXPERIMENT_SCOPED_FIELDS``).
    """
    combined: Dict[str, Any] = {}
    for section in sorted(sections):
        section_obj = getattr(config, section, None)
        if section_obj is None:
            continue
        dumped = section_obj.model_dump(by_alias=False)
        for field in (exclude or {}).get(section, []):
            dumped.pop(field, None)
        combined[section] = dumped

    payload = json.dumps(combined, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stage_exclusions(stage_name: str) -> Dict[str, List[str]]:
    exclude: Dict[str, List[str]] = {}
    if stage_name in DOMAIN_SHARED_STAGES:
        for section, fields in EXPERIMENT_SCOPED_FIELDS.items():
            exclude.setdefault(section, []).extend(fields)
    for section, fields in STAGE_EXCLUDED_FIELDS.get(stage_name, {}).items():
        exclude.setdefault(section, []).extend(fields)
    return exclude


def compute_stage_hash(config: Any, stage_name: str) -> str:
    """
    Compute the invalidation hash for *stage_name*.

    Hashes the stage's config sections minus the fields the stage does not
    depend on: experiment-scoped fields for domain-shared stages (so two
    experiments sharing a domain agree on every shared product) and the
    per-stage ``STAGE_EXCLUDED_FIELDS`` (e.g. changing CALIBRATION_PERIOD does
    not invalidate delineation or forcing).  Returns ``""`` for unknown stages
    (no sections mapped → the orchestrator falls back to output-existence checks).
    """
    sections = STAGE_CONFIG_SECTIONS.get(stage_name, [])
    if not sections:
        return ""
    return compute_config_hash(config, sections, exclude=_stage_exclusions(stage_name))


def compute_legacy_stage_hash(config: Any, stage_name: str) -> str:
    """Stage hash under the pre-``HASH_VERSION`` scheme (whole sections).

    Only used to honour markers written by older releases.
    """
    sections = STAGE_CONFIG_SECTIONS.get(stage_name, [])
    if not sections:
        return ""
    exclude = EXPERIMENT_SCOPED_FIELDS if stage_name in DOMAIN_SHARED_STAGES else None
    return compute_config_hash(config, sections, exclude=exclude)


def is_experiment_scoped(stage_name: str) -> bool:
    """True for stages whose product belongs to one experiment (per-experiment marker)."""
    return stage_name in STAGE_CONFIG_SECTIONS and stage_name not in DOMAIN_SHARED_STAGES


# ---------------------------------------------------------------------------
# Marker dataclass
# ---------------------------------------------------------------------------


@dataclass
class StageMarker:
    """Metadata written after a stage completes successfully."""

    stage: str
    completed_utc: str
    config_hash: str
    symfluence_version: str
    git_commit: Optional[str] = None
    # None for markers written before HASH_VERSION existed (legacy scheme).
    hash_version: Optional[int] = None
    experiment_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _marker_dir(project_dir: Path) -> Path:
    return project_dir / ".symfluence" / MARKER_DIR_NAME


def _safe_component(value: str) -> str:
    """Make an experiment id safe to use as a single path component."""
    cleaned = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(value))
    return cleaned.strip(".") or "_"


def _marker_path(
    project_dir: Path,
    stage_name: str,
    experiment_id: Optional[str] = None,
) -> Path:
    """Marker location.

    Domain-shared stages (and callers that pass no experiment id) use the flat
    ``stage_markers/{stage}.json``. Experiment-scoped stages get one marker per
    experiment, ``stage_markers/experiments/{experiment_id}/{stage}.json``, so
    experiments sharing a domain no longer overwrite each other's markers.
    """
    if experiment_id and is_experiment_scoped(stage_name):
        return (
            _marker_dir(project_dir) / "experiments"
            / _safe_component(experiment_id) / f"{stage_name}.json"
        )
    return _marker_dir(project_dir) / f"{stage_name}.json"


def _current_git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _read_marker_file(path: Path, stage_name: str) -> Optional[StageMarker]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return StageMarker(
            stage=data["stage"],
            completed_utc=data["completed_utc"],
            config_hash=data["config_hash"],
            symfluence_version=data["symfluence_version"],
            git_commit=data.get("git_commit"),
            hash_version=data.get("hash_version"),
            experiment_id=data.get("experiment_id"),
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("Corrupt stage marker for '%s' — will re-execute", stage_name)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_marker(
    project_dir: Path,
    stage_name: str,
    config_hash: str,
    git_commit: Optional[str] = None,
    experiment_id: Optional[str] = None,
) -> Path:
    """Write a JSON marker for *stage_name* and return its path."""
    marker = StageMarker(
        stage=stage_name,
        completed_utc=datetime.now(timezone.utc).isoformat(),
        config_hash=config_hash,
        symfluence_version=__version__,
        git_commit=git_commit or _current_git_commit(),
        hash_version=HASH_VERSION,
        experiment_id=experiment_id if is_experiment_scoped(stage_name) else None,
    )
    path = _marker_path(project_dir, stage_name, experiment_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(marker), indent=2), encoding="utf-8")
    return path


def read_marker(
    project_dir: Path,
    stage_name: str,
    experiment_id: Optional[str] = None,
) -> Optional[StageMarker]:
    """Read a stage marker, returning ``None`` if missing or corrupt.

    For an experiment-scoped stage with an ``experiment_id``, the
    per-experiment marker is preferred; the legacy flat marker (written by
    older releases, shared by all experiments) is the fallback.
    """
    marker = _read_marker_file(
        _marker_path(project_dir, stage_name, experiment_id), stage_name
    )
    if marker is None and experiment_id and is_experiment_scoped(stage_name):
        marker = _read_marker_file(_marker_path(project_dir, stage_name), stage_name)
    return marker


def is_stage_current(
    project_dir: Path,
    stage_name: str,
    current_hash: str,
    *,
    experiment_id: Optional[str] = None,
    legacy_hash: Optional[str] = None,
) -> bool:
    """Return ``True`` only when a marker exists and its hash matches.

    A marker from before ``HASH_VERSION`` (no ``hash_version`` field) is
    matched against ``legacy_hash`` when one is supplied, so an upgrade does
    not invalidate every completed stage on its own.
    """
    marker = read_marker(project_dir, stage_name, experiment_id)
    if marker is None:
        return False
    if marker.hash_version is None and legacy_hash is not None:
        return marker.config_hash == legacy_hash
    return marker.config_hash == current_hash


def clear_markers(
    project_dir: Path,
    stage_names: Optional[Sequence[str]] = None,
    experiment_id: Optional[str] = None,
) -> None:
    """Remove markers.

    With ``stage_names=None`` every marker (all experiments) is removed.
    Otherwise only the named stages' markers go: the flat marker, plus the
    per-experiment marker of ``experiment_id`` for experiment-scoped stages
    (other experiments' markers are left alone).
    """
    d = _marker_dir(project_dir)
    if not d.exists():
        return
    if stage_names is None:
        for f in d.rglob("*.json"):
            f.unlink()
        return
    for name in stage_names:
        paths = {_marker_path(project_dir, name)}
        if experiment_id:
            paths.add(_marker_path(project_dir, name, experiment_id))
        for p in paths:
            if p.exists():
                p.unlink()


# ---------------------------------------------------------------------------
# Config-level convenience wrappers (used by the orchestrator)
# ---------------------------------------------------------------------------


def _experiment_id(config: Any) -> Optional[str]:
    domain = getattr(config, "domain", None)
    value = getattr(domain, "experiment_id", None)
    return str(value) if value else None


def stage_is_current(config: Any, project_dir: Path, stage_name: str) -> bool:
    """Marker check for *stage_name* under *config* (unknown stages → True)."""
    if not STAGE_CONFIG_SECTIONS.get(stage_name):
        return True
    return is_stage_current(
        project_dir,
        stage_name,
        compute_stage_hash(config, stage_name),
        experiment_id=_experiment_id(config),
        legacy_hash=compute_legacy_stage_hash(config, stage_name),
    )


def mark_stage_complete(config: Any, project_dir: Path, stage_name: str) -> Optional[Path]:
    """Write *stage_name*'s marker for *config* (no-op for unknown stages)."""
    if not STAGE_CONFIG_SECTIONS.get(stage_name):
        return None
    return write_marker(
        project_dir,
        stage_name,
        compute_stage_hash(config, stage_name),
        experiment_id=_experiment_id(config),
    )


def clear_stage_markers(config: Any, project_dir: Path, stage_names: Sequence[str]) -> None:
    """Remove the markers of *stage_names* for *config*'s experiment."""
    clear_markers(project_dir, list(stage_names), experiment_id=_experiment_id(config))
