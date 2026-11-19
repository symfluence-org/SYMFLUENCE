# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Provenance tracking for SYMFLUENCE workflow runs.

Captures framework version, git state, dependency versions, platform details,
and per-step timing into a self-documenting run manifest (JSON).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import secrets
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

try:
    from symfluence.symfluence_version import __version__
except ImportError:
    __version__ = "0+unknown"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _git_info(repo_dir: Optional[Path] = None) -> Dict[str, Optional[str]]:
    """Return git commit, branch, and dirty status for *repo_dir*."""
    cwd = str(repo_dir) if repo_dir else None
    info: Dict[str, Optional[str]] = {
        "commit": None,
        "commit_short": None,
        "branch": None,
        "dirty": None,
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
        info["commit"] = commit
        info["commit_short"] = commit[:8]
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
        info["dirty"] = bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    return info


def _dependency_versions() -> Dict[str, str]:
    """Return installed versions of key scientific dependencies."""
    packages = [
        "numpy", "xarray", "pandas", "scipy", "netCDF4", "geopandas",
        "shapely", "rasterio", "pyproj", "dask", "jax", "torch",
    ]
    versions: Dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return versions


def _executable_info(exe_path: Path) -> Dict[str, Any]:
    """Return size, SHA-256, mtime, and best-effort version for a binary."""
    info: Dict[str, Any] = {"path": str(exe_path)}
    try:
        stat = exe_path.stat()
        info["size_bytes"] = stat.st_size
        info["mtime"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        sha = hashlib.sha256()
        with open(exe_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                sha.update(chunk)
        info["sha256"] = sha.hexdigest()
    except OSError:
        pass
    # Best-effort version string
    for flag in ("--version", "-v", "-V"):
        try:
            out = subprocess.check_output(
                [str(exe_path), flag],
                stderr=subprocess.STDOUT,
                timeout=5,
            ).decode(errors="replace").strip()
            if out and len(out) < 500:
                info["version_string"] = out.splitlines()[0]
                break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError, OSError, PermissionError):
            continue
    return info


#: Written by ``symfluence binary install`` into each tool's install directory.
SOURCE_RECORD = ".symfluence_source.json"
_SOURCE_KEYS = ("repository", "ref", "commit", "pinned_commit", "pinned_by",
                "tracked_changes", "patched")


def _source_info(exe_path: Path, name: str, max_levels: int = 5) -> Dict[str, Any]:
    """Return the upstream repository and commit *exe_path* was built from.

    Walks up from the executable looking for the per-tool source record the
    installer writes (``<install_dir>/.symfluence_source.json``), or for a
    ``toolchain.json`` manifest (the installs root, or an npm/release bundle)
    with an entry for *name*. Empty when neither is found, e.g. a binary built
    outside SYMFLUENCE or installed before source records existed.
    """
    for parent in list(Path(exe_path).parents)[:max_levels]:
        record = parent / SOURCE_RECORD
        if record.is_file():
            try:
                data = json.loads(record.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}
            return {k: data[k] for k in _SOURCE_KEYS if k in data}
        manifest = parent / "toolchain.json"
        if manifest.is_file():
            try:
                tools = json.loads(manifest.read_text(encoding="utf-8")).get("tools", {})
            except (OSError, ValueError, AttributeError):
                return {}
            entry = tools.get(name.lower()) if isinstance(tools, dict) else None
            if not isinstance(entry, dict):
                return {}
            entry = {**entry, **(entry.get("source") or {})}
            return {k: entry[k] for k in _SOURCE_KEYS if entry.get(k) not in (None, "unknown")}
    return {}


def _geospatial_versions() -> Dict[str, str]:
    """Return GDAL/PROJ/GEOS versions.

    These are recorded separately from ``dependencies`` because they are not
    merely Python packages: they are the C libraries that decide delineation
    geometry. A DEM conditioned by a different GDAL, or reprojected by a
    different PROJ, can shift a basin boundary enough to change discretization
    counts — a difference that is otherwise indistinguishable from a genuine
    cross-platform divergence.
    """
    versions: Dict[str, str] = {}
    try:
        from osgeo import gdal
        versions["gdal"] = gdal.__version__
    except (ImportError, AttributeError):
        pass
    try:
        import pyproj
        versions["proj"] = pyproj.proj_version_str
    except (ImportError, AttributeError):
        pass
    try:
        import shapely
        versions["geos"] = shapely.geos_version_string
    except (ImportError, AttributeError):
        pass
    return versions


def _platform_info() -> Dict[str, str]:
    """Return OS, architecture, hostname, and Python version."""
    return {
        "os": platform.system(),
        "os_version": platform.release(),
        "arch": platform.machine(),
        "hostname": platform.node(),
        "python": platform.python_version(),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_run_id(experiment_id: str, git_short: Optional[str] = None) -> str:
    """Build a human-readable, unique run identifier."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = git_short if git_short else "nogit"
    return f"{experiment_id}_{ts}_{suffix}"


@dataclass
class RunProvenance:
    """Immutable record of everything needed to reproduce a run."""

    run_id: str
    experiment_id: str
    domain_name: str
    config_path: Optional[str]
    symfluence_version: str
    git: Dict[str, Optional[str]]
    platform: Dict[str, str]
    dependencies: Dict[str, str]
    start_utc: str
    end_utc: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    executables: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    inputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    geospatial: Dict[str, str] = field(default_factory=dict)
    resolved_config: Dict[str, Any] = field(default_factory=dict)
    calibration: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "running"
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "domain_name": self.domain_name,
            "config_path": self.config_path,
            "symfluence_version": self.symfluence_version,
            "git": self.git,
            "platform": self.platform,
            "dependencies": self.dependencies,
            "start_utc": self.start_utc,
            "end_utc": self.end_utc,
            "elapsed_seconds": self.elapsed_seconds,
            "steps": list(self.steps),
            "executables": dict(self.executables),
            "inputs": dict(self.inputs),
            "geospatial": dict(self.geospatial),
            "resolved_config": dict(self.resolved_config),
            "calibration": list(self.calibration),
            "status": self.status,
            "errors": list(self.errors),
        }

    def write(self, output_dir: Path) -> Path:
        """Write ``run_manifest.json`` into *output_dir* and return its path."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "run_manifest.json"
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        return path


def capture_provenance(
    experiment_id: str,
    domain_name: str,
    config_path: Optional[str] = None,
    repo_dir: Optional[Path] = None,
) -> RunProvenance:
    """Create a fresh ``RunProvenance`` capturing the current environment."""
    git = _git_info(repo_dir)
    return RunProvenance(
        run_id=make_run_id(experiment_id, git.get("commit_short")),
        experiment_id=experiment_id,
        domain_name=domain_name,
        config_path=str(config_path) if config_path else None,
        symfluence_version=__version__,
        git=git,
        platform=_platform_info(),
        dependencies=_dependency_versions(),
        geospatial=_geospatial_versions(),
        start_utc=datetime.now(timezone.utc).isoformat(),
    )


def record_executable(
    prov: Optional[RunProvenance],
    name: str,
    exe_path: Path,
) -> None:
    """Capture file hash, size, version, and upstream source for *exe_path*.

    ``source`` (repository, pinned and actual commit) is filled from the
    install's source record when one is found; see :func:`_source_info`.

    No-op when *prov* is ``None`` or the executable does not exist.
    """
    if prov is None:
        return
    exe_path = Path(exe_path)
    if exe_path.exists():
        info = _executable_info(exe_path)
        source = _source_info(exe_path, name)
        if source:
            info["source"] = source
        prov.executables[name] = info


def _input_info(path: Path) -> Dict[str, Any]:
    """Return size, SHA-256, and mtime for an input dataset."""
    info: Dict[str, Any] = {"path": str(path)}
    try:
        stat = path.stat()
        info["size_bytes"] = stat.st_size
        info["mtime"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        sha = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                sha.update(chunk)
        info["sha256"] = sha.hexdigest()
    except OSError:
        pass
    return info


def record_input(
    prov: Optional[RunProvenance],
    name: str,
    path: Path,
) -> None:
    """Capture hash, size, and mtime for an input dataset such as the DEM.

    Executables were already hashed; input *data* was not. Without this, a run
    whose delineation shifted because the source DEM changed is
    indistinguishable from one that shifted because the code did — the same
    ambiguity ``symfluence_version``/``code_commit`` were added to resolve for
    code. No-op when *prov* is ``None`` or the file does not exist.
    """
    if prov is None:
        return
    path = Path(path)
    if path.is_file():
        prov.inputs[name] = _input_info(path)


def record_step(
    prov: Optional[RunProvenance],
    name: str,
    duration_s: float,
    status: str = "completed",
    error: Optional[str] = None,
) -> None:
    """Append a step record to *prov*.  No-op when *prov* is ``None``."""
    if prov is None:
        return
    entry: Dict[str, Any] = {
        "name": name,
        "duration_s": round(duration_s, 3),
        "status": status,
    }
    if error is not None:
        entry["error"] = error
    prov.steps.append(entry)


def finalize(
    prov: Optional[RunProvenance],
    status: str,
    errors: Optional[List[str]] = None,
) -> Optional[RunProvenance]:
    """Stamp *prov* with end time, elapsed duration, and final status."""
    if prov is None:
        return None
    now = datetime.now(timezone.utc)
    prov.end_utc = now.isoformat()
    start = datetime.fromisoformat(prov.start_utc)
    prov.elapsed_seconds = round((now - start).total_seconds(), 3)
    prov.status = status
    if errors:
        prov.errors.extend(errors)
    return prov


# ---------------------------------------------------------------------------
# Active run provenance
# ---------------------------------------------------------------------------
#
# Components deep inside a step (the calibration optimizer) learn things the
# manifest must carry -- the seed actually used, the evaluation budget -- but
# are not handed the RunProvenance object. The SYMFLUENCE entry point
# registers its record here; such components append to it via
# get_active_provenance(). None when no workflow run is active (library use,
# tests), in which case recording is a no-op.

_ACTIVE: Optional[RunProvenance] = None


def set_active_provenance(prov: Optional[RunProvenance]) -> None:
    """Register *prov* as the record of the run in progress (None clears it)."""
    global _ACTIVE
    _ACTIVE = prov


def get_active_provenance() -> Optional[RunProvenance]:
    """The record registered by :func:`set_active_provenance`, if any."""
    return _ACTIVE


def record_calibration(prov: Optional[RunProvenance], info: Mapping[str, Any]) -> None:
    """Append one calibration's conditions (seed, budget, ...) to *prov*."""
    if prov is None:
        return
    prov.calibration.append(dict(info))


# ---------------------------------------------------------------------------
# Random seeds
# ---------------------------------------------------------------------------

#: Upper bound (exclusive) of drawn seeds: fits numpy's legacy 32-bit seeding
#: and ``random.seed`` alike, and stays readable in a log line.
SEED_UPPER = 2**31 - 1


def is_unset_seed(value: Any) -> bool:
    """True for the spellings of "no RANDOM_SEED" found in configs."""
    return value is None or (isinstance(value, str) and value.strip() in ("", "None", "none", "null"))


def resolve_seed(value: Any, logger: Optional[logging.Logger] = None,
                 context: str = "run") -> Tuple[int, str]:
    """Return ``(seed, source)`` for a configured RANDOM_SEED.

    A configured seed is returned unchanged (``source='config'``). An unset
    one is replaced by a freshly drawn seed (``source='drawn'``) that is
    logged, so a stochastic run is never unrecorded: setting
    ``RANDOM_SEED`` to the logged value regenerates it.
    """
    if not is_unset_seed(value):
        return int(value), "config"
    seed = secrets.randbelow(SEED_UPPER)
    if logger is not None:
        logger.info(
            f"RANDOM_SEED not set; drew seed {seed} for this {context}. "
            f"Set RANDOM_SEED: {seed} to regenerate it."
        )
    return seed, "drawn"


# ---------------------------------------------------------------------------
# Resolved configuration
# ---------------------------------------------------------------------------

RESOLVED_CONFIG_NAME = "resolved_config.yaml"

_SENSITIVE_KEY_RE = re.compile(r"PASSWORD|PASSWD|SECRET|TOKEN|API_?KEY|CREDENTIAL", re.IGNORECASE)


def _json_safe(value: Any) -> Any:
    """Plain JSON types for *value* (paths, dates, numpy scalars -> str/float)."""
    return json.loads(json.dumps(value, default=str))


def resolved_config_dict(config: Any) -> Dict[str, Any]:
    """The validated, defaults-filled configuration as a flat, JSON-safe dict.

    Accepts a ``SymfluenceConfig`` (anything with ``to_dict(flatten=True)``)
    or an already-flat mapping. Credential-like keys are masked, so the dict
    can be written next to a run and hashed without leaking secrets.
    """
    if hasattr(config, "to_dict"):
        flat = config.to_dict(flatten=True)
    else:
        flat = dict(config)
    out: Dict[str, Any] = {}
    for key, value in flat.items():
        if isinstance(key, str) and _SENSITIVE_KEY_RE.search(key) and value not in (None, ""):
            value = "***MASKED***"
        out[str(key)] = value
    return _json_safe(out)


def config_sha256(config: Any) -> str:
    """SHA-256 of the resolved configuration's canonical JSON form.

    Canonical = sorted keys, no whitespace, non-JSON values stringified, so
    the hash depends only on the resolved values, not on key order, YAML
    layout or which defaults the user happened to spell out.
    """
    flat = resolved_config_dict(config)
    canonical = json.dumps(flat, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_resolved_config(config: Any, output_dir: Path,
                          prov: Optional[RunProvenance] = None,
                          name: str = RESOLVED_CONFIG_NAME) -> Tuple[Path, str]:
    """Write the resolved config as YAML into *output_dir*; return (path, sha256).

    The file holds exactly the dict that is hashed (sorted keys), so
    ``config_sha256(yaml.safe_load(file))`` reproduces the recorded hash, and
    the file can be passed back to ``symfluence`` as a config to rerun.
    """
    import yaml

    flat = resolved_config_dict(config)
    sha = config_sha256(flat)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# Resolved SYMFLUENCE configuration (validated, defaults filled)\n"
                 f"# sha256 (canonical JSON of these values): {sha}\n")
        yaml.safe_dump(flat, fh, default_flow_style=False, sort_keys=True)
    if prov is not None:
        prov.resolved_config = {"path": str(path), "sha256": sha, "n_keys": len(flat)}
    return path, sha


# ---------------------------------------------------------------------------
# Input data fingerprints (forcing, observations)
# ---------------------------------------------------------------------------

INPUT_HASH_MODES = ("listing", "content", "none")


def _file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def fingerprint_files(root: Path, mode: str = "listing",
                      patterns: Iterable[str] = ("*",)) -> Optional[Dict[str, Any]]:
    """Fingerprint the files under *root* (recursively).

    ``mode='listing'`` hashes each file's relative path, size and mtime --
    cheap even for decades of gridded forcing, and it changes whenever a file
    is added, removed, resized or rewritten. ``mode='content'`` hashes every
    byte (SHA-256 per file, combined in path order) and is immune to
    ``touch``/copy mtime changes but reads all the data. Returns None when
    *root* has no matching files or mode is ``'none'``.
    """
    if mode == "none":
        return None
    if mode not in INPUT_HASH_MODES:
        raise ValueError(f"Unknown input hash mode {mode!r}; expected one of {INPUT_HASH_MODES}")
    root = Path(root)
    if root.is_file():
        files = [root]
        base = root.parent
    elif root.is_dir():
        found: set = set()
        for pattern in patterns:
            found.update(p for p in root.rglob(pattern) if p.is_file())
        files = sorted(found)
        base = root
    else:
        return None
    if not files:
        return None
    combined = hashlib.sha256()
    total = 0
    for f in files:
        st = f.stat()
        total += st.st_size
        rel = os.path.relpath(f, base).replace(os.sep, "/")
        if mode == "content":
            entry = f"{rel}\t{_file_sha256(f)}\n"
        else:
            entry = f"{rel}\t{st.st_size}\t{st.st_mtime_ns}\n"
        combined.update(entry.encode("utf-8"))
    return {
        "path": str(root),
        "mode": mode,
        "n_files": len(files),
        "total_bytes": total,
        "sha256": combined.hexdigest(),
    }


def record_input_tree(prov: Optional[RunProvenance], name: str, root: Path,
                      mode: str = "listing", patterns: Iterable[str] = ("*",)) -> None:
    """Record :func:`fingerprint_files` of *root* under ``inputs[name]``.

    No-op when *prov* is None, the mode is ``'none'`` or nothing is there.
    """
    if prov is None:
        return
    info = fingerprint_files(root, mode=mode, patterns=patterns)
    if info is not None:
        prov.inputs[name] = info
