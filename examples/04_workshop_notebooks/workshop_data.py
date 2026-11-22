"""Prepared-data bundles for the Intelligent Rivers workshop notebooks (04d, 04e).

ERA5 (CDS) and MODIS (NASA Earthdata/AppEEARS) acquisition needs accounts and can
queue for hours. Each notebook can instead download a zip of its project folder
(``domain_<name>``) as it was after a complete run on the instructors' machine:
acquired inputs, preprocessed model inputs, SUMMA runs and calibration results.

With the bundle in place a notebook can be read as a walkthrough without running
anything, or re-run from preprocessing onwards with the acquisition stages
skipped (see ``ACQUISITION_STAGES``).

Sources, in order:

1. ``SYMFLUENCE_WORKSHOP_DATA``: a local folder holding the zip files, e.g. a
   USB stick or a shared drive in a computer lab.
2. ``SYMFLUENCE_WORKSHOP_DATA_URL``: a base URL of a mirror holding the same
   zip file names, e.g. a server close to the workshop.
3. The release URL in ``prepared_data.json``.

Every zip is checked against the SHA-256 in ``prepared_data.json``.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ACQUISITION_STAGES = frozenset({"acquire_attributes", "process_observed_data", "acquire_forcings"})
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "prepared_data.json"
READY_MARKER = ".workshop_bundle.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _bundle_entry(domain: str) -> dict:
    bundles = json.loads(MANIFEST.read_text())["bundles"]
    if domain not in bundles:
        raise KeyError(f"No prepared bundle for domain '{domain}' in {MANIFEST.name}")
    return bundles[domain]


def _obtain_zip(entry: dict, scratch: Path) -> Path:
    local_dir = os.environ.get("SYMFLUENCE_WORKSHOP_DATA")
    if local_dir:
        candidate = Path(local_dir).expanduser() / entry["file"]
        if candidate.is_file():
            print(f"Using local bundle: {candidate}")
            return candidate
        print(f"{candidate} not found; downloading instead.")
    base = os.environ.get("SYMFLUENCE_WORKSHOP_DATA_URL", "").rstrip("/")
    url = f"{base}/{entry['file']}" if base else entry["url"]
    target = scratch / entry["file"]
    print(f"Downloading {url} ({entry['size_mb']:.0f} MB) ...")
    with urllib.request.urlopen(url) as response, target.open("wb") as out:  # noqa: S310 - fixed https/user mirror
        shutil.copyfileobj(response, out)
    return target


def ensure_prepared_project(project: Path) -> Path:
    """Make sure ``project`` (a ``domain_<name>`` folder) holds the workshop bundle.

    Does nothing if the bundle was already unpacked there. Refuses to unpack over
    an existing project that did not come from the bundle, so a participant's own
    run is never overwritten; move that folder aside first.
    """
    project = Path(project)
    domain = project.name.removeprefix("domain_")
    entry = _bundle_entry(domain)
    marker = project / READY_MARKER
    if marker.is_file() and json.loads(marker.read_text()).get("sha256") == entry["sha256"]:
        print(f"Prepared data already in place: {project}")
        return project
    if project.exists() and any(project.iterdir()):
        raise FileExistsError(
            f"{project} already exists and was not unpacked from the workshop bundle. "
            "Rename it (or set USE_PREPARED_DATA = False to use it as is)."
        )
    project.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=project.parent) as tmp:
        archive = _obtain_zip(entry, Path(tmp))
        if _sha256(archive) != entry["sha256"]:
            raise ValueError(f"Checksum mismatch for {archive.name}; delete it and download again.")
        with zipfile.ZipFile(archive) as bundle:
            prefix = f"{project.name}/"
            for name in bundle.namelist():
                if not name.startswith(prefix) or ".." in Path(name).parts:
                    raise ValueError(f"Unexpected path in bundle: {name}")
            bundle.extractall(project.parent)
    marker.write_text(json.dumps({"sha256": entry["sha256"], "file": entry["file"]}, indent=2))
    print(f"Prepared data unpacked to {project}")
    return project


def open_experiment(config_name: str, use_prepared_data: bool):
    """Load a workshop YAML and return ``(config, project)``.

    The YAMLs use paths relative to the repository root, so this changes the
    working directory there. With ``use_prepared_data`` the project folder is
    filled from the prepared bundle first.
    """
    import symfluence
    from symfluence.core.config.models import SymfluenceConfig

    os.chdir(REPO)
    config = SymfluenceConfig.from_file(HERE / config_name)
    project = config.system.data_dir / f"domain_{config.domain.name}"
    if use_prepared_data:
        ensure_prepared_project(project)
    print(f"SYMFLUENCE {symfluence.__version__} · config {Path(config_name).name}")
    print(f"Project: {project}")
    print("Data:", "prepared workshop bundle" if use_prepared_data else "own acquisition")
    return config, project


def stages_to_run(names, use_prepared_data: bool) -> list[str]:
    """Drop the online acquisition stages when the prepared bundle supplies their outputs."""
    if not use_prepared_data:
        return list(names)
    return [name for name in names if name not in ACQUISITION_STAGES]
