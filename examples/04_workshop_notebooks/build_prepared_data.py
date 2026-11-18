"""Build the prepared-data zips for workshop notebooks 04d and 04e (maintainers only).

Run after a complete execution of both notebooks with USE_PREPARED_DATA = False
and RUN_STEPS = True:

    python examples/04_workshop_notebooks/build_prepared_data.py --out dist/workshop \
        --release-url https://github.com/symfluence-org/SYMFLUENCE/releases/download/<tag>

Each zip holds one ``domain_<name>/`` folder. Run logs (``_workLog_*``, ``*.log``)
are left out because they record machine-specific paths. The script rewrites
``prepared_data.json`` with each zip's size and SHA-256; commit that file and
upload the zips under the same release tag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "prepared_data.json"
DOMAINS = {
    "rohtang_snow": "04d_intelligent_rivers_snow",
    "ayilam_camels_ind": "04e_intelligent_rivers_streamflow",
}
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints"}
SKIP_SUFFIXES = {".log", ".pyc"}


def _included(path: Path, project: Path) -> bool:
    rel = path.relative_to(project)
    if rel.parts[0].startswith("_workLog_") or rel.name == ".workshop_bundle.json":
        return False
    if SKIP_DIRS.intersection(rel.parts) or path.suffix in SKIP_SUFFIXES:
        return False
    return path.is_file() and not path.is_symlink()


def build(data_dir: Path, out_dir: Path, release_url: str, domains) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundles = {}
    for domain in domains:
        notebook = DOMAINS[domain]
        project = data_dir / f"domain_{domain}"
        if not project.is_dir():
            raise FileNotFoundError(project)
        archive = out_dir / f"workshop_{domain}.zip"
        files = sorted(p for p in project.rglob("*") if _included(p, project))
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path in files:
                zf.write(path, Path(project.name) / path.relative_to(project))
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        bundles[domain] = {
            "notebook": notebook,
            "file": archive.name,
            "url": f"{release_url.rstrip('/')}/{archive.name}",
            "sha256": digest,
            "size_mb": round(archive.stat().st_size / 1e6, 1),
            "files": len(files),
        }
        print(f"{archive}: {bundles[domain]['size_mb']} MB, {len(files)} files")
    return bundles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=HERE.parents[2] / "SYMFLUENCE_data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--release-url", required=True)
    parser.add_argument("--domains", nargs="+", choices=sorted(DOMAINS), default=list(DOMAINS))
    args = parser.parse_args()
    bundles = build(args.data_dir.resolve(), args.out, args.release_url, args.domains)
    if MANIFEST.exists():  # keep entries for domains not rebuilt this time
        bundles = {**json.loads(MANIFEST.read_text())["bundles"], **bundles}
    MANIFEST.write_text(json.dumps({"bundles": bundles}, indent=2) + "\n")
    print(f"Updated {MANIFEST}")


if __name__ == "__main__":
    main()
