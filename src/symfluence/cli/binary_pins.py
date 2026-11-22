# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Upstream source pins for external tool builds.

Every build recipe that clones an upstream repository names a branch (or
nothing, meaning the default branch). A branch moves, so two builds of the
same SYMFLUENCE release a week apart could compile different upstream code.
``binary_pins.json`` (next to this module, shipped in the wheel) maps each
tool to the exact commit SHA it is built from; the branch is kept there as
metadata for humans and for ``scripts/update_binary_pins.py``, which bumps
the pins.

Precedence when choosing what to check out (see ``ToolInstaller.install``):

1. ``symfluence binary install <tool> --git-hash <sha>``
2. ``--branch <name>`` (tip of that branch, unpinned, on request)
3. a ``git_hash`` written into the recipe itself (a deliberate hold, e.g. a
   fork commit a patch was tested against)
4. the pin in ``binary_pins.json``

A pin only applies while its ``repository`` matches the recipe's: a recipe
that switches upstream without re-pinning is built unpinned (with a warning)
rather than asked to check out a commit that does not exist there.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess  # nosec B404 - git ls-remote with fixed argv
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

PINS_FILE = Path(__file__).with_name("binary_pins.json")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def load_pins(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Return ``{tool: {repository, ref, commit, ...}}`` from the pin file."""
    pins_path = Path(path) if path else PINS_FILE
    try:
        data = json.loads(pins_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read binary source pins %s: %s", pins_path, exc)
        return {}
    tools = data.get("tools", {})
    return tools if isinstance(tools, dict) else {}


def _same_repository(a: Optional[str], b: Optional[str]) -> bool:
    """Compare repository URLs ignoring a trailing ``.git`` and slash."""
    def norm(url: Optional[str]) -> str:
        url = (url or "").strip().rstrip("/")
        return url[:-4] if url.endswith(".git") else url
    return bool(a) and norm(a) == norm(b)


def apply_pins(
    tools: Dict[str, Dict[str, Any]],
    pins: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Fill ``git_hash`` on each recipe from the pin file, in place.

    Recipes that already carry a ``git_hash`` keep it (it is a deliberate
    hold). Each pinned recipe also gets ``source_pin`` describing where its
    commit came from, which the installer records in the install manifest.
    """
    pins = load_pins() if pins is None else pins
    for name, info in tools.items():
        repository = info.get("repository")
        if not repository:
            continue
        previous = info.get("source_pin") or {}
        # A git_hash this function filled in earlier is not a recipe hold:
        # re-derive it so repeated calls stay idempotent.
        recipe_hash = (
            None if previous.get("pinned_by") == "binary_pins.json" else info.get("git_hash")
        )
        if recipe_hash:
            info.setdefault("source_pin", {
                "repository": repository,
                "ref": info.get("branch"),
                "commit": recipe_hash,
                "pinned_by": "recipe",
            })
            continue
        info.pop("git_hash", None)
        info.pop("source_pin", None)
        pin = pins.get(name)
        if not pin:
            continue
        if not _same_repository(pin.get("repository"), repository):
            logger.warning(
                "Source pin for %s names %s but the recipe clones %s; building "
                "unpinned. Run scripts/update_binary_pins.py --write %s.",
                name, pin.get("repository"), repository, name,
            )
            continue
        commit = str(pin.get("commit", ""))
        if not SHA_RE.match(commit):
            logger.warning("Source pin for %s is not a 40-hex commit: %r", name, commit)
            continue
        info["git_hash"] = commit
        info["source_pin"] = {
            "repository": repository,
            "ref": pin.get("ref", info.get("branch")),
            "commit": commit,
            "pinned_by": "binary_pins.json",
        }
    return tools


def resolve_remote_ref(
    repository: str, ref: Optional[str], timeout: int = 60,
) -> Optional[str]:
    """Return the commit *ref* (branch, tag, or default HEAD) points to upstream.

    Annotated tags are peeled to their commit. Returns ``None`` when the ref
    cannot be found or the remote cannot be reached.
    """
    if ref:
        patterns = [f"refs/heads/{ref}", f"refs/tags/{ref}", f"refs/tags/{ref}^{{}}"]
    else:
        patterns = ["HEAD"]
    try:
        out = subprocess.run(  # nosec B603 B607
            ["git", "ls-remote", repository, *patterns],
            capture_output=True, text=True, timeout=timeout, check=True,
            env={"GIT_TERMINAL_PROMPT": "0", **_minimal_env()},
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("git ls-remote %s %s failed: %s", repository, ref or "HEAD", exc)
        return None
    refs: Dict[str, str] = {}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            refs[parts[1]] = parts[0]
    if not ref:
        return refs.get("HEAD")
    for key in (f"refs/heads/{ref}", f"refs/tags/{ref}^{{}}", f"refs/tags/{ref}"):
        if key in refs:
            return refs[key]
    return None


def _minimal_env() -> Dict[str, str]:
    import os
    keep = ("PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "SSL_CERT_FILE",
            "GIT_SSL_CAINFO", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY",
            "https_proxy", "http_proxy", "no_proxy")
    return {k: os.environ[k] for k in keep if k in os.environ}
