#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Show, check, or bump the upstream commit pins for external tool builds.

The pins live in ``src/symfluence/cli/binary_pins.json``; every recipe with a
``repository`` is built from the commit pinned there (or from a ``git_hash``
held in the recipe itself). This script asks each upstream, with
``git ls-remote``, where the recipe's branch (or default branch) points today.

Usage::

    # Table of pinned vs upstream commits (no changes)
    python scripts/update_binary_pins.py

    # Report drift for CI; exits 0 when pins are merely behind upstream,
    # 1 only when a recipe has no valid pin at all
    python scripts/update_binary_pins.py --check

    # Move pins to the current upstream commit (all tools, or the ones named)
    python scripts/update_binary_pins.py --write
    python scripts/update_binary_pins.py --write summa fuse

Bumping a pin changes what the next release builds: rebuild and rerun the
affected model tests before merging a bump, and say so in the PR.

Run from a checkout (``PYTHONPATH=src``) or with SYMFLUENCE installed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from symfluence.cli.binary_pins import (  # noqa: E402
    PINS_FILE,
    SHA_RE,
    _same_repository,
    load_pins,
    resolve_remote_ref,
)


def _recipes() -> dict:
    from symfluence.cli.external_tools_config import get_external_tools_definitions
    return {k: v for k, v in get_external_tools_definitions().items() if v.get("repository")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="report drift (never fails on drift; fails on a missing pin)")
    mode.add_argument("--write", action="store_true",
                      help="update binary_pins.json to the current upstream commits")
    parser.add_argument("tools", nargs="*", help="limit to these tools (default: all)")
    parser.add_argument("--pins-file", type=Path, default=PINS_FILE)
    args = parser.parse_args(argv)

    recipes = _recipes()
    unknown = [t for t in args.tools if t not in recipes]
    if unknown:
        print(f"Unknown or repository-less tools: {', '.join(unknown)}", file=sys.stderr)
        return 2
    selected = args.tools or sorted(recipes)

    pins = load_pins(args.pins_file)
    on_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    missing, drifted, changed = [], [], []

    print(f"{'tool':<15} {'ref':<18} {'pinned':<10} {'upstream':<10} status")
    for name in selected:
        info = recipes[name]
        repository, branch = info["repository"], info.get("branch")
        held = info.get("git_hash") if info.get("source_pin", {}).get("pinned_by") == "recipe" else None
        pin = pins.get(name, {})
        pinned = held or pin.get("commit")
        valid = bool(pinned and SHA_RE.match(pinned)) and (
            bool(held) or _same_repository(pin.get("repository"), repository))
        upstream = resolve_remote_ref(repository, branch)

        if held:
            status = "held in recipe" + ("" if upstream in (None, held) else " (branch has moved)")
        elif not valid:
            status = "MISSING/INVALID PIN"
            missing.append(name)
        elif upstream is None:
            status = "upstream unreachable"
        elif upstream == pinned:
            status = "up to date"
        else:
            status = "behind upstream"
            drifted.append(name)
        print(f"{name:<15} {(branch or 'HEAD'):<18} {(pinned or '-')[:8]:<10} "
              f"{(upstream or '-')[:8]:<10} {status}")

        if args.write and not held and upstream and upstream != pin.get("commit"):
            pins[name] = {
                "repository": repository,
                "ref": branch,
                "commit": upstream,
                "resolved": date.today().isoformat(),
            }
            changed.append(name)

    if args.write:
        if changed:
            payload = json.loads(args.pins_file.read_text(encoding="utf-8")) \
                if args.pins_file.exists() else {}
            payload["tools"] = dict(sorted(pins.items()))
            args.pins_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(f"\nUpdated {len(changed)} pin(s): {', '.join(changed)} -> {args.pins_file}")
        else:
            print("\nNo pins changed.")
        return 0

    if drifted:
        print(f"\n{len(drifted)} pin(s) behind upstream: {', '.join(drifted)}")
        if on_ci and args.check:
            print(f"::notice title=Binary source pins::{len(drifted)} pin(s) behind "
                  f"upstream ({', '.join(drifted)}); bump with scripts/update_binary_pins.py --write")
    if missing:
        print(f"\nRecipes without a valid pin: {', '.join(missing)} "
              "(run with --write to add them)", file=sys.stderr)
        return 1 if args.check else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
