# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Tests for ``scripts/check_broad_exceptions.py`` allowlist key stability.

Background: the original line-numbered key
(``path:lineno:source``) drifted on every PR that added or removed
lines anywhere in a file containing allowlisted handlers — 5 of the 9
PRs in the 2026-04-20 batch tripped this CI failure mode and needed a
mechanical allowlist refresh. The new key
(``path:enclosing.scope:source``) is stable against unrelated line
shifts but still detects:

  - new occurrences in a brand-new scope
  - additional duplicate occurrences in an already-allowlisted scope
    (multiset semantics)
"""

import importlib.util
from pathlib import Path
from textwrap import dedent

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_broad_exceptions.py"

# Load as a module so we can call the helpers directly without subprocess overhead.
spec = importlib.util.spec_from_file_location("check_broad_exceptions", SCRIPT_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # type: ignore[union-attr]

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).lstrip("\n"))


def test_key_format_uses_scope_not_lineno(tmp_path):
    """Records must use ``path:enclosing.scope:source`` keys."""
    src = tmp_path / "pkg" / "mod.py"
    _write(src, """
        class Foo:
            def bar(self):
                try:
                    risky()
                except Exception:
                    pass

        def baz():
            try:
                risky()
            except Exception as e:
                pass
    """)
    records = mod.collect_matches(tmp_path)
    assert any("Foo.bar:except Exception:" in r for r in records)
    assert any("baz:except Exception as e:" in r for r in records)


def test_module_scope_uses_placeholder(tmp_path):
    """Top-level (no enclosing function/class) uses ``<module>``."""
    src = tmp_path / "topmod.py"
    _write(src, """
        try:
            x = 1
        except Exception:
            pass
    """)
    records = mod.collect_matches(tmp_path)
    assert len(records) == 1
    assert records[0].endswith("topmod.py:<module>:except Exception:")


def test_unrelated_line_shift_does_not_drift_records(tmp_path):
    """The whole point of this PR: editing/reordering code elsewhere
    in a file must NOT change the records produced by collect_matches.
    Pre-fix this is exactly what tripped Lint on 5 of 9 PRs."""
    src = tmp_path / "stable.py"
    _write(src, """
        def helper():
            try:
                risky()
            except Exception:
                pass
    """)
    before = mod.collect_matches(tmp_path)

    # Insert a docstring + 5 unrelated blank lines above the function
    _write(src, """
        '''Module docstring added.'''




        def helper():
            try:
                risky()
            except Exception:
                pass
    """)
    after = mod.collect_matches(tmp_path)
    assert before == after, (
        "collect_matches output changed when only unrelated lines moved; "
        "the key format is still line-number-sensitive."
    )


def test_new_handler_in_new_scope_is_detected(tmp_path):
    """Adding a broad-except in a brand-new function must produce a
    new record (different scope key)."""
    src = tmp_path / "growing.py"
    _write(src, """
        def alpha():
            try:
                pass
            except Exception:
                pass
    """)
    before = set(mod.collect_matches(tmp_path))

    _write(src, """
        def alpha():
            try:
                pass
            except Exception:
                pass

        def beta():
            try:
                pass
            except Exception:
                pass
    """)
    after = set(mod.collect_matches(tmp_path))
    new = after - before
    assert any("growing.py:beta:except Exception:" in r for r in new), (
        f"New broad-except in beta() was not detected. New records: {new}"
    )


def test_new_duplicate_in_same_scope_is_detected_via_multiset(tmp_path):
    """Adding a SECOND identical broad-except in the same function
    must be visible as a count delta (multiset semantics) so it doesn't
    silently match the existing allowlist entry."""
    src = tmp_path / "dup.py"
    _write(src, """
        def helper():
            try:
                a()
            except Exception:
                pass
    """)
    before = mod.collect_matches(tmp_path)

    _write(src, """
        def helper():
            try:
                a()
            except Exception:
                pass
            try:
                b()
            except Exception:
                pass
    """)
    after = mod.collect_matches(tmp_path)

    # Multiset semantics: count of the duplicate key must increase from 1 to 2
    key_suffix = "dup.py:helper:except Exception:"
    before_count = sum(1 for r in before if r.endswith(key_suffix))
    after_count = sum(1 for r in after if r.endswith(key_suffix))
    assert before_count == 1
    assert after_count == 2
