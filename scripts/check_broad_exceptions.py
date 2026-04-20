#!/usr/bin/env python3
"""
Fail CI if new broad Exception catches are introduced.

Matches:
  - except Exception:
  - except Exception as e:
  - except (..., Exception, ...):

Comparison is done against a checked-in allowlist so existing technical debt
does not fail CI, while any newly introduced occurrences do.

Allowlist key format: ``path:context:source``
  - ``path``: posix-style file path relative to the scan root.
  - ``context``: dotted enclosing scope (e.g. ``MyClass.my_method`` or
    ``<module>.helper_func``); ``<module>`` is the literal placeholder
    for top-level scope.
  - ``source``: the stripped source line of the ``except`` clause.

This format is stable against unrelated line shifts in the same file
— editing or reordering code elsewhere doesn't drift entries. Two
identical except clauses inside the same function still collapse to a
single entry; if you need to allowlist a *new* identical clause, that
counts as a new occurrence and CI will (correctly) flag it for review.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import List, Tuple

DEFAULT_SCAN_ROOT = Path("src/symfluence")
DEFAULT_ALLOWLIST = Path("tools/quality/broad_exception_allowlist.txt")
MODULE_SCOPE = "<module>"


def _is_broad_exception(node: ast.expr | None) -> bool:
    """Return True when an except handler catches built-in Exception."""
    if node is None:
        return False

    if isinstance(node, ast.Name):
        return node.id == "Exception"

    if isinstance(node, ast.Attribute):
        return node.attr == "Exception"

    if isinstance(node, ast.Tuple):
        return any(_is_broad_exception(elt) for elt in node.elts)

    return False


def _qualified_context(stack: List[str]) -> str:
    """Render a context stack into a dotted path, with module placeholder."""
    if not stack:
        return MODULE_SCOPE
    return ".".join(stack)


def _walk_except_handlers(
    tree: ast.AST,
    source_lines: List[str],
) -> List[Tuple[str, str]]:
    """Yield (context, source_line) pairs for each broad-Exception handler.

    Context is the dotted enclosing scope (class/function names),
    determined by walking the AST manually so we can track the stack
    rather than relying on ``ast.walk`` which loses parent context.
    """
    matches: List[Tuple[str, str]] = []

    def _visit(node: ast.AST, stack: List[str]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            new_stack = stack + [node.name]
        else:
            new_stack = stack

        if isinstance(node, ast.ExceptHandler) and _is_broad_exception(node.type):
            lineno = node.lineno
            src = (
                source_lines[lineno - 1].strip()
                if 1 <= lineno <= len(source_lines)
                else ""
            )
            matches.append((_qualified_context(stack), src))

        for child in ast.iter_child_nodes(node):
            _visit(child, new_stack)

    for child in ast.iter_child_nodes(tree):
        _visit(child, [])

    return matches


def collect_matches(scan_root: Path) -> list[str]:
    """Return normalized match records in form: path:context:source."""
    records: list[str] = []
    py_files = sorted(scan_root.rglob("*.py"))

    for path in py_files:
        rel = path.as_posix()
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"warning: could not read {rel}: {exc}", file=sys.stderr)
            continue

        source_lines = source.splitlines()

        try:
            tree = ast.parse(source, filename=rel)
        except SyntaxError as exc:
            print(f"warning: could not parse {rel}: {exc}", file=sys.stderr)
            continue

        for context, line in _walk_except_handlers(tree, source_lines):
            records.append(f"{rel}:{context}:{line}")

    # Multiset semantics: keep duplicate occurrences. Functions
    # legitimately have multiple identical ``except Exception:`` blocks
    # (78 such duplicate keys in this codebase as of writing). If we
    # collapsed duplicates, adding a *new* identical block in the same
    # function would silently match the existing allowlist entry. By
    # preserving counts the comparison in ``main()`` correctly detects
    # any going-from-N-to-N+1 change.
    return sorted(records)


def read_allowlist(path: Path) -> list[str]:
    """Read allowlist lines (ignoring comments/blank lines)."""
    if not path.exists():
        return []

    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        lines.append(text)
    return sorted(lines)


def write_allowlist(path: Path, records: list[str]) -> None:
    """Write allowlist with a short header and sorted records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Broad Exception Catch Allowlist",
        "# Auto-generated by scripts/check_broad_exceptions.py --update",
        "# Format: relative/path.py:enclosing.scope:source",
        "# Stable against unrelated line shifts; duplicate lines are kept",
        "# (multiset semantics) so a new identical handler in the same scope",
        "# is still detected as a new occurrence.",
        "",
    ]
    body = [f"{r}\n" for r in records]
    path.write_text("\n".join(header) + "".join(body), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-root",
        type=Path,
        default=DEFAULT_SCAN_ROOT,
        help=f"Root directory to scan (default: {DEFAULT_SCAN_ROOT})",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=DEFAULT_ALLOWLIST,
        help=f"Allowlist path (default: {DEFAULT_ALLOWLIST})",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Regenerate allowlist from current scan",
    )
    args = parser.parse_args()

    current = collect_matches(args.scan_root)

    if args.update:
        write_allowlist(args.allowlist, current)
        print(f"Updated allowlist with {len(current)} entries: {args.allowlist}")
        return 0

    allowed = read_allowlist(args.allowlist)

    # Multiset comparison via Counter so duplicate occurrences in the
    # same scope are tracked individually. ``current - allowed`` flags
    # newly introduced excess; ``allowed - current`` flags stale entries.
    from collections import Counter
    allowed_mset = Counter(allowed)
    current_mset = Counter(current)

    new_items = sorted((current_mset - allowed_mset).elements())
    stale_items = sorted((allowed_mset - current_mset).elements())

    if new_items:
        print("New broad Exception catches detected (not in allowlist):", file=sys.stderr)
        for item in new_items:
            print(f"  {item}", file=sys.stderr)
        print(
            "\nTo accept intentional changes, run:\n"
            "  python scripts/check_broad_exceptions.py --update",
            file=sys.stderr,
        )
        return 1

    print(f"Broad exception guard passed ({len(current)} allowlisted matches).")
    if stale_items:
        print(
            f"Note: {len(stale_items)} stale allowlist entries detected. "
            "Run --update to prune.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
