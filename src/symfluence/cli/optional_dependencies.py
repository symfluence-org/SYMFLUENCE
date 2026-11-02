# SPDX-License-Identifier: GPL-3.0-or-later
"""Safe, interactive handling of missing optional dependency groups."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from typing import Callable

from symfluence.core.exceptions import OptionalDependencyError

from .console import Console
from .exit_codes import ExitCode

logger = logging.getLogger(__name__)


def installation_prompt_allowed(console: Console) -> bool:
    """Return whether this process is safe for an installation prompt."""
    disabled = os.environ.get("SYMFLUENCE_NO_INSTALL_PROMPT", "").lower()
    return (
        not console.is_quiet
        and disabled not in {"1", "true", "yes"}
        and not os.environ.get("CI")
        and bool(getattr(sys.stdin, "isatty", lambda: False)())
        and bool(getattr(sys.stdout, "isatty", lambda: False)())
    )


def _install_command(target: str) -> list[str]:
    """Choose an installer while targeting the running Python environment."""
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip", "install", "--python", sys.executable, target]
    return [sys.executable, "-m", "pip", "install", target]


def handle_optional_dependency(
    error: OptionalDependencyError,
    console: Console,
    *,
    confirm: Callable[[str], bool] | None = None,
) -> int:
    """Explain a missing extra and, in a TTY, offer an explicit install."""
    console.error(str(error))
    logger.warning("Optional dependency unavailable for %s: extra=%s dependency=%s",
                   error.feature, error.extra, error.dependency or "unspecified")

    if not installation_prompt_allowed(console):
        return ExitCode.DEPENDENCY_ERROR

    ask = confirm or (lambda message: input(f"{message} [y/N]: ").strip().lower() in {"y", "yes"})
    if not ask(f'Install "{error.install_target}" into this environment now?'):
        logger.info("Optional dependency installation declined: extra=%s", error.extra)
        return ExitCode.DEPENDENCY_ERROR

    command = _install_command(error.install_target)
    logger.info("Installing optional dependency group: extra=%s", error.extra)
    try:
        result = subprocess.run(command, check=False)  # fixed argv, no shell
    except OSError as exc:
        logger.error("Could not start dependency installer", exc_info=True)
        console.error(f"Could not start the installer: {exc}")
        return ExitCode.DEPENDENCY_ERROR

    if result.returncode:
        logger.error("Optional dependency installation failed: extra=%s code=%s",
                     error.extra, result.returncode)
        console.error(f"Installation failed with exit code {result.returncode}.")
        return ExitCode.DEPENDENCY_ERROR

    logger.info("Optional dependency installation completed: extra=%s", error.extra)
    console.success("Dependencies installed. Re-run the command to continue.")
    return ExitCode.DEPENDENCY_ERROR
