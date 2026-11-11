# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
FEWS PI Diagnostics collector and writer.

Implements the Delft-FEWS diagnostics contract: the adapter MUST write a
``diag.xml`` file to the output directory after every run, even on catastrophic
failure. The ``write()`` method therefore catches all internal exceptions.
"""
from __future__ import annotations

import logging

# Security rationale: writing trusted local diagnostics XML
import xml.etree.ElementTree as ET  # nosec B405
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal

logger = logging.getLogger(__name__)

DiagLevel = Literal["0", "1", "2", "3", "4"]

# Delft-FEWS levels (pi_diag.xsd): 0 = fatal, 1 = error, 2 = warn, 3 = info;
# "all levels higher than 3 are regarded as non-essential (debug)". Low is
# severe: writing 4 for a fatal error would have FEWS file it as debug chatter.
LEVEL_FATAL: DiagLevel = "0"
LEVEL_ERROR: DiagLevel = "1"
LEVEL_WARN: DiagLevel = "2"
LEVEL_INFO: DiagLevel = "3"
LEVEL_DEBUG: DiagLevel = "4"


@dataclass
class DiagMessage:
    """Single diagnostic message."""
    level: DiagLevel
    text: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DiagnosticsCollector:
    """Buffers diagnostic messages and writes PI ``<Diag>`` XML.

    Usage::

        diag = DiagnosticsCollector(output_dir / "diag.xml")
        diag.info("Pre-adapter started")
        try:
            ...  # adapter logic
        except Exception as exc:  # noqa: BLE001 — must-not-raise contract
            diag.fatal(str(exc))
        finally:
            diag.write()  # ALWAYS writes, never raises
    """

    def __init__(self, output_path: Path) -> None:
        self._output_path = Path(output_path)
        self._messages: List[DiagMessage] = []

    # ---- convenience methods ----

    def debug(self, text: str) -> None:
        self._messages.append(DiagMessage(level=LEVEL_DEBUG, text=text))

    def info(self, text: str) -> None:
        self._messages.append(DiagMessage(level=LEVEL_INFO, text=text))

    def warning(self, text: str) -> None:
        self._messages.append(DiagMessage(level=LEVEL_WARN, text=text))

    def error(self, text: str) -> None:
        self._messages.append(DiagMessage(level=LEVEL_ERROR, text=text))

    def fatal(self, text: str) -> None:
        self._messages.append(DiagMessage(level=LEVEL_FATAL, text=text))

    # ---- query ----

    @property
    def has_fatal(self) -> bool:
        return any(m.level == LEVEL_FATAL for m in self._messages)

    @property
    def has_errors(self) -> bool:
        return any(m.level in (LEVEL_FATAL, LEVEL_ERROR) for m in self._messages)

    @property
    def messages(self) -> List[DiagMessage]:
        return list(self._messages)

    # ---- writer ----

    def write(self) -> None:
        """Write ``diag.xml``.  **Never raises** — logs errors internally."""
        try:
            root = ET.Element("Diag")
            root.set("xmlns", "http://www.wldelft.nl/fews/PI")
            root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")

            for msg in self._messages:
                # pi_diag.xsd LineComplexType: level and description are
                # required *attributes* of <line>, not child elements.
                line = ET.SubElement(root, "line")
                line.set("level", msg.level)
                line.set("description", msg.text)

            self._output_path.parent.mkdir(parents=True, exist_ok=True)

            tree = ET.ElementTree(root)
            ET.indent(tree, space="  ")
            tree.write(
                str(self._output_path),
                xml_declaration=True,
                encoding="UTF-8",
            )
        except Exception:  # noqa: BLE001 — must-not-raise contract
            # Contract: write() MUST NOT raise.
            logger.exception("Failed to write FEWS diagnostics to %s", self._output_path)
