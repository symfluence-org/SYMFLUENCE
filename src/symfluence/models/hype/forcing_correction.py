# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared FORCING_CORRECTION in HYPE forward runs: pcurain/pcusnow in par.txt."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

from symfluence.core.exceptions import ModelExecutionError


def apply_forward(config: Any, logger: Any, project_dir: Path, info: Dict[str, Any]) -> Dict[str, Any]:
    par = Path(project_dir) / 'settings' / 'HYPE' / 'par.txt'
    native = {k: float(v) for k, v in info['native'].items()}
    if not native:
        return {'par_file': str(par), 'applied': {}}
    content = par.read_text(encoding='utf-8')
    missing = [n for n in native if not re.search(rf'^{n}[ \t]', content, re.MULTILINE)]
    if missing:
        if all(v == 0.0 for v in native.values()):
            return {'par_file': str(par), 'applied': 'identity, par.txt has no entries'}
        raise ModelExecutionError(f'{missing} not in {par}; add them with HYPE_EXTRA_PARAMETERS')
    for name, value in native.items():
        def replace(match: 're.Match[str]', _value: float = value) -> str:
            count = len(match.group(2).split())
            return match.group(1) + '\t'.join([repr(_value)] * count)
        content = re.sub(rf'^({name}[ \t]+)([\d\.\-\+eE]+(?:[ \t]+[\d\.\-\+eE]+)*)', replace,
                         content, count=1, flags=re.MULTILINE)
    par.write_text(content, encoding='utf-8')
    return {'par_file': str(par), 'applied': native}
