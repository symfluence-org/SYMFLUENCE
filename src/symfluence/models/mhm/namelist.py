# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""The one writer of parameter values into mHM namelists.

Used by the calibration worker and the parameter manager alike.

``mhm_parameter.nml`` rows are ``name = lower, upper, value, FLAG, SCALING``.
mHM aborts at start-up ("parameter in namelist ... out of bound",
``mo_mpr_read_config.f90`` / ``mo_mrm_read_config.f90``) when ``value`` is
outside ``[lower, upper]``, so a calibration bound wider than the row's own
range used to kill every run that sampled the difference. Writing a value
therefore also widens the row's bounds to include it; values already inside
leave the bounds untouched, character for character.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Optional


def format_namelist_value(value: float) -> str:
    """Format a parameter value for Fortran namelist syntax.

    Scientific notation for very small or very large values, fixed-point
    otherwise.
    """
    abs_val = abs(value)
    if abs_val == 0.0:
        return '0.0'
    elif abs_val < 0.001 or abs_val >= 1e6:
        return f'{value:.6e}'
    elif abs_val == int(abs_val) and abs_val < 1e6:
        return f'{value:.1f}'
    else:
        return f'{value:.6f}'


def _fortran_float(text: str) -> float:
    return float(text.strip().lower().replace('d', 'e'))


def _param_row_pattern(param_name: str) -> 're.Pattern[str]':
    """Row of the 5-column format, anchored on the exact parameter name."""
    return re.compile(
        r'(^[ \t]*' + re.escape(param_name) + r'\s*=\s*)'  # 1: name =
        r'([^,\n]+)(,\s*)'                                 # 2: lower   3: sep
        r'([^,\n]+)(,\s*)'                                 # 4: upper   5: sep
        r'([^,\n]+)'                                       # 6: VALUE
        r'(,\s*\d+,\s*\d+)',                               # 7: , flag, scaling
        re.MULTILINE,
    )


def update_namelist_parameters(
    namelist_file: Path,
    params: Dict[str, float],
    logger: logging.Logger,
    is_param_nml: Optional[bool] = None,
) -> None:
    """Write ``params`` into an mHM namelist.

    Two formats are supported:

    1. ``mhm_parameter.nml`` (``name = lower, upper, value, FLAG, SCALING``):
       the value is replaced and ``lower``/``upper`` are widened to
       ``min(lower, value)`` / ``max(upper, value)`` when the value falls
       outside them.
    2. Legacy single-value (``name = value``).

    Args:
        namelist_file: Namelist to rewrite in place.
        params: Parameter name -> value.
        logger: Receives a warning for names missing from the file.
        is_param_nml: Force the format; by default it follows the file name.
    """
    namelist_file = Path(namelist_file)
    content = namelist_file.read_text(encoding='utf-8')
    if is_param_nml is None:
        is_param_nml = 'mhm_parameter' in namelist_file.name

    for param_name, value in params.items():
        formatted_value = format_namelist_value(value)

        if is_param_nml:
            pattern = _param_row_pattern(param_name)

            def _rewrite(m: 're.Match[str]', formatted=formatted_value, name=param_name) -> str:
                # Compare what mHM will read (the formatted value), not the
                # unrounded float.
                value = float(formatted)
                lower, upper = m.group(2), m.group(4)
                if value < _fortran_float(lower):
                    logger.debug(f"{name}: namelist lower bound {lower.strip()} widened to {formatted}")
                    lower = formatted
                if value > _fortran_float(upper):
                    logger.debug(f"{name}: namelist upper bound {upper.strip()} widened to {formatted}")
                    upper = formatted
                return m.group(1) + lower + m.group(3) + upper + m.group(5) + formatted + m.group(7)

            content, count = pattern.subn(_rewrite, content)
        else:
            pattern = re.compile(
                r'(^[ \t]*' + re.escape(param_name) + r'\s*=\s*)([^\s,!/]+)',
                re.MULTILINE,
            )

            def _replace(m: 're.Match[str]', formatted=formatted_value) -> str:
                return m.group(1) + formatted

            content, count = pattern.subn(_replace, content)

        if count:
            logger.debug(f"Updated {param_name} = {formatted_value}")
        else:
            logger.warning(f"Parameter {param_name} not found in {namelist_file}")

    namelist_file.write_text(content, encoding='utf-8')
