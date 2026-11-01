# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Calibration component auto-discovery.

Each model registers its optimizer / parameter manager / worker via decorators
(``@R.optimizers.add('X')`` etc.) that only fire when the model's
``calibration.<component>`` module is *imported*. This helper scans every model
package and imports those modules so the decorators run.

The important behaviour here is *how import failures are reported*. There are
two very different reasons an import can fail, and conflating them is what made
the old bare ``except ... : logger.debug(...)`` so misleading:

* The model genuinely has no calibration support for this component (no
  ``calibration`` subpackage, or no ``optimizer.py``). This is expected for the
  many run-only models — log at DEBUG and move on.
* The module *exists* but raised on import — almost always a missing optional
  dependency (e.g. a compiled model binding) or a real bug in that module. The
  old code swallowed this at DEBUG, so the model silently vanished from the
  registry and the user later saw a baffling ``No optimizer registered for
  model: X``. We now surface these at WARNING with the actual cause.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil

#: Back-compat shim packages under ``symfluence.models`` (see
#: docs/adr/0011-retire-unused-back-compat-shims.md). They are not models and
#: have no ``calibration`` subpackage; importing them only to find that out
#: would fire their ``DeprecationWarning`` on every ``import symfluence``.
_SHIM_PACKAGES = frozenset({'adapters', 'base', 'execution', 'mixins', 'state', 'utilities'})


def discover_calibration_components(component: str, logger: logging.Logger) -> None:
    """Import every model's ``calibration.<component>`` module to fire its decorator.

    Parameters
    ----------
    component:
        The calibration submodule to import for each model, e.g. ``"optimizer"``
        or ``"parameter_manager"``.
    logger:
        Logger used for the DEBUG (no support) and WARNING (real failure) lines.
    """
    try:
        import symfluence.models as models_pkg
    except ImportError:
        return

    for _importer, model_name, is_pkg in pkgutil.iter_modules(models_pkg.__path__):
        if not is_pkg or model_name in _SHIM_PACKAGES:
            continue

        module_path = f'symfluence.models.{model_name}.calibration.{component}'
        try:
            importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            missing = exc.name or ''
            # The target module — or its `calibration` parent — genuinely does
            # not exist: this model has no calibration support for `component`.
            if missing == module_path or module_path.startswith(missing + '.'):
                logger.debug("No calibration.%s module for model '%s'", component, model_name)
            else:
                # The module exists but a dependency it imports is missing.
                logger.warning(
                    "Model '%s': calibration.%s could not import dependency '%s', so the "
                    "model will NOT be available for calibration. Install the missing "
                    "dependency, or run `python -c \"import %s\"` to see the full traceback.",
                    model_name, component, missing, module_path,
                )
        except (ImportError, AttributeError) as exc:
            # The module exists and was found, but raised while importing — a real
            # error in that module (bad import, registration-time failure, ...).
            logger.warning(
                "Model '%s': calibration.%s exists but failed to import (%s: %s), so the "
                "model will NOT be available for calibration. Run "
                "`python -c \"import %s\"` to see the full traceback.",
                model_name, component, type(exc).__name__, exc, module_path,
            )
