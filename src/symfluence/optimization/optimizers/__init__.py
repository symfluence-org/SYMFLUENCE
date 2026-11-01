# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Back-compat shim package: the optimizers moved to ``symfluence.core.calibration.optimizers``.

Only ``symfluence.optimization.optimizers.base_model_optimizer`` is kept (external
plugins import it); that module emits the ``DeprecationWarning``. The package-level
re-exports had no importer and were removed (see
docs/adr/0011-retire-unused-back-compat-shims.md).
"""
