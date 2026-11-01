# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
LSTM Model Calibration Module.

Provides calibration infrastructure for LSTM (Long Short-Term Memory) neural
network hydrological models, supporting sequence-to-sequence learning.

Components:
    optimizer: LSTM-specific optimizer using PyTorch training loops
    (hyperparameters are managed by the shared ``MLParameterManager`` in
    ``symfluence.models.gnn.calibration.parameter_manager``)
    worker: Executes LSTM training runs with sequence batching

The calibration system supports:
- Hidden state dimension optimization
- Stacked LSTM layer configuration
- Dropout rate tuning for regularization
- Learning rate scheduling
- Early stopping with patience-based selection
- Entity-aware LSTM variants (EA-LSTM)
"""
