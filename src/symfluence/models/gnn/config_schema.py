# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for GNN.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG


class GNNConfig(BaseModel):
    """GNN (Graph Neural Network) hydrological model configuration"""
    model_config = FROZEN_CONFIG

    load: bool = Field(default=False, alias='GNN_LOAD')
    hidden_size: int = Field(default=128, alias='GNN_HIDDEN_SIZE', ge=8, le=2048)
    num_layers: int = Field(default=3, alias='GNN_NUM_LAYERS', ge=1, le=10)
    epochs: int = Field(default=300, alias='GNN_EPOCHS', ge=1, le=10000)
    checkpoint_interval: int = Field(
        default=50, alias='GNN_CHECKPOINT_INTERVAL', ge=0,
        description='Save a training checkpoint every this many epochs; 0 disables'
    )
    batch_size: int = Field(default=64, alias='GNN_BATCH_SIZE', ge=1, le=4096)
    learning_rate: float = Field(default=0.001, alias='GNN_LEARNING_RATE', gt=0, le=1.0)
    learning_patience: int = Field(default=30, alias='GNN_LEARNING_PATIENCE', ge=1)
    dropout: float = Field(default=0.2, alias='GNN_DROPOUT', ge=0, le=0.9)
    l2_regularization: float = Field(default=1e-6, alias='GNN_L2_REGULARIZATION', ge=0)
    params_to_calibrate: str = Field(
        default='precip_mult,temp_offset,routing_velocity',
        alias='GNN_PARAMS_TO_CALIBRATE'
    )
    parameter_bounds: Optional[Dict[str, List[float]]] = Field(default=None, alias='GNN_PARAMETER_BOUNDS')
    output_size: int = Field(default=32, alias='GNN_OUTPUT_SIZE', ge=1, le=2048)
    use_snow: bool = Field(default=False, alias='GNN_USE_SNOW')
