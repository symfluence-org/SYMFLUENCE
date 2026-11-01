# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SUMMA Model Visualizer.

Provides model-specific visualization registration for SUMMA.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.core.registries import R


@R.visualizers.add('SUMMA')
def visualize_summa(reporting_manager: Any, config: Dict[str, Any], project_dir: Path, experiment_id: str, workflow: List[str]):
    """
    Visualize SUMMA model outputs.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running SUMMA visualizer for experiment {experiment_id}")

    domain_name = resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')

    reporting_manager.visualize_summa_outputs(experiment_id)
    obs_files = [('Observed', str(resolve_data_subdir(project_dir, 'observations') / "streamflow" / "preprocessed" / f"{domain_name}_streamflow_processed.csv"))]

    # Check if MizuRoute was part of the workflow
    if 'MIZUROUTE' in workflow and resolve_config_value(config, lambda: config.model.mizuroute.from_model, None, 'MIZU_FROM_MODEL') == 'SUMMA':
        reporting_manager.update_sim_reach_id()
        model_outputs = [('SUMMA', str(project_dir / "simulations" / experiment_id / "mizuRoute" / f"{experiment_id}*.nc"))]
        reporting_manager.visualize_model_outputs(model_outputs, obs_files)
    else:
        summa_output_file = str(project_dir / "simulations" / experiment_id / "SUMMA" / f"{experiment_id}_timestep.nc")
        model_outputs = [('SUMMA', summa_output_file)]
        reporting_manager.visualize_lumped_model_outputs(model_outputs, obs_files)
