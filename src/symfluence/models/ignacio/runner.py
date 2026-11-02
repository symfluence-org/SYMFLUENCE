# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
IGNACIO Model Runner for SYMFLUENCE

Executes the IGNACIO fire spread model using the ignacio Python package.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler
from symfluence.core.modeling.base.base_runner import BaseModelRunner
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R

logger = logging.getLogger(__name__)


@R.runners.add('IGNACIO')
class IGNACIORunner(BaseModelRunner):
    """
    Runs the IGNACIO fire spread model.

    IGNACIO is a Python package implementing the Canadian FBP System
    for fire spread simulation. This runner invokes the ignacio package
    either via its Python API or CLI.

    Handles:
    - Loading IGNACIO configuration
    - Running fire spread simulation
    - Collecting and organizing outputs
    """


    MODEL_NAME = "IGNACIO"
    def __init__(self, config, logger_instance=None, reporting_manager=None):
        """
        Initialize the IGNACIO runner.

        Args:
            config: SymfluenceConfig object with IGNACIO settings
            logger_instance: Optional logger for status messages
            reporting_manager: Optional reporting manager for experiment tracking
        """
        super().__init__(config, logger_instance or logger, reporting_manager)

        # IGNACIO-specific paths
        self.ignacio_input_dir = self.project_dir / "IGNACIO_input"
        self.ignacio_config_path = self.ignacio_input_dir / "ignacio_config.yaml"

    def _is_fwi_only(self) -> bool:
        """Check if this is an FWI-only run (no ignition configured)."""
        import yaml
        if not self.ignacio_config_path.exists():
            return False
        with open(self.ignacio_config_path, encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        ignition = cfg.get('ignition', {})
        return ignition.get('source_type') == 'fwi_only'

    def run_ignacio(self, **kwargs) -> Optional[Path]:
        """
        Execute the IGNACIO fire spread simulation or FWI-only analysis.

        For FWI-only mode (no ignition), calculates FWI components from
        weather data without running fire spread.  Otherwise attempts
        the full simulation via Python API, falling back to CLI.

        Returns:
            Path to output directory on success, None on failure
        """
        self.logger.info(f"Running IGNACIO for domain: {self.config.domain.name}")

        with symfluence_error_handler(
            "IGNACIO fire simulation",
            self.logger,
            error_type=ModelExecutionError
        ):
            # Setup output directory
            self.output_dir = self.get_simulation_dir('IGNACIO', self.config.domain.experiment_id)
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Verify config file exists
            if not self.ignacio_config_path.exists():
                raise FileNotFoundError(
                    f"IGNACIO config not found: {self.ignacio_config_path}\n"
                    f"Run preprocessing first: symfluence workflow step preprocessing"
                )

            if self._is_fwi_only():
                success = self._run_fwi_only()
            else:
                # Try Python API first
                success = self._run_via_api()
                if not success:
                    self.logger.info("Falling back to IGNACIO CLI...")
                    success = self._run_via_cli()

            if success:
                self.logger.info(f"IGNACIO completed. Output: {self.output_dir}")
                return self.output_dir
            else:
                raise ModelExecutionError("IGNACIO simulation failed")

    def _run_fwi_only(self) -> bool:
        """
        Calculate FWI components from weather data without fire spread.

        Reads the preprocessed weather CSV and computes daily FWI
        components (FFMC, DMC, DC, ISI, BUI, FWI, DSR).

        Returns:
            True if successful, False otherwise
        """
        try:
            import pandas as pd
            import yaml
            from ignacio.fwi import calculate_fwi_from_weather

            with open(self.ignacio_config_path, encoding='utf-8') as f:
                cfg = yaml.safe_load(f)

            weather_path = cfg.get('weather', {}).get('station_path')
            if not weather_path or not Path(weather_path).exists():
                self.logger.error(f"Weather CSV not found: {weather_path}")
                return False

            latitude = cfg.get('weather', {}).get('fwi_latitude', 65.0)

            self.logger.info(f"FWI-only mode: calculating from {weather_path}")
            self.logger.info(f"  Latitude for day-length adjustment: {latitude}")

            weather_df = pd.read_csv(weather_path)
            self.logger.info(f"  Weather records: {len(weather_df)}")

            fwi_df = calculate_fwi_from_weather(
                weather_df,
                latitude=latitude,
            )

            fwi_csv = self.output_dir / 'fwi_results.csv'
            fwi_df.to_csv(fwi_csv, index=False)
            self.logger.info(f"  FWI results: {len(fwi_df)} days → {fwi_csv}")

            if len(fwi_df) > 0:
                self.logger.info(
                    f"  FFMC range: {fwi_df['FFMC'].min():.1f} – {fwi_df['FFMC'].max():.1f}"
                )
                self.logger.info(
                    f"  DMC  range: {fwi_df['DMC'].min():.1f} – {fwi_df['DMC'].max():.1f}"
                )
                self.logger.info(
                    f"  DC   range: {fwi_df['DC'].min():.1f} – {fwi_df['DC'].max():.1f}"
                )
                self.logger.info(
                    f"  ISI  range: {fwi_df['ISI'].min():.2f} – {fwi_df['ISI'].max():.2f}"
                )
                self.logger.info(
                    f"  BUI  range: {fwi_df['BUI'].min():.1f} – {fwi_df['BUI'].max():.1f}"
                )
                self.logger.info(
                    f"  FWI  range: {fwi_df['FWI'].min():.2f} – {fwi_df['FWI'].max():.2f}"
                )

                if cfg.get('output', {}).get('generate_plots', True):
                    self._generate_fwi_plots(fwi_df)

            return True

        except ImportError as e:
            self.logger.warning(f"IGNACIO FWI package not available: {e}")
            return False
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"FWI calculation failed: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _generate_fwi_plots(self, fwi_df) -> None:
        """Generate FWI time series plots."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.dates as mdates
            import matplotlib.pyplot as plt
            import pandas as pd

            dates = pd.to_datetime(fwi_df['DATE'])

            fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
            fig.suptitle(f'Fire Weather Index — {self.config.domain.name}', fontsize=14)

            components = [
                ('FFMC', 'Fine Fuel Moisture Code', 'tab:orange'),
                ('DMC', 'Duff Moisture Code', 'tab:brown'),
                ('DC', 'Drought Code', 'tab:red'),
                ('ISI', 'Initial Spread Index', 'tab:purple'),
                ('BUI', 'Buildup Index', 'tab:green'),
                ('FWI', 'Fire Weather Index', 'tab:blue'),
            ]

            for ax, (col, title, color) in zip(axes.flat, components):
                ax.plot(dates, fwi_df[col], color=color, linewidth=0.5)
                ax.set_ylabel(col)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            plt.tight_layout()
            plot_path = self.output_dir / 'fwi_timeseries.png'
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            self.logger.info(f"  FWI plot saved: {plot_path}")

        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"Could not generate FWI plots: {e}")

    def _run_via_api(self) -> bool:
        """
        Run IGNACIO via Python API.

        Returns:
            True if successful, False if API not available or failed
        """
        try:
            from ignacio.config import load_config, validate_paths
            from ignacio.simulation import run_simulation

            self.logger.info(f"Loading IGNACIO config from {self.ignacio_config_path}")
            config = load_config(self.ignacio_config_path)

            # Update output directory to SYMFLUENCE location
            config.project.output_dir = str(self.output_dir)

            # Validate input files
            warnings = validate_paths(config)
            for warning in warnings:
                self.logger.warning(f"IGNACIO: {warning}")

            # Run simulation
            self.logger.info(f"Starting IGNACIO simulation: {config.project.name}")
            results = run_simulation(config)

            # Log results summary
            self.logger.info("IGNACIO simulation complete:")
            self.logger.info(f"  - Fires simulated: {results.n_fires}")
            self.logger.info(f"  - Total area burned: {results.total_area_ha:.2f} ha")

            return True

        except ImportError as e:
            self.logger.warning(f"IGNACIO Python package not available: {e}")
            self.logger.warning("Install with: symfluence binary install ignacio")
            return False

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"IGNACIO API execution failed: {e}", exc_info=True)
            return False

    def _run_via_cli(self) -> bool:
        """
        Run IGNACIO via command line interface.

        Returns:
            True if successful, False otherwise
        """
        import shutil
        import subprocess

        # Check if ignacio CLI is available
        ignacio_cli = shutil.which('ignacio')
        if ignacio_cli is None:
            self.logger.error(
                "IGNACIO CLI not found. Install with: symfluence binary install ignacio"
            )
            return False

        try:
            cmd = [
                ignacio_cli,
                'run',
                str(self.ignacio_config_path),
                '--output', str(self.output_dir),
            ]

            self.logger.info(f"Running IGNACIO CLI: {' '.join(cmd)}")

            result = run_subprocess(
                cmd,
                cwd=str(self.project_dir),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.stdout:
                self.logger.debug(f"IGNACIO stdout: {result.stdout[-2000:]}")
            if result.stderr:
                self.logger.debug(f"IGNACIO stderr: {result.stderr[-2000:]}")

            if result.returncode != 0:
                self.logger.error(f"IGNACIO CLI failed with code {result.returncode}")
                self.logger.error(f"stderr: {result.stderr[-1000:] if result.stderr else 'none'}")
                return False

            return True

        except subprocess.TimeoutExpired:
            self.logger.error("IGNACIO simulation timed out after 1 hour")
            return False

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"IGNACIO CLI execution failed: {e}", exc_info=True)
            return False

    def run(self, **kwargs) -> Optional[Path]:
        """Execute IGNACIO fire spread simulation."""
        return self.run_ignacio(**kwargs)

    def _should_create_output_dir(self) -> bool:
        """Don't create output dir in __init__, we do it in run_ignacio."""
        return False
