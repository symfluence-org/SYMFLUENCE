"""Write native Raven outlet discharge to the standard SYMFLUENCE results file."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from symfluence.core.modeling.base import StandardModelPostProcessor

from .adapter import discharge


class RavenPostProcessor(StandardModelPostProcessor):
    model_name = "RAVEN"
    output_file_pattern = "Hydrographs.csv"
    streamflow_unit = "cms"

    def extract_streamflow(self) -> Optional[Path]:
        output = self._get_output_dir()
        if not (output / "Hydrographs.csv").exists():
            self.logger.error(f"No Raven Hydrographs.csv in {output}")
            return None
        return self.save_streamflow_to_results(discharge(output))
