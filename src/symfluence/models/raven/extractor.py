"""Read interval-mean native Raven discharge."""

from __future__ import annotations

from symfluence.core.modeling.base import ModelResultExtractor

from .adapter import discharge


class RavenResultExtractor(ModelResultExtractor):
    def get_output_file_patterns(self):
        return {"streamflow": ["Hydrographs.csv"]}

    def get_variable_names(self, variable_type):
        return ["outlet [m3/s]"] if variable_type == "streamflow" else [variable_type]

    def extract_variable(self, output_file, variable_type, **kwargs):
        return discharge(output_file.parent)

    def requires_unit_conversion(self, variable_type):
        return False

    def get_spatial_aggregation_method(self, variable_type):
        return None
