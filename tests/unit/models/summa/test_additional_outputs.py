"""Config-only diagnostic outputs survive native SUMMA setup."""
from __future__ import annotations

import logging

from symfluence.core.config.models import SymfluenceConfig
from symfluence.models.summa.config_manager import SummaConfigManager


def test_configured_outputs_replace_frequency_and_add_new_variable(tmp_path):
    source, destination = tmp_path / 'source', tmp_path / 'settings'
    source.mkdir()
    (source / 'outputControl.txt').write_text('! template\nscalarSWE | 24\nscalarSnowfall | 24\n')
    manager = SummaConfigManager.__new__(SummaConfigManager)
    manager.logger = logging.getLogger(__name__)
    manager._config = SymfluenceConfig.from_file('examples/04_workshop_notebooks/config_rohtang_snow.yaml')
    manager._get_base_settings_source_dir_callback = lambda: source
    manager._get_default_path_callback = lambda *args: destination
    manager.copy_base_settings()
    content = (destination / 'outputControl.txt').read_text()
    assert 'scalarSWE | 24' in content
    assert content.count('scalarSnowfall') == 1
    assert 'scalarSnowfall | 1' in content
    assert 'scalarRainfall | 1' in content
