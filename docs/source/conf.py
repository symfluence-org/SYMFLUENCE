# Configuration file for the Sphinx documentation builder.
import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))

# Mock imports for autodoc - these packages have complex dependencies
# that are not needed for documentation generation
autodoc_mock_imports = [
    # Core scientific stack
    'numpy',
    'pandas',
    'scipy',
    'xarray',
    'pint_xarray',
    'distributed',
    'pyviscous',
    # Machine learning
    'torch',
    'sklearn',
    'scikit-learn',
    # Geospatial
    'gdal',
    'osgeo',
    'geopandas',
    'rasterio',
    'pyproj',
    'shapely',
    'fiona',
    'easymore',
    'rasterstats',
    'pvlib',
    'cdo',
    # Visualization
    'seaborn',
    'plotly',
    'matplotlib',
    'contextily',
    # Cloud and data access
    'gcsfs',
    'intake_xarray',
    'intake',
    'netCDF4',
    'h5netcdf',
    'cdsapi',
    's3fs',
    'cftime',
    # Hydrology specific
    'hydrobm',
    'baseflow',
    # Utilities
    'SALib',
    'psutil',
    'tqdm',
    'yaml',
    'requests',
    'numexpr',
    'bottleneck',
    'networkx',
    # AI/LLM
    'openai',
    # CLI
    'rich',
    # R integration
    'rpy2',
    # JAX (optional)
    'jax',
    'jaxlib',
    'equinox',
    'jfuse',
    # Pydantic
    'pydantic',
    'pydantic_core',
    # Units
    'pint',
    # Routing/model deps that are heavy or version-sensitive
    'meshflow',
    'timezonefinder',
    'numba',
    # Internal optional modules that can error at import time
    'symfluence.models.jfuse',
]

# Project information
project = 'SYMFLUENCE'
copyright = '2025-2026, Darri Eythorsson'
author = 'Darri Eythorsson'

# Read version from single source of truth (parse file directly to avoid import chain)
import re

_version_file = os.path.join(os.path.dirname(__file__), '../../src/symfluence/symfluence_version.py')
with open(_version_file) as _f:
    release = re.search(r'__version__\s*=\s*["\']([^"\']+)', _f.read()).group(1)

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_design',
    'myst_parser',
]

# Theme
html_theme = 'pydata_sphinx_theme'

# Theme options
html_theme_options = {
    "logo": {
        "text": "SYMFLUENCE",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/symfluence-org/SYMFLUENCE",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_align": "left",
    "navigation_with_keys": True,
    "show_nav_level": 1,
    "show_toc_level": 2,
    "navigation_depth": 3,
    "collapse_navigation": True,
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "primary_sidebar_end": [],
}

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'inherited-members': False,
    'member-order': 'bysource',
    'exclude-members': 'normalize_flat_config',
}
# Skip private members by default
autodoc_default_flags = ['members', 'undoc-members']
# Better type hint formatting
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Add support for both RST and Markdown
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

html_context = {
    "github_user": "DarriEy",
    "github_repo": "SYMFLUENCE",
    "github_version": "main",
    "doc_path": "docs/source",
}

# Static files
html_static_path = ['_static']
html_css_files = ['custom.css']
