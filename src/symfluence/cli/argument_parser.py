# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SYMFLUENCE CLI Argument Parser.

Provides the main command-line interface parser with a hierarchical subcommand
structure. The CLI follows a category-action pattern (e.g., 'workflow run',
'project init', 'binary install') for intuitive navigation.

Categories:
    - workflow: Execute and manage modeling workflows
    - project: Initialize projects and configure domains
    - binary: Manage external tool installations
    - config: Configuration file management and validation
    - job: SLURM job submission for HPC environments
    - example: Launch tutorial notebooks
    - agent: AI assistant interface
    - doctor: System diagnostics
"""

import argparse
from typing import List, Optional

from symfluence.workflow_steps import (
    WORKFLOW_STEP_ALIAS_REVERSE,
    WORKFLOW_STEP_ALIASES,
    WORKFLOW_STEP_NAMES,
    resolve_workflow_step_name,
)

try:
    from symfluence.symfluence_version import __version__
except ImportError:
    __version__ = "0+unknown"

# Workflow steps available for individual execution
WORKFLOW_STEPS = list(WORKFLOW_STEP_NAMES)

# Short aliases for workflow steps (alias -> canonical name)
STEP_ALIASES = dict(WORKFLOW_STEP_ALIASES)

# Build reverse lookup: canonical name -> list of aliases
_STEP_ALIAS_REVERSE: dict[str, list[str]] = {
    key: list(value) for key, value in WORKFLOW_STEP_ALIAS_REVERSE.items()
}


def resolve_step_name(name: str) -> str:
    """Resolve a step name or alias to the canonical workflow step name.

    Accepts the full canonical name (e.g. 'model_agnostic_preprocessing')
    or any registered alias (e.g. 'map', 'agnostic_prep').

    Raises:
        argparse.ArgumentTypeError: If the name is not recognised.
    """
    try:
        return resolve_workflow_step_name(name)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc

# Domain definition methods
DOMAIN_DEFINITION_METHODS = ['lumped', 'point', 'subset', 'delineate']

# Available tools for binary installation
# Default tools are installed by `symfluence binary install` (no arguments).
# Experimental tools require explicit naming, e.g. `symfluence binary install rhessys`.
# The default tier is the set `symfluence binary install` (no args)
# compiles out of the box. Every process-based model that appears in
# the paper's Fig 4 / Fig 8 multi-model ensemble is here so a fresh
# install is enough to reproduce the paper. JAX re-implementations
# (HBV*, SACSMA*, XAJ*, HECHMS*, TOPMODEL*, SUMMA+MOD) arrive as
# pip/jax dependencies in pyproject.toml and don't need a binary
# install. LSTM and GR4J are pure-Python / R-bridge models (no binary).
DEFAULT_TOOLS = [
    # Framework glue
    'sundials', 'taudem', 'gistool', 'datatool',
    # Routing
    'mizuroute', 'troute',
    # Hydrology engines in the paper's ensemble (alphabetical)
    'clm', 'clmparflow', 'crhm', 'fuse', 'gsflow', 'hype',
    'mesh', 'mhm', 'ngen', 'ngiab', 'parflow', 'pihm',
    'prms', 'rhessys', 'summa', 'swat', 'vic', 'watflood',
    'wflow', 'wrfhydro',
]
# Experimental tier: not in the paper ensemble, still buildable on
# explicit request.
EXPERIMENTAL_TOOLS = [
    'openfews', 'wmfire', 'cfuse', 'droute', 'ignacio',
    'modflow', 'enzyme',
]
EXTERNAL_TOOLS = DEFAULT_TOOLS + EXPERIMENTAL_TOOLS

# Hydrological models
MODELS = ['SUMMA', 'FUSE', 'GR', 'HYPE', 'MESH', 'RHESSys', 'NGEN', 'LSTM']


class CLIParser:
    """
    Main CLI parser with hierarchical subcommand architecture.

    Implements a two-level command structure where the first level represents
    a functional category (workflow, project, binary, etc.) and the second
    level represents specific actions within that category.

    Attributes:
        common_parser: Parent parser with global options (--config, --debug, etc.)
        parser: Main argument parser with all subcommands registered
    """

    def __init__(self):
        """Initialize the CLI parser with common options and all subcommands."""
        self.common_parser = self._create_common_parser()
        self.parser = self._create_parser()

    def _create_common_parser(self) -> argparse.ArgumentParser:
        """Create a parent parser with common arguments."""
        # Use SUPPRESS to avoid overwriting global flags with subcommand defaults
        parser = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)

        # Global options available to all commands
        parser.add_argument('--config', type=str,
                          help='Path to configuration file (default: ./config.yaml)')
        parser.add_argument('--debug', action='store_true',
                          help='Enable debug output')
        parser.add_argument('--visualise', '--visualize', action='store_true', dest='visualise',
                          help='Enable visualization during execution')
        parser.add_argument('--diagnostic', action='store_true',
                          help='Enable diagnostic plots for workflow validation')
        parser.add_argument('--dry-run', action='store_true', dest='dry_run',
                          help='Show what would be executed without running')
        parser.add_argument('--profile', action='store_true', dest='profile',
                          help='Enable I/O profiling to diagnose IOPS bottlenecks')
        parser.add_argument('--profile-output', type=str, dest='profile_output',
                          help='Path for profiling report output (default: profile_report.json)')
        parser.add_argument('--profile-stacks', action='store_true', dest='profile_stacks',
                          help='Capture stack traces in profiling (expensive, for debugging)')
        return parser

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the main parser with global options and subparsers."""
        parser = argparse.ArgumentParser(
            prog='symfluence',
            description='SYMFLUENCE - Hydrological Modeling Framework',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            parents=[self.common_parser],
            epilog="""
Examples:
  symfluence workflow run --config my_config.yaml
  symfluence workflow step calibrate_model
  symfluence project init fuse-provo --scaffold
  symfluence binary install summa mizuroute
  symfluence doctor
  symfluence project pour-point 51.1722/-115.5717 --domain-name Bow --definition delineate

For more help on a specific command:
  symfluence <category> --help
  symfluence <category> <action> --help
"""
        )

        parser.add_argument('--version', action='version',
                          version=f'SYMFLUENCE {__version__}')

        # Create subparsers for command categories
        subparsers = parser.add_subparsers(
            dest='category',
            required=True,
            help='Command category',
            metavar='<category>'
        )

        # Register all category commands
        self._register_workflow_commands(subparsers)
        self._register_project_commands(subparsers)
        self._register_binary_commands(subparsers)
        self._register_config_commands(subparsers)
        self._register_job_commands(subparsers)
        self._register_example_commands(subparsers)
        self._register_agent_commands(subparsers)
        self._register_gui_commands(subparsers)
        self._register_tui_commands(subparsers)
        self._register_data_commands(subparsers)
        self._register_doctor_commands(subparsers)
        self._register_fews_commands(subparsers)

        return parser

    def _register_workflow_commands(self, subparsers):
        """Register workflow category commands."""
        from .commands import WorkflowCommands

        workflow_parser = subparsers.add_parser(
            'workflow',
            help='Workflow execution and management',
            description='Execute and manage SYMFLUENCE workflows'
        )
        workflow_subparsers = workflow_parser.add_subparsers(
            dest='action',
            required=True,
            help='Workflow action',
            metavar='<action>'
        )

        # workflow run
        run_parser = workflow_subparsers.add_parser(
            'run',
            help='Run complete workflow',
            parents=[self.common_parser]
        )
        run_parser.add_argument('--force-rerun', action='store_true', dest='force_rerun',
                              help='Force rerun of all steps')
        run_parser.add_argument('--continue-on-error', action='store_true', dest='continue_on_error',
                              help='Continue executing steps even if errors occur')
        run_parser.set_defaults(func=WorkflowCommands.run)

        # workflow step
        step_parser = workflow_subparsers.add_parser(
            'step',
            help='Run a single workflow step',
            parents=[self.common_parser]
        )
        step_parser.add_argument('step_name', type=resolve_step_name, metavar='STEP_NAME',
                               help='Step to execute (use full name or alias; see "workflow list-steps")')
        step_parser.add_argument('--force-rerun', action='store_true', dest='force_rerun',
                               help='Force rerun of this step')
        step_parser.set_defaults(func=WorkflowCommands.run_step)

        # workflow steps (multiple)
        steps_parser = workflow_subparsers.add_parser(
            'steps',
            help='Run multiple workflow steps',
            parents=[self.common_parser]
        )
        steps_parser.add_argument('step_names', nargs='+', type=resolve_step_name, metavar='STEP_NAME',
                                help='Steps to execute in order (use full names or aliases; see "workflow list-steps")')
        steps_parser.add_argument('--force-rerun', action='store_true', dest='force_rerun',
                                help='Force rerun of these steps')
        steps_parser.set_defaults(func=WorkflowCommands.run_steps)

        # workflow status
        status_parser = workflow_subparsers.add_parser(
            'status',
            help='Show workflow execution status',
            parents=[self.common_parser]
        )
        status_parser.set_defaults(func=WorkflowCommands.status)

        # workflow validate
        validate_parser = workflow_subparsers.add_parser(
            'validate',
            help='Validate configuration file',
            parents=[self.common_parser]
        )
        validate_parser.set_defaults(func=WorkflowCommands.validate)

        # workflow list-steps
        list_steps_parser = workflow_subparsers.add_parser(
            'list-steps',
            help='List available workflow steps'
        )
        list_steps_parser.set_defaults(func=WorkflowCommands.list_steps)

        # workflow resume
        resume_parser = workflow_subparsers.add_parser(
            'resume',
            help='Resume workflow from a specific step',
            parents=[self.common_parser]
        )
        resume_parser.add_argument('step_name', type=resolve_step_name, metavar='STEP_NAME',
                                 help='Step to resume from (use full name or alias; see "workflow list-steps")')
        resume_parser.add_argument('--force-rerun', action='store_true', dest='force_rerun',
                                 help='Force rerun from this step')
        resume_parser.set_defaults(func=WorkflowCommands.resume)

        # workflow clean
        clean_parser = workflow_subparsers.add_parser(
            'clean',
            help='Clean intermediate or output files',
            parents=[self.common_parser]
        )
        clean_parser.add_argument('--level', choices=['intermediate', 'outputs', 'all'],
                                default='intermediate',
                                help='Cleaning level (default: intermediate)')
        clean_parser.set_defaults(func=WorkflowCommands.clean)

        # workflow diagnose
        diagnose_parser = workflow_subparsers.add_parser(
            'diagnose',
            help='Run diagnostic plots on existing workflow outputs',
            parents=[self.common_parser]
        )
        diagnose_parser.add_argument('--step', type=resolve_step_name, metavar='STEP',
                                    help='Run diagnostics for a specific step (use full name or alias; see "workflow list-steps")')
        diagnose_parser.set_defaults(func=WorkflowCommands.diagnose)

    def _register_project_commands(self, subparsers):
        """Register project category commands."""
        from .commands import ProjectCommands

        project_parser = subparsers.add_parser(
            'project',
            help='Project initialization and setup',
            description='Initialize projects and configure pour points'
        )
        project_subparsers = project_parser.add_subparsers(
            dest='action',
            required=True,
            help='Project action',
            metavar='<action>'
        )

        # project init
        init_parser = project_subparsers.add_parser(
            'init',
            help='Initialize a new project'
        )
        init_parser.add_argument('preset', nargs='?', default=None,
                               help='Preset name to use (optional)')
        init_parser.add_argument('--domain', type=str,
                               help='Domain name')
        init_parser.add_argument('--model', choices=MODELS,
                               help=f'Hydrological model. Choices: {", ".join(MODELS)}')
        init_parser.add_argument('--start-date', dest='start_date', type=str,
                               help='Start date (YYYY-MM-DD)')
        init_parser.add_argument('--end-date', dest='end_date', type=str,
                               help='End date (YYYY-MM-DD)')
        init_parser.add_argument('--forcing', type=str,
                               help='Forcing dataset')
        init_parser.add_argument('--discretization', type=str,
                               help='Discretization method')
        init_parser.add_argument('--definition-method', dest='definition_method', type=str,
                               help='Domain definition method')
        init_parser.add_argument('--output-dir', dest='output_dir', type=str,
                               default='./',
                               help='Output directory for config file (default: ./)')
        init_parser.add_argument('--scaffold', action='store_true',
                               help='Create full directory structure')
        init_parser.add_argument('--minimal', action='store_true',
                               help='Create minimal configuration')
        init_parser.add_argument('--comprehensive', action='store_true',
                               help='Create comprehensive configuration (default)')
        init_parser.add_argument('--interactive', '-i', action='store_true',
                               help='Run interactive configuration wizard')
        init_parser.set_defaults(func=ProjectCommands.init)

        # project pour-point
        pour_point_parser = project_subparsers.add_parser(
            'pour-point',
            help='Set up pour point workflow'
        )
        pour_point_parser.add_argument('coordinates', type=str,
                                      help='Pour point coordinates in format lat/lon (e.g., 51.1722/-115.5717)')
        pour_point_parser.add_argument('--domain-name', dest='domain_name', type=str, required=True,
                                      help='Domain name (required)')
        pour_point_parser.add_argument('--definition', dest='domain_def',
                                      choices=DOMAIN_DEFINITION_METHODS, required=True,
                                      help=f'Domain definition method. Choices: {", ".join(DOMAIN_DEFINITION_METHODS)}')
        pour_point_parser.add_argument('--bounding-box', dest='bounding_box_coords', type=str,
                                      help='Bounding box in format lat_max/lon_min/lat_min/lon_max')
        pour_point_parser.add_argument('--experiment-id', dest='experiment_id', type=str,
                                      help='Override experiment ID')
        pour_point_parser.set_defaults(func=ProjectCommands.pour_point)

        # project list-presets
        list_presets_parser = project_subparsers.add_parser(
            'list-presets',
            help='List available initialization presets'
        )
        list_presets_parser.set_defaults(func=ProjectCommands.list_presets)

        # project show-preset
        show_preset_parser = project_subparsers.add_parser(
            'show-preset',
            help='Show details of a specific preset'
        )
        show_preset_parser.add_argument('preset_name', type=str,
                                       help='Name of preset to display')
        show_preset_parser.set_defaults(func=ProjectCommands.show_preset)

    def _register_binary_commands(self, subparsers):
        """Register binary/tool management commands."""
        from .commands import BinaryCommands

        binary_parser = subparsers.add_parser(
            'binary',
            help='External tool management and pass-through execution',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=(
                'Install, validate, and manage external tools.\n\n'
                'Pass-through execution:\n'
                '  symfluence binary <tool> [args...]   Run a bundled binary directly\n\n'
                'Examples:\n'
                '  symfluence binary summa --version     Run SUMMA with --version\n'
                '  symfluence binary summa -m fm.txt     Run SUMMA with a file-manager\n'
                '  symfluence binary fuse.exe --help     Show FUSE help\n'
                '  symfluence binary pitremove ...       Run a TauDEM sub-tool'
            ),
        )
        binary_subparsers = binary_parser.add_subparsers(
            dest='action',
            required=True,
            help='Binary action',
            metavar='<action>'
        )

        # binary install
        install_parser = binary_subparsers.add_parser(
            'install',
            help='Install external tools'
        )
        _tools_help = (
            'Tools to install. If not specified, installs all default tools.\n'
            '\n'
            f'  Default ({len(DEFAULT_TOOLS)} tools — covers every process-based model '
            'in the paper Fig 4/Fig 8 ensemble):\n'
            f'    {", ".join(DEFAULT_TOOLS)}\n'
            f'  Experimental (explicit opt-in): {", ".join(EXPERIMENTAL_TOOLS)}\n'
            '\n'
            '  JAX re-implementations (HBV*, SACSMA*, XAJ*, HECHMS*, TOPMODEL*,\n'
            '  SUMMA+MOD) arrive automatically with "pip install symfluence"\n'
            '  and do not need a binary build.\n'
            '  LSTM is PyTorch-only; GR4J uses airGR via rpy2 (R optional dep).\n'
            '\n'
            '  Binaries install under SYMFLUENCE_DATA_DIR/installs/<tool>/bin/\n'
            '  (defaults to ./SYMFLUENCE_data/installs when SYMFLUENCE_DATA_DIR is unset).'
        )
        install_parser.add_argument('tools', nargs='*', metavar='TOOL',
                                  help=_tools_help)
        install_parser.add_argument('--force', action='store_true',
                                  help='Force reinstall even if already installed')
        install_parser.add_argument('--patched', action='store_true',
                                  help='Apply SYMFLUENCE patches (currently: RHESSys GW recharge + NaN guards)')
        install_parser.add_argument('--branch', type=str, default=None,
                                  help='Override the default git branch for the tool(s) being installed')
        install_parser.add_argument('--git-hash', type=str, default=None,
                                  help='Checkout a specific git commit hash after cloning')
        install_parser.set_defaults(func=BinaryCommands.install)

        # binary validate
        validate_parser = binary_subparsers.add_parser(
            'validate',
            help='Validate installed binaries'
        )
        validate_parser.add_argument('--verbose', action='store_true',
                                   help='Show detailed validation output')
        validate_parser.set_defaults(func=BinaryCommands.validate)

        # binary doctor
        doctor_parser = binary_subparsers.add_parser(
            'doctor',
            help='Run system diagnostics'
        )
        doctor_parser.set_defaults(func=BinaryCommands.doctor)

        # binary install-sysdeps
        install_sysdeps_parser = binary_subparsers.add_parser(
            'install-sysdeps',
            help='Install system dependencies (compilers, libraries)'
        )
        install_sysdeps_parser.add_argument(
            '--tool', type=str, default=None,
            help='Install deps for a specific tool only (e.g. summa, fuse)'
        )
        install_sysdeps_parser.add_argument(
            '--dry-run', action='store_true',
            help='Show install commands without executing them'
        )
        install_sysdeps_parser.set_defaults(func=BinaryCommands.install_sysdeps)

        # binary info
        info_parser = binary_subparsers.add_parser(
            'info',
            help='Display information about installed tools'
        )
        info_parser.set_defaults(func=BinaryCommands.info)

    def _register_config_commands(self, subparsers):
        """Register configuration management commands."""
        from .commands import ConfigCommands

        config_parser = subparsers.add_parser(
            'config',
            help='Configuration management',
            description='Manage and validate configuration files'
        )
        config_subparsers = config_parser.add_subparsers(
            dest='action',
            required=True,
            help='Config action',
            metavar='<action>'
        )

        # config list-templates
        list_templates_parser = config_subparsers.add_parser(
            'list-templates',
            help='List available configuration templates'
        )
        list_templates_parser.set_defaults(func=ConfigCommands.list_templates)

        # config update
        update_parser = config_subparsers.add_parser(
            'update',
            help='Update an existing configuration file'
        )
        update_parser.add_argument('config_file', type=str,
                                  help='Configuration file to update')
        update_parser.add_argument('--interactive', action='store_true',
                                  help='Interactive update mode')
        update_parser.set_defaults(func=ConfigCommands.update)

        # config validate
        validate_parser = config_subparsers.add_parser(
            'validate',
            help='Validate configuration file syntax'
        )
        validate_parser.set_defaults(func=ConfigCommands.validate)

        # config validate-env
        validate_env_parser = config_subparsers.add_parser(
            'validate-env',
            help='Validate system environment'
        )
        validate_env_parser.set_defaults(func=ConfigCommands.validate_env)

        # config resolve
        resolve_parser = config_subparsers.add_parser(
            'resolve',
            help='Show fully resolved configuration after merging all sources',
            parents=[self.common_parser]
        )
        resolve_parser.add_argument('--flat', action='store_true',
                                    help='Show flat UPPERCASE keys (what models see)')
        resolve_parser.add_argument('--json', action='store_true', dest='as_json',
                                    help='Output as JSON (machine-readable)')
        resolve_parser.add_argument('--diff', action='store_true',
                                    help='Only show values that differ from defaults')
        resolve_parser.add_argument('--section', type=str, choices=[
                                    'system', 'domain', 'data', 'forcing',
                                    'model', 'optimization', 'evaluation', 'paths'],
                                    help='Show only a specific config section')
        resolve_parser.set_defaults(func=ConfigCommands.resolve)

    def _register_job_commands(self, subparsers):
        """Register SLURM job submission commands."""
        from .commands import JobCommands

        job_parser = subparsers.add_parser(
            'job',
            help='SLURM job submission',
            description='Submit workflow commands as SLURM jobs'
        )
        job_subparsers = job_parser.add_subparsers(
            dest='action',
            required=True,
            help='Job action',
            metavar='<action>'
        )

        # job submit
        submit_parser = job_subparsers.add_parser(
            'submit',
            help='Submit workflow as SLURM job'
        )
        submit_parser.add_argument('--name', dest='job_name', type=str,
                                  help='SLURM job name')
        submit_parser.add_argument('--time', dest='job_time', type=str, default='48:00:00',
                                  help='Time limit (default: 48:00:00)')
        submit_parser.add_argument('--nodes', dest='job_nodes', type=int, default=1,
                                  help='Number of nodes (default: 1)')
        submit_parser.add_argument('--tasks', dest='job_ntasks', type=int, default=1,
                                  help='Number of tasks (default: 1)')
        submit_parser.add_argument('--memory', dest='job_memory', type=str, default='50G',
                                  help='Memory requirement (default: 50G)')
        submit_parser.add_argument('--account', dest='job_account', type=str,
                                  help='Account to charge')
        submit_parser.add_argument('--partition', dest='job_partition', type=str,
                                  help='Partition/queue name')
        submit_parser.add_argument('--modules', dest='job_modules', type=str, default='symfluence_modules',
                                  help='Module to restore (default: symfluence_modules)')
        submit_parser.add_argument('--conda-env', dest='conda_env', type=str, default='symfluence',
                                  help='Conda environment (default: symfluence)')
        submit_parser.add_argument('--wait', dest='submit_and_wait', action='store_true',
                                  help='Submit and monitor job until completion')
        submit_parser.add_argument('--template', dest='slurm_template', type=str,
                                  help='Custom SLURM template file')
        submit_parser.add_argument('workflow_args', nargs=argparse.REMAINDER,
                                  help='Workflow command and arguments to submit')
        submit_parser.set_defaults(func=JobCommands.submit)

    def _register_example_commands(self, subparsers):
        """Register example notebook commands."""
        from .commands import ExampleCommands

        example_parser = subparsers.add_parser(
            'example',
            help='Example notebooks',
            description='Launch and manage example Jupyter notebooks'
        )
        example_subparsers = example_parser.add_subparsers(
            dest='action',
            required=True,
            help='Example action',
            metavar='<action>'
        )

        # example launch
        launch_parser = example_subparsers.add_parser(
            'launch',
            help='Launch an example notebook'
        )
        launch_parser.add_argument('example_id', type=str,
                                  help='Example ID (e.g., 1a, 2b, 3c)')
        launch_parser.add_argument('--lab', action='store_true',
                                  help='Launch in JupyterLab (default)')
        launch_parser.add_argument('--notebook', action='store_true',
                                  help='Launch in classic Jupyter Notebook')
        launch_parser.set_defaults(func=ExampleCommands.launch)

        # example list
        list_parser = example_subparsers.add_parser(
            'list',
            help='List available example notebooks'
        )
        list_parser.set_defaults(func=ExampleCommands.list_examples)

    def _register_agent_commands(self, subparsers):
        """Register AI agent commands."""
        from .commands import AgentCommands

        agent_parser = subparsers.add_parser(
            'agent',
            help='AI agent interface',
            description='Interactive AI agent for SYMFLUENCE'
        )
        agent_subparsers = agent_parser.add_subparsers(
            dest='action',
            required=True,
            help='Agent action',
            metavar='<action>'
        )

        # agent start
        start_parser = agent_subparsers.add_parser(
            'start',
            help='Start interactive agent mode'
        )
        start_parser.add_argument('--verbose', action='store_true',
                                help='Show verbose agent output')
        start_parser.set_defaults(func=AgentCommands.start)

        # agent run
        run_parser = agent_subparsers.add_parser(
            'run',
            help='Execute a single agent prompt'
        )
        run_parser.add_argument('prompt', type=str,
                              help='Prompt to execute')
        run_parser.add_argument('--verbose', action='store_true',
                              help='Show verbose agent output')
        run_parser.set_defaults(func=AgentCommands.run)

    def _register_gui_commands(self, subparsers):
        """Register GUI launch commands."""
        from .commands import GUICommands

        gui_parser = subparsers.add_parser(
            'gui',
            help='Graphical user interface',
            description='Launch the SYMFLUENCE web-based GUI'
        )
        gui_subparsers = gui_parser.add_subparsers(
            dest='action',
            required=True,
            help='GUI action',
            metavar='<action>'
        )

        # gui launch
        launch_parser = gui_subparsers.add_parser(
            'launch',
            help='Launch the Panel web GUI',
            parents=[self.common_parser]
        )
        launch_parser.add_argument('--port', type=int, default=5006,
                                   help='Server port (default: 5006)')
        launch_parser.add_argument('--no-browser', action='store_true', dest='no_browser',
                                   help='Do not auto-open a browser tab')
        launch_parser.add_argument('--demo', type=str, default=None, metavar='NAME',
                                   help='Load a built-in demo (e.g. "bow" for Bow at Banff)')
        launch_parser.set_defaults(func=GUICommands.launch)

    def _register_tui_commands(self, subparsers):
        """Register TUI interactive terminal commands."""
        from .commands import TUICommands

        tui_parser = subparsers.add_parser(
            'tui',
            help='Interactive terminal user interface',
            description='Launch the SYMFLUENCE interactive terminal UI'
        )
        tui_subparsers = tui_parser.add_subparsers(
            dest='action',
            required=True,
            help='TUI action',
            metavar='<action>'
        )

        # tui launch
        launch_parser = tui_subparsers.add_parser(
            'launch',
            help='Launch the interactive terminal UI',
            parents=[self.common_parser]
        )
        launch_parser.add_argument('--demo', type=str, default=None, metavar='NAME',
                                   help='Load a built-in demo (e.g. "bow" for Bow at Banff)')
        launch_parser.set_defaults(func=TUICommands.launch)

    def _register_data_commands(self, subparsers):
        """Register standalone data acquisition commands."""
        from .commands import DataCommands

        data_parser = subparsers.add_parser(
            'data',
            help='Standalone data acquisition',
            description='Download, list, and inspect acquisition datasets'
        )
        data_subparsers = data_parser.add_subparsers(
            dest='action',
            required=True,
            help='Data action',
            metavar='<action>'
        )

        # data download
        download_parser = data_subparsers.add_parser(
            'download',
            help='Download a dataset',
            parents=[self.common_parser]
        )
        download_parser.add_argument('dataset', type=str, metavar='DATASET',
                                     help='Dataset name (e.g. modis_lai, era5, grace)')
        download_parser.add_argument('--bbox', type=str, default=None,
                                     metavar='LAT_MAX/LON_MIN/LAT_MIN/LON_MAX',
                                     help='Bounding box as N/W/S/E (required unless --config)')
        download_parser.add_argument('--shapefile', type=str, default=None,
                                     metavar='PATH',
                                     help='Shapefile to extract bounding box from (alternative to --bbox)')
        download_parser.add_argument('--start', type=str, default=None,
                                     metavar='YYYY-MM-DD',
                                     help='Start date (required unless --config)')
        download_parser.add_argument('--end', type=str, default=None,
                                     metavar='YYYY-MM-DD',
                                     help='End date (required unless --config)')
        download_parser.add_argument('--output', type=str, default=None,
                                     metavar='PATH',
                                     help='Output directory (default: ./data/<dataset>)')
        download_parser.add_argument('--domain', dest='domain_name', type=str,
                                     default='standalone',
                                     help='Domain name for file naming (default: standalone)')
        download_parser.add_argument('--force', action='store_true', default=False,
                                     help='Force re-download existing data')
        download_parser.add_argument('--vars', type=str, default=None,
                                     metavar='VAR1,VAR2,...',
                                     help='Comma-separated variables to download (e.g. tmax,tmin,prcp)')
        download_parser.add_argument('--extra', action='append', default=None,
                                     metavar='KEY=VALUE',
                                     help='Extra config keys (repeatable)')
        download_parser.set_defaults(func=DataCommands.download)

        # data list
        list_parser = data_subparsers.add_parser(
            'list',
            help='List available datasets'
        )
        list_parser.set_defaults(func=DataCommands.list_datasets)

        # data info
        info_parser = data_subparsers.add_parser(
            'info',
            help='Show info about a dataset'
        )
        info_parser.add_argument('dataset', type=str, metavar='DATASET',
                                 help='Dataset name to show info about')
        info_parser.set_defaults(func=DataCommands.info)

    def _register_doctor_commands(self, subparsers):
        """Register top-level doctor command."""
        from .commands import DoctorCommands

        doctor_parser = subparsers.add_parser(
            'doctor',
            help='Run system diagnostics',
            description='Comprehensive environment, path resolution, and binary diagnostics'
        )
        doctor_parser.set_defaults(func=DoctorCommands.doctor, action='doctor')

    def _register_fews_commands(self, subparsers):
        """Register Delft-FEWS adapter commands."""
        from .commands import FEWSCommands

        fews_parser = subparsers.add_parser(
            'fews',
            help='Delft-FEWS adapter operations',
            description='Run FEWS General Adapter pre/post processing and launch openFEWS'
        )
        fews_subparsers = fews_parser.add_subparsers(
            dest='action',
            required=True,
            help='FEWS action',
            metavar='<action>'
        )

        # fews pre
        pre_parser = fews_subparsers.add_parser(
            'pre',
            help='Run FEWS pre-adapter (import forcing, generate config)',
            parents=[self.common_parser]
        )
        pre_parser.add_argument('--run-info', dest='run_info', type=str, required=True,
                                help='Path to run_info.xml')
        pre_parser.add_argument('--format', choices=['pi-xml', 'netcdf-cf'],
                                default='netcdf-cf',
                                help='Data exchange format (default: netcdf-cf)')
        pre_parser.add_argument('--id-map', dest='id_map', type=str, default=None,
                                help='Path to variable ID mapping YAML file')
        pre_parser.set_defaults(func=FEWSCommands.pre)

        # fews post
        post_parser = fews_subparsers.add_parser(
            'post',
            help='Run FEWS post-adapter (export results, write diagnostics)',
            parents=[self.common_parser]
        )
        post_parser.add_argument('--run-info', dest='run_info', type=str, required=True,
                                 help='Path to run_info.xml')
        post_parser.add_argument('--format', choices=['pi-xml', 'netcdf-cf'],
                                 default='netcdf-cf',
                                 help='Data exchange format (default: netcdf-cf)')
        post_parser.add_argument('--id-map', dest='id_map', type=str, default=None,
                                 help='Path to variable ID mapping YAML file')
        post_parser.set_defaults(func=FEWSCommands.post)

        # fews run
        run_parser = fews_subparsers.add_parser(
            'run',
            help='Run full FEWS adapter cycle (pre -> model -> post)',
            parents=[self.common_parser]
        )
        run_parser.add_argument('--run-info', dest='run_info', type=str, required=True,
                                help='Path to run_info.xml')
        run_parser.add_argument('--format', choices=['pi-xml', 'netcdf-cf'],
                                default='netcdf-cf',
                                help='Data exchange format (default: netcdf-cf)')
        run_parser.add_argument('--id-map', dest='id_map', type=str, default=None,
                                help='Path to variable ID mapping YAML file')
        run_parser.set_defaults(func=FEWSCommands.run_full)

        # fews launch
        launch_parser = fews_subparsers.add_parser(
            'launch',
            help='Launch openFEWS with SYMFLUENCE adapter support',
            parents=[self.common_parser]
        )
        launch_parser.add_argument('--port', type=int, default=8080,
                                   help='Server port for openFEWS (default: 8080)')
        launch_parser.add_argument('--no-browser', action='store_true', dest='no_browser',
                                   help='Do not auto-open a browser')
        launch_parser.set_defaults(func=FEWSCommands.launch)

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """
        Parse command-line arguments.

        Args:
            args: List of argument strings (for testing). If None, uses sys.argv.

        Returns:
            Parsed arguments namespace
        """
        return self.parser.parse_args(args)
