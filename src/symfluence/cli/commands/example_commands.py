# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Example notebook command handlers for SYMFLUENCE CLI.

This module implements handlers for launching and managing example Jupyter notebooks.
"""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from ..exit_codes import ExitCode
from .base import BaseCommand, cli_exception_handler, supports_dry_run


class ExampleCommands(BaseCommand):
    """Handlers for example notebook commands."""

    @staticmethod
    @cli_exception_handler
    def launch(args: Namespace) -> int:
        """
        Execute: symfluence example launch EXAMPLE_ID

        Launch an example Jupyter notebook.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        from symfluence.cli.services import NotebookService

        notebook_service = NotebookService()

        example_id = args.example_id
        prefer_lab = not BaseCommand.get_arg(args, 'notebook', False)  # Default to lab unless --notebook specified

        BaseCommand._console.info(f"Launching example notebook: {example_id}")
        if prefer_lab:
            BaseCommand._console.indent("Using JupyterLab")
        else:
            BaseCommand._console.indent("Using classic Jupyter Notebook")

        # Launch the notebook
        success = notebook_service.launch_example_notebook(
            example_id=example_id,
            prefer_lab=prefer_lab
        )

        if success == 0:
            return ExitCode.SUCCESS
        else:
            BaseCommand._console.error("Failed to launch notebook")
            return ExitCode.GENERAL_ERROR

    @staticmethod
    @supports_dry_run
    @cli_exception_handler
    def list_examples(args: Namespace) -> int:
        """
        Execute: symfluence example list

        List available example notebooks.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        # Look for example notebooks
        examples_dir = Path(__file__).parent.parent.parent.parent.parent / 'examples'

        BaseCommand._console.info("Available example notebooks:")
        BaseCommand._console.rule()

        if examples_dir.exists():
            # Find all notebook files
            notebooks = sorted(
                nb for nb in examples_dir.rglob('*.ipynb')
                if '.ipynb_checkpoints' not in nb.parts
            )

            if notebooks:
                for i, notebook in enumerate(notebooks, 1):
                    # Extract example ID from path
                    rel_path = notebook.relative_to(examples_dir)
                    BaseCommand._console.info(f"{i:2}. {rel_path}")
                BaseCommand._console.rule()
                BaseCommand._console.info(f"Total: {len(notebooks)} notebooks")
            else:
                BaseCommand._console.info("No example notebooks found")
        else:
            BaseCommand._console.info(f"Examples directory not found: {examples_dir}")

        return ExitCode.SUCCESS
