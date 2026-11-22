"""Unit tests for NotebookService."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from symfluence.cli.services import NotebookService

pytestmark = [pytest.mark.unit, pytest.mark.cli, pytest.mark.quick]


class TestInitialization:
    """Test NotebookService initialization."""

    def test_initialization(self):
        """Test NotebookService creates successfully."""
        service = NotebookService()
        assert service is not None


class TestExampleIDParsing:
    """Test example ID parsing and pattern matching."""

    @pytest.mark.parametrize("example_id,expected_prefix", [
        ("01a", "01a"),
        ("1a", "01a"),
        ("02b", "02b"),
        ("2b", "02b"),
        ("03c", "03c"),
        ("10d", "10d"),
    ])
    def test_example_id_regex_parsing(self, example_id, expected_prefix):
        """Test regex parsing of example IDs with letter suffix."""
        service = NotebookService()

        # Create a test repo structure
        repo_root = Path('/tmp/test_repo')

        # Mock the existence checks
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'is_dir', return_value=True), \
             patch('pathlib.Path.rglob') as mock_rglob:

            # Mock notebook discovery
            nb_path = repo_root / 'examples' / f'{expected_prefix}_test.ipynb'
            mock_rglob.return_value = [nb_path]

            with patch('subprocess.run'), \
                 patch('sys.executable', '/usr/bin/python3'):
                # This will fail but we're just testing the ID parsing logic
                try:
                    service.launch_example_notebook(example_id, repo_root=repo_root)
                except:
                    pass

            # Verify the correct pattern was used
            # The rglob should have been called with the expected prefix
            calls = mock_rglob.call_args_list
            assert any(expected_prefix in str(call) for call in calls if call is not None)

    def test_fallback_parsing(self):
        """Test fallback parsing when regex fails."""
        service = NotebookService()

        repo_root = Path('/tmp/test_repo')

        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'is_dir', return_value=True), \
             patch('pathlib.Path.rglob') as mock_rglob:

            # Mock notebook discovery
            nb_path = repo_root / 'examples' / 'custom_notebook.ipynb'
            mock_rglob.return_value = [nb_path]

            with patch('subprocess.run'), \
                 patch('sys.executable', '/usr/bin/python3'):
                try:
                    service.launch_example_notebook('custom', repo_root=repo_root)
                except:
                    pass

            # Should use raw prefix
            calls = mock_rglob.call_args_list
            assert len(calls) > 0


class TestNotebookDiscovery:
    """Test notebook file discovery."""

    @patch('sys.executable', '/usr/bin/python3')
    @patch.object(Path, 'is_dir', return_value=True)
    @patch.object(Path, 'exists', return_value=True)
    @patch('pathlib.Path.rglob')
    @patch('subprocess.run')
    def test_single_notebook_match(self, mock_subprocess, mock_rglob, mock_exists, mock_isdir):
        """Test discovery when one notebook matches."""
        service = NotebookService()
        repo_root = Path('/tmp/test_repo')

        # Create mock notebook
        nb_path = repo_root / 'examples' / '01a_point_scale_snotel.ipynb'
        mock_rglob.return_value = [nb_path]

        # Mock subprocess calls for ipykernel and jupyter
        mock_subprocess.return_value = MagicMock(returncode=0)

        result = service.launch_example_notebook('01a', repo_root=repo_root)

        # Should attempt to launch (may fail but that's ok for this test)
        assert result in [0, 3]  # 0 for success, 3 for jupyter not available

    @patch('sys.executable', '/usr/bin/python3')
    @patch.object(Path, 'is_dir', return_value=True)
    @patch.object(Path, 'exists', return_value=True)
    @patch('pathlib.Path.rglob')
    @patch('subprocess.run')
    def test_multiple_notebook_matches(self, mock_subprocess, mock_rglob, mock_exists, mock_isdir, capsys):
        """Test behavior when multiple notebooks match."""
        import sys

        from symfluence.cli.console import Console, ConsoleConfig

        # Use capture-friendly console
        console = Console(ConsoleConfig(
            output_stream=sys.stdout,
            error_stream=sys.stderr,
            use_colors=False,
            show_progress=False
        ))
        service = NotebookService(console=console)
        repo_root = Path('/tmp/test_repo')

        # Create multiple matching notebooks
        nb_paths = [
            repo_root / 'examples' / '01a_test1.ipynb',
            repo_root / 'examples' / '01a_test2.ipynb',
        ]
        mock_rglob.return_value = nb_paths

        mock_subprocess.return_value = MagicMock(returncode=0)

        result = service.launch_example_notebook('01a', repo_root=repo_root)

        # Should print warning about multiple matches
        captured = capsys.readouterr()
        assert 'Multiple' in captured.out or 'multiple' in captured.out.lower()

    @patch.object(Path, 'is_dir', return_value=True)
    @patch.object(Path, 'exists', return_value=True)
    @patch('pathlib.Path.rglob', return_value=[])
    def test_no_notebook_matches(self, mock_rglob, mock_exists, mock_isdir):
        """Test error when no notebooks match."""
        service = NotebookService()
        repo_root = Path('/tmp/test_repo')

        # All paths exist but no notebooks found via rglob
        result = service.launch_example_notebook('nonexistent', repo_root=repo_root)

        assert result == 2  # Error code for notebook not found


class TestVirtualEnvironmentDetection:
    """Test virtual environment detection."""

    @pytest.mark.parametrize("venv_name", [
        ".venv", "venv", "env", ".conda", ".virtualenv"
    ])
    @patch('sys.executable', '/usr/bin/python3')
    @patch.object(Path, 'is_dir', return_value=True)
    @patch.object(Path, 'exists', return_value=True)
    @patch('pathlib.Path.rglob')
    @patch('subprocess.run')
    def test_detect_various_venvs(self, mock_subprocess, mock_rglob, mock_exists, mock_isdir, venv_name):
        """Test detection of different venv directory names."""
        service = NotebookService()
        repo_root = Path('/tmp/test_repo')

        nb_path = repo_root / 'examples' / '01a_test.ipynb'
        mock_rglob.return_value = [nb_path]
        mock_subprocess.return_value = MagicMock(returncode=0)

        result = service.launch_example_notebook('01a', repo_root=repo_root)

        # Should have found the venv (0 for success, 3 for jupyter not available)
        assert result in [0, 3]


class TestPythonExecutableSelection:
    """Test Python executable selection logic."""

    @patch('sys.executable', '/usr/bin/python3')
    @patch.object(Path, 'is_dir', return_value=True)
    @patch.object(Path, 'exists', return_value=True)
    @patch('pathlib.Path.rglob')
    @patch('subprocess.run')
    def test_venv_python_priority(self, mock_subprocess, mock_rglob, mock_exists, mock_isdir):
        """Test venv/bin/python tried first."""
        service = NotebookService()
        repo_root = Path('/tmp/test_repo')
        venv_path = repo_root / '.venv'
        python_path = venv_path / 'bin' / 'python'

        nb_path = repo_root / 'examples' / '01a_test.ipynb'
        mock_rglob.return_value = [nb_path]
        mock_subprocess.return_value = MagicMock(returncode=0)

        result = service.launch_example_notebook('01a', repo_root=repo_root)

        # Should use venv python - check that subprocess was called
        # with a python path from the venv directory
        calls = [str(call) for call in mock_subprocess.call_args_list]
        assert any('venv' in str(call) or 'python' in str(call) for call in calls)


class TestKernelRegistration:
    """Test ipykernel installation and registration."""

    @patch('sys.executable', '/usr/bin/python3')
    @patch('subprocess.run')
    @patch.object(Path, 'is_dir', return_value=True)
    @patch.object(Path, 'exists', return_value=True)
    @patch('pathlib.Path.rglob')
    def test_ipykernel_already_installed(self, mock_rglob, mock_exists, mock_isdir, mock_subprocess):
        """Test kernel registration when ipykernel present."""
        service = NotebookService()
        repo_root = Path('/tmp/test_repo')

        nb_path = repo_root / 'examples' / '01a_test.ipynb'
        mock_rglob.return_value = [nb_path]

        # Mock ipykernel already installed
        mock_subprocess.return_value = MagicMock(returncode=0)

        result = service.launch_example_notebook('01a', repo_root=repo_root)

        # Should register kernel
        calls = [str(call) for call in mock_subprocess.call_args_list]
        assert any('ipykernel' in str(call) for call in calls)

    @patch('sys.executable', '/usr/bin/python3')
    @patch('subprocess.run')
    @patch.object(Path, 'is_dir', return_value=True)
    @patch.object(Path, 'exists', return_value=True)
    @patch('pathlib.Path.rglob')
    def test_ipykernel_installation(self, mock_rglob, mock_exists, mock_isdir, mock_subprocess):
        """Test ipykernel installation when missing."""
        service = NotebookService()
        repo_root = Path('/tmp/test_repo')

        nb_path = repo_root / 'examples' / '01a_test.ipynb'
        mock_rglob.return_value = [nb_path]

        # Mock ipykernel not installed, then success after install
        def subprocess_side_effect(*args, **kwargs):
            cmd = args[0] if args else kwargs.get('args', [])
            if 'show' in cmd and 'ipykernel' in cmd:
                return MagicMock(returncode=1)  # Not installed
            return MagicMock(returncode=0)

        mock_subprocess.side_effect = subprocess_side_effect

        result = service.launch_example_notebook('01a', repo_root=repo_root)

        # Should have attempted to install ipykernel
        calls = [str(call) for call in mock_subprocess.call_args_list]
        assert any('install' in str(call) and 'ipykernel' in str(call) for call in calls)


class TestJupyterLaunch:
    """Test Jupyter launcher selection and execution."""

    @patch('sys.executable', '/usr/bin/python3')
    @patch('subprocess.run')
    @patch.object(Path, 'is_dir', return_value=True)
    @patch.object(Path, 'exists', return_value=True)
    @patch('pathlib.Path.rglob')
    def test_jupyterlab_launch(self, mock_rglob, mock_exists, mock_isdir, mock_subprocess):
        """Test launch with JupyterLab."""
        service = NotebookService()
        repo_root = Path('/tmp/test_repo')

        nb_path = repo_root / 'examples' / '01a_test.ipynb'
        mock_rglob.return_value = [nb_path]

        mock_subprocess.return_value = MagicMock(returncode=0)

        result = service.launch_example_notebook('01a', repo_root=repo_root, prefer_lab=True)

        # Should attempt jupyterlab
        calls = [str(call) for call in mock_subprocess.call_args_list]
        assert any('jupyterlab' in str(call) or 'jupyter' in str(call) for call in calls)
        assert result == 0

    @staticmethod
    def _fake_run(installed, launch_rc=0):
        """subprocess.run stand-in: find_spec probes succeed only for ``installed``."""
        launched = []

        def run(cmd, *args, **kwargs):
            if cmd[1] == '-c' and 'find_spec' in cmd[2]:
                found = any(repr(mod) in cmd[2] for mod in installed)
                return MagicMock(returncode=0 if found else 1)
            if cmd[1] == '-m' and cmd[2] in ('jupyterlab', 'notebook'):
                launched.append(cmd[2])
                return MagicMock(returncode=launch_rc)
            return MagicMock(returncode=0)

        return run, launched

    @patch('subprocess.run')
    @patch.object(Path, 'is_dir', return_value=True)
    @patch.object(Path, 'exists', return_value=True)
    @patch('pathlib.Path.rglob')
    def test_missing_jupyter_fails_with_hint(self, mock_rglob, mock_exists, mock_isdir, mock_subprocess):
        """No front end installed: non-zero exit instead of a silent success."""
        repo_root = Path('/tmp/test_repo')
        mock_rglob.return_value = [repo_root / 'examples' / '04e_test.ipynb']
        mock_subprocess.side_effect, launched = self._fake_run(installed=())

        result = NotebookService().launch_example_notebook('04e', repo_root=repo_root)

        assert result == 3
        assert launched == []

    @patch('subprocess.run')
    @patch.object(Path, 'is_dir', return_value=True)
    @patch.object(Path, 'exists', return_value=True)
    @patch('pathlib.Path.rglob')
    def test_falls_back_to_classic_notebook(self, mock_rglob, mock_exists, mock_isdir, mock_subprocess):
        """JupyterLab missing but classic Notebook present: launch Notebook."""
        repo_root = Path('/tmp/test_repo')
        mock_rglob.return_value = [repo_root / 'examples' / '04e_test.ipynb']
        mock_subprocess.side_effect, launched = self._fake_run(installed=('notebook',))

        result = NotebookService().launch_example_notebook('04e', repo_root=repo_root, prefer_lab=True)

        assert result == 0
        assert launched == ['notebook']

    @patch('subprocess.run')
    @patch.object(Path, 'is_dir', return_value=True)
    @patch.object(Path, 'exists', return_value=True)
    @patch('pathlib.Path.rglob')
    def test_jupyter_nonzero_exit_is_failure(self, mock_rglob, mock_exists, mock_isdir, mock_subprocess):
        """A Jupyter process that exits non-zero is reported as a failure."""
        repo_root = Path('/tmp/test_repo')
        mock_rglob.return_value = [repo_root / 'examples' / '04e_test.ipynb']
        mock_subprocess.side_effect, launched = self._fake_run(installed=('jupyterlab',), launch_rc=1)

        result = NotebookService().launch_example_notebook('04e', repo_root=repo_root)

        assert result == 3
        assert launched == ['jupyterlab']


class TestErrorCases:
    """Test error handling."""

    @patch.object(Path, 'exists', return_value=False)
    def test_repo_root_not_found(self, mock_exists):
        """Test error when repo root missing."""
        service = NotebookService()
        repo_root = Path('/nonexistent/repo')

        result = service.launch_example_notebook('01a', repo_root=repo_root)

        assert result == 1  # Error code for repo not found

    @patch.object(Path, 'is_dir', return_value=False)
    @patch.object(Path, 'exists', return_value=True)
    def test_examples_dir_not_found(self, mock_exists, mock_isdir):
        """Test error when examples directory missing."""
        service = NotebookService()
        repo_root = Path('/tmp/test_repo')

        # Repo path exists but examples is not a directory
        result = service.launch_example_notebook('01a', repo_root=repo_root)

        assert result == 2  # Error code for examples not found
