"""
Unit tests for WorkflowOrchestrator.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from symfluence.core.config.models import SymfluenceConfig
from symfluence.core.stage_marker import read_marker, write_marker
from symfluence.project.workflow_orchestrator import WorkflowOrchestrator, WorkflowStep


class TestWorkflowOrchestrator:
    @pytest.fixture
    def mock_managers(self):
        return {
            'project': MagicMock(),
            'domain': MagicMock(),
            'data': MagicMock(),
            'model': MagicMock(),
            'analysis': MagicMock(),
            'optimization': MagicMock()
        }

    @pytest.fixture
    def config(self):
        return SymfluenceConfig.from_minimal(
            domain_name='test_domain',
            model='SUMMA',
            time_start='2010-01-01 00:00',
            time_end='2010-12-31 23:00',
            SYMFLUENCE_DATA_DIR='/tmp/data',
        )

    @pytest.fixture
    def logger(self):
        return MagicMock()

    @pytest.fixture
    def orchestrator(self, mock_managers, config, logger):
        return WorkflowOrchestrator(mock_managers, config, logger)

    def test_run_individual_steps_success(self, orchestrator, mock_managers):
        """Test executing specific steps successfully."""
        # Setup mock steps
        with patch.object(orchestrator, 'define_workflow_steps') as mock_define:
            step1 = WorkflowStep("step1", "cli1", MagicMock(), lambda: True, "desc1")
            step2 = WorkflowStep("step2", "cli2", MagicMock(), lambda: True, "desc2")
            mock_define.return_value = [step1, step2]

            results = orchestrator.run_individual_steps(["cli1", "cli2"])

            assert len(results) == 2
            assert results[0]["success"] is True
            assert results[1]["success"] is True
            step1.func.assert_called_once()
            step2.func.assert_called_once()

    def test_run_individual_steps_unrecognized(self, orchestrator):
        """Test handling of unrecognized step names."""
        with patch.object(orchestrator, 'define_workflow_steps') as mock_define:
            mock_define.return_value = []

            with pytest.raises(ValueError, match="Step 'unknown' not recognized"):
                orchestrator.run_individual_steps(["unknown"])

    def test_run_individual_steps_unrecognized_continue_on_error(self, orchestrator):
        """Test unrecognized step recorded as failure when continue_on_error=True."""
        with patch.object(orchestrator, 'define_workflow_steps') as mock_define:
            mock_define.return_value = []

            results = orchestrator.run_individual_steps(["unknown"], continue_on_error=True)

            assert len(results) == 1
            assert results[0]["success"] is False
            assert "not recognized" in results[0]["error"]

    def test_run_individual_steps_failure(self, orchestrator):
        """Test step failure behavior."""
        with patch.object(orchestrator, 'define_workflow_steps') as mock_define:
            def failing_func():
                raise ValueError("Step failed")

            step1 = WorkflowStep("fail", "fail_cli", failing_func, lambda: False, "fail_desc")
            mock_define.return_value = [step1]

            with pytest.raises(ValueError, match="Step failed"):
                orchestrator.run_individual_steps(["fail_cli"])

    def test_run_individual_steps_continue_on_error(self, orchestrator):
        """Test continue_on_error parameter."""
        with patch.object(orchestrator, 'define_workflow_steps') as mock_define:
            def failing_func():
                raise ValueError("Step failed")

            step1 = WorkflowStep("fail", "fail_cli", failing_func, lambda: False, "fail_desc")
            step2 = WorkflowStep("success", "ok_cli", MagicMock(), lambda: True, "ok_desc")
            mock_define.return_value = [step1, step2]

            results = orchestrator.run_individual_steps(["fail_cli", "ok_cli"], continue_on_error=True)

            assert len(results) == 2
            assert results[0]["success"] is False
            assert results[1]["success"] is True
            step2.func.assert_called_once()

    def _build_orchestrator_for_observation_checks(
        self,
        tmp_path: Path,
        *,
        evaluation_data: str
    ) -> WorkflowOrchestrator:
        """Create an orchestrator rooted at tmp_path for observation output checks."""
        managers = {
            'project': MagicMock(),
            'domain': MagicMock(),
            'data': MagicMock(),
            'model': MagicMock(),
            'analysis': MagicMock(),
            'optimization': MagicMock(),
        }
        config = SymfluenceConfig.from_minimal(
            domain_name='obs_domain',
            model='SUMMA',
            time_start='2010-01-01 00:00',
            time_end='2010-01-02 00:00',
            SYMFLUENCE_DATA_DIR=tmp_path,
            EVALUATION_DATA=evaluation_data,
        )
        return WorkflowOrchestrator(managers, config, MagicMock())

    @staticmethod
    def _touch_streamflow_output(project_dir: Path, domain_name: str) -> None:
        path = project_dir / "observations" / "streamflow" / "preprocessed" / f"{domain_name}_streamflow_processed.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("datetime,streamflow\n2010-01-01,1.0\n", encoding="utf-8")

    @staticmethod
    def _touch_snow_output(project_dir: Path, domain_name: str) -> None:
        path = project_dir / "observations" / "snow" / "preprocessed" / f"{domain_name}_snow_processed.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("datetime,snow\n2010-01-01,1.0\n", encoding="utf-8")

    def test_check_observed_data_requires_all_requested_families(self, tmp_path):
        """
        When multiple observation families are requested, all must exist.
        """
        orchestrator = self._build_orchestrator_for_observation_checks(
            tmp_path,
            evaluation_data="streamflow,swe",
        )

        # Only streamflow exists -> should still be incomplete.
        self._touch_streamflow_output(orchestrator.project_dir, orchestrator.domain_name)
        assert orchestrator._check_observed_data_exists() is False

    def test_check_observed_data_true_when_all_requested_families_exist(self, tmp_path):
        """Requested families are complete only when each required output exists."""
        orchestrator = self._build_orchestrator_for_observation_checks(
            tmp_path,
            evaluation_data="streamflow,swe",
        )

        self._touch_streamflow_output(orchestrator.project_dir, orchestrator.domain_name)
        self._touch_snow_output(orchestrator.project_dir, orchestrator.domain_name)

        assert orchestrator._check_observed_data_exists() is True

    def test_check_observed_data_default_behavior_streamflow_only(self, tmp_path):
        """
        With no explicit observation request, fallback still checks streamflow.
        """
        orchestrator = self._build_orchestrator_for_observation_checks(
            tmp_path,
            evaluation_data="",
        )
        assert orchestrator._check_observed_data_exists() is False

        self._touch_streamflow_output(orchestrator.project_dir, orchestrator.domain_name)
        assert orchestrator._check_observed_data_exists() is True


class TestWorkflowOrchestratorMarkers:
    """Tests for configuration-hash stage invalidation in the orchestrator."""

    @staticmethod
    def _make_orchestrator(tmp_path, **extra_config):
        managers = {
            'project': MagicMock(),
            'domain': MagicMock(),
            'data': MagicMock(),
            'model': MagicMock(),
            'analysis': MagicMock(),
            'optimization': MagicMock(),
        }
        config = SymfluenceConfig.from_minimal(
            domain_name='marker_test',
            model='SUMMA',
            time_start='2010-01-01 00:00',
            time_end='2010-12-31 23:00',
            SYMFLUENCE_DATA_DIR=tmp_path,
            **extra_config,
        )
        return WorkflowOrchestrator(managers, config, MagicMock())

    def test_run_workflow_writes_markers(self, tmp_path):
        """Markers are written after each successfully executed step."""
        orch = self._make_orchestrator(tmp_path)

        with patch.object(orch, 'define_workflow_steps') as mock_define:
            step = WorkflowStep(
                "setup_project", "setup_project", MagicMock(),
                lambda: False, "Setup"
            )
            mock_define.return_value = [step]

            orch.run_workflow()

        marker = read_marker(orch.project_dir, "setup_project")
        assert marker is not None
        assert marker.stage == "setup_project"
        step.func.assert_called_once()

    def test_run_workflow_skips_when_marker_valid(self, tmp_path):
        """Step is skipped when output exists AND marker hash matches."""
        orch = self._make_orchestrator(tmp_path)

        with patch.object(orch, 'define_workflow_steps') as mock_define:
            func_mock = MagicMock()
            step = WorkflowStep(
                "setup_project", "setup_project", func_mock,
                lambda: True, "Setup"  # output exists
            )
            mock_define.return_value = [step]

            # Pre-write a marker with the correct hash. compute_stage_hash is the
            # canonical keying for a stage: it applies the stage's section list
            # *and* its experiment-scoped exclusions.
            from symfluence.core.stage_marker import compute_stage_hash
            current_hash = compute_stage_hash(orch._config, "setup_project")
            write_marker(orch.project_dir, "setup_project", current_hash)

            orch.run_workflow()

        func_mock.assert_not_called()

    def test_run_workflow_reruns_when_marker_stale(self, tmp_path):
        """Step re-executes when output exists but config hash changed."""
        orch = self._make_orchestrator(tmp_path)

        with patch.object(orch, 'define_workflow_steps') as mock_define:
            func_mock = MagicMock()
            step = WorkflowStep(
                "setup_project", "setup_project", func_mock,
                lambda: True, "Setup"  # output exists
            )
            mock_define.return_value = [step]

            # Pre-write a marker with a WRONG hash
            write_marker(orch.project_dir, "setup_project", "stale_hash_value")

            orch.run_workflow()

        func_mock.assert_called_once()

    def test_force_run_clears_markers(self, tmp_path):
        """force_run=True clears the markers of the steps it runs, then re-runs them."""
        orch = self._make_orchestrator(tmp_path)

        write_marker(orch.project_dir, "setup_project", "old")
        write_marker(orch.project_dir, "define_domain", "old")

        with patch.object(orch, 'define_workflow_steps') as mock_define:
            mock_define.return_value = []
            orch.run_workflow(force_run=True)

        # Nothing ran, so nothing was cleared (markers belong to steps that run).
        assert read_marker(orch.project_dir, "setup_project") is not None

        func = MagicMock()
        with patch.object(orch, 'define_workflow_steps') as mock_define:
            mock_define.return_value = [
                WorkflowStep("setup_project", "setup_project", func, lambda: True, "Setup"),
                WorkflowStep("define_domain", "define_domain", func, lambda: True, "Define"),
            ]
            orch.run_workflow(force_run=True)

        assert func.call_count == 2
        # Re-written with the current hash, not the old one.
        assert read_marker(orch.project_dir, "setup_project").config_hash != "old"

    def test_scoped_force_run_clears_only_selected_markers(self, tmp_path):
        """WORKFLOW_STEPS + force_run must not wipe the other steps' markers."""
        orch = self._make_orchestrator(tmp_path, WORKFLOW_STEPS=["define_domain"])

        write_marker(orch.project_dir, "setup_project", "keep")
        write_marker(orch.project_dir, "define_domain", "old")

        func = MagicMock()
        with patch.object(orch, 'define_workflow_steps') as mock_define:
            mock_define.return_value = [
                WorkflowStep("setup_project", "setup_project", func, lambda: True, "Setup"),
                WorkflowStep("define_domain", "define_domain", func, lambda: True, "Define"),
            ]
            orch.run_workflow(force_run=True)

        func.assert_called_once()
        assert read_marker(orch.project_dir, "setup_project").config_hash == "keep"
        assert read_marker(orch.project_dir, "define_domain").config_hash != "old"

    def test_get_workflow_status_includes_marker_fields(self, tmp_path):
        """get_workflow_status() reports marker_valid and config_stale."""
        orch = self._make_orchestrator(tmp_path)

        with patch.object(orch, 'define_workflow_steps') as mock_define:
            step = WorkflowStep(
                "setup_project", "setup_project", MagicMock(),
                lambda: True, "Setup"  # output exists
            )
            mock_define.return_value = [step]

            # No marker yet → stale
            status = orch.get_workflow_status()

        detail = status['step_details'][0]
        assert detail['marker_valid'] is False
        assert detail['config_stale'] is True
        assert detail['complete'] is False
        assert status['completed_steps'] == 0
        assert status['pending_steps'] == 1

    def test_run_individual_steps_writes_marker(self, tmp_path):
        """run_individual_steps writes markers after successful execution."""
        orch = self._make_orchestrator(tmp_path)

        with patch.object(orch, 'define_workflow_steps') as mock_define:
            step = WorkflowStep(
                "define_domain", "define_domain", MagicMock(),
                lambda: False, "Define domain"
            )
            mock_define.return_value = [step]

            orch.run_individual_steps(["define_domain"])

        marker = read_marker(orch.project_dir, "define_domain")
        assert marker is not None
        assert marker.stage == "define_domain"


class TestWorkflowSummaryHonesty:
    """The end-of-run summary must reflect the true outcome."""

    @staticmethod
    def _make_orchestrator(tmp_path, **extra_config):
        managers = {name: MagicMock() for name in
                    ('project', 'domain', 'data', 'model', 'analysis', 'optimization')}
        config = SymfluenceConfig.from_minimal(
            domain_name='summary_test',
            model='SUMMA',
            time_start='2010-01-01 00:00',
            time_end='2010-12-31 23:00',
            SYMFLUENCE_DATA_DIR=tmp_path,
            **extra_config,
        )
        return WorkflowOrchestrator(managers, config, MagicMock())

    @staticmethod
    def _logged_messages(mock_logger):
        calls = []
        for method in ('info', 'warning', 'error'):
            calls.extend(str(c.args[0]) for c in getattr(mock_logger, method).call_args_list
                         if c.args)
        return calls

    def test_success_run_logs_success_line(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        steps = [WorkflowStep("ok", "ok_cli", MagicMock(), lambda: False, "ok step")]
        with patch.object(orch, 'define_workflow_steps', return_value=steps):
            orch.run_workflow()

        messages = self._logged_messages(orch.logger)
        assert any('✓ Workflow completed successfully' in m for m in messages)
        assert not any('✗ Workflow finished with failures' in m for m in messages)

    def test_failed_run_does_not_claim_success(self, tmp_path):
        orch = self._make_orchestrator(tmp_path, STOP_ON_ERROR=False)

        def boom():
            raise ValueError('exploded')

        steps = [
            WorkflowStep("bad", "bad_cli", boom, lambda: False, "bad step"),
            WorkflowStep("ok", "ok_cli", MagicMock(), lambda: False, "ok step"),
        ]
        with patch.object(orch, 'define_workflow_steps', return_value=steps):
            orch.run_workflow()

        messages = self._logged_messages(orch.logger)
        assert not any('✓ Workflow completed successfully' in m for m in messages)
        failure_lines = [m for m in messages
                         if '✗ Workflow finished with failures' in m]
        assert failure_lines, "failed run must log the failure summary"
        assert '1 steps failed' in failure_lines[0]

    def test_failed_run_summary_logged_even_when_stopping(self, tmp_path):
        orch = self._make_orchestrator(tmp_path, STOP_ON_ERROR=True)

        def boom():
            raise ValueError('exploded')

        steps = [WorkflowStep("bad", "bad_cli", boom, lambda: False, "bad step")]
        with patch.object(orch, 'define_workflow_steps', return_value=steps):
            with pytest.raises(ValueError, match='exploded'):
                orch.run_workflow()

        messages = self._logged_messages(orch.logger)
        assert any('✗ Workflow finished with failures' in m for m in messages)

    def test_last_step_results_schema(self, tmp_path):
        orch = self._make_orchestrator(tmp_path, STOP_ON_ERROR=False)

        def boom():
            raise ValueError('exploded')

        steps = [
            WorkflowStep("ok", "ok_cli", MagicMock(), lambda: False, "ok step"),
            WorkflowStep("skipme", "skip_cli", MagicMock(), lambda: True, "skipped step"),
            WorkflowStep("bad", "bad_cli", boom, lambda: False, "bad step"),
        ]
        with patch.object(orch, 'define_workflow_steps', return_value=steps):
            orch.run_workflow()

        results = orch.last_step_results
        assert [r['status'] for r in results] == ['completed', 'skipped', 'failed']
        for entry in results:
            assert set(entry) >= {'name', 'cli_name', 'description',
                                  'status', 'duration_s'}
            assert isinstance(entry['duration_s'], float)
        assert results[2]['error'] == 'exploded'
        # NOTE: 'skipme'/'bad' aren't hash-tracked stages, so no markers written

    def test_run_individual_steps_populates_last_step_results(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        steps = [WorkflowStep("one", "one_cli", MagicMock(), lambda: True, "one")]
        with patch.object(orch, 'define_workflow_steps', return_value=steps):
            orch.run_individual_steps(["one_cli"])

        assert len(orch.last_step_results) == 1
        entry = orch.last_step_results[0]
        assert entry['cli_name'] == 'one_cli'
        assert entry['status'] == 'completed'
        assert entry['duration_s'] >= 0.0


class TestWorkflowStepsScoping:
    """Tests for config-level WORKFLOW_STEPS run scoping."""

    @staticmethod
    def _make_orchestrator(tmp_path, **extra_config):
        managers = {name: MagicMock() for name in
                    ('project', 'domain', 'data', 'model', 'analysis', 'optimization')}
        config = SymfluenceConfig.from_minimal(
            domain_name='scoping_test',
            model='SUMMA',
            time_start='2010-01-01 00:00',
            time_end='2010-12-31 23:00',
            SYMFLUENCE_DATA_DIR=tmp_path,
            **extra_config,
        )
        return WorkflowOrchestrator(managers, config, MagicMock())

    @staticmethod
    def _three_steps():
        return [
            WorkflowStep("setup_project", "setup_project", MagicMock(), lambda: False, "Setup"),
            WorkflowStep("define_domain", "define_domain", MagicMock(), lambda: False, "Define"),
            WorkflowStep("run_models", "run_model", MagicMock(), lambda: False, "Run"),
        ]

    def test_workflow_steps_scopes_run(self, tmp_path):
        """Only steps listed in WORKFLOW_STEPS execute; the rest are never invoked."""
        orch = self._make_orchestrator(
            tmp_path, WORKFLOW_STEPS=['setup_project', 'define_domain']
        )
        steps = self._three_steps()
        with patch.object(orch, 'define_workflow_steps', return_value=steps):
            orch.run_workflow()

        steps[0].func.assert_called_once()
        steps[1].func.assert_called_once()
        steps[2].func.assert_not_called()

    def test_workflow_steps_aliases_resolve(self, tmp_path):
        """Aliases in WORKFLOW_STEPS resolve to canonical CLI names."""
        orch = self._make_orchestrator(tmp_path, WORKFLOW_STEPS=['setup', 'domain'])
        steps = self._three_steps()
        with patch.object(orch, 'define_workflow_steps', return_value=steps):
            orch.run_workflow()

        steps[0].func.assert_called_once()
        steps[1].func.assert_called_once()
        steps[2].func.assert_not_called()

    def test_workflow_steps_absent_runs_all(self, tmp_path):
        """Without WORKFLOW_STEPS every step executes (existing behavior)."""
        orch = self._make_orchestrator(tmp_path)
        steps = self._three_steps()
        with patch.object(orch, 'define_workflow_steps', return_value=steps):
            orch.run_workflow()

        for step in steps:
            step.func.assert_called_once()

    def test_workflow_steps_unknown_name_rejected(self, tmp_path):
        """Unknown step names fail at config parse time with a helpful error."""
        with pytest.raises(Exception, match="unknown step"):
            self._make_orchestrator(tmp_path, WORKFLOW_STEPS=['not_a_step'])


class TestContinueOnErrorFailures:
    """STOP_ON_ERROR=False: failures are tracked and no marker is written."""

    def test_failed_step_tracked_without_marker(self, tmp_path):
        managers = {name: MagicMock() for name in
                    ('project', 'domain', 'data', 'model', 'analysis', 'optimization')}
        config = SymfluenceConfig.from_minimal(
            domain_name='fail_test', model='SUMMA',
            time_start='2010-01-01 00:00', time_end='2010-12-31 23:00',
            SYMFLUENCE_DATA_DIR=tmp_path, STOP_ON_ERROR=False,
        )
        orch = WorkflowOrchestrator(managers, config, MagicMock())
        boom = MagicMock(side_effect=RuntimeError("boom"))
        ok = MagicMock()
        with patch.object(orch, 'define_workflow_steps') as mock_define:
            mock_define.return_value = [
                WorkflowStep("define_domain", "define_domain", boom, lambda: False, "Define"),
                WorkflowStep("setup_project", "setup_project", ok, lambda: False, "Setup"),
            ]
            orch.run_workflow()  # does not raise

        ok.assert_called_once()
        assert orch.failed_step_names() == ["define_domain"]
        assert read_marker(orch.project_dir, "define_domain") is None
        assert read_marker(orch.project_dir, "setup_project") is not None
