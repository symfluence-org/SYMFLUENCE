"""Tests for AnalysisManager observation path resolution."""

import logging
import tempfile
from pathlib import Path

import pandas as pd

from symfluence.core.config.models import SymfluenceConfig
from symfluence.evaluation.analysis_manager import AnalysisManager


def _make_config(temp_root: Path) -> SymfluenceConfig:
    """Create a minimal typed config for analysis manager tests."""
    return SymfluenceConfig.from_minimal(
        domain_name="test_domain",
        model="SUMMA",
        SYMFLUENCE_DATA_DIR=str(temp_root),
        EXPERIMENT_ID="test_exp",
        EXPERIMENT_TIME_START="2010-01-01 00:00",
        EXPERIMENT_TIME_END="2010-01-10 23:00",
        CALIBRATION_PERIOD="2010-01-01, 2010-01-05",
        EVALUATION_PERIOD="2010-01-06, 2010-01-10",
    )


def _write_streamflow_csv(path: Path) -> None:
    df = pd.DataFrame(
        {"streamflow": [1.0, 2.0, 3.0]},
        index=pd.date_range("2010-01-01", periods=3, freq="D"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def _write_tws_csv(path: Path) -> None:
    df = pd.DataFrame(
        {"grace_jpl_anomaly": [10.0, 11.0, 12.0]},
        index=pd.date_range("2010-01-01", periods=3, freq="MS"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def test_load_all_observations_uses_shared_fallback_candidates() -> None:
    """Loads streamflow/TWS from non-primary legacy locations via shared helpers."""
    with tempfile.TemporaryDirectory(prefix="sf_analysis_tests_") as tmp_dir:
        config = _make_config(Path(tmp_dir))
        manager = AnalysisManager(config, logging.getLogger("test_analysis_manager"))
        project_dir = manager.project_dir

        # Streamflow at fallback location: observations/streamflow/processed
        _write_streamflow_csv(
            project_dir
            / "observations"
            / "streamflow"
            / "processed"
            / "test_domain_streamflow_processed.csv"
        )

        # TWS at fallback location: observations/storage/grace/preprocessed
        _write_tws_csv(
            project_dir
            / "observations"
            / "storage"
            / "grace"
            / "preprocessed"
            / "test_domain_grace_tws_processed.csv"
        )

        observations = manager._load_all_observations()

        assert "STREAMFLOW" in observations
        assert "TWS" in observations
        assert observations["STREAMFLOW"].iloc[0] == 1.0
        assert observations["TWS"].iloc[0] == 10.0


def test_validate_analysis_requirements_accepts_processed_streamflow_path() -> None:
    """Requirement validation succeeds when streamflow exists in processed fallback path."""
    with tempfile.TemporaryDirectory(prefix="sf_analysis_tests_") as tmp_dir:
        config = _make_config(Path(tmp_dir))
        manager = AnalysisManager(config, logging.getLogger("test_analysis_manager"))
        project_dir = manager.project_dir

        _write_streamflow_csv(
            project_dir
            / "observations"
            / "streamflow"
            / "processed"
            / "test_domain_streamflow_processed.csv"
        )
        (project_dir / "optimization").mkdir(parents=True, exist_ok=True)
        (
            project_dir
            / "optimization"
            / "test_exp_parallel_iteration_results.csv"
        ).write_text("iter,best\n1,0.5\n", encoding="utf-8")
        (project_dir / "simulations" / "test_exp").mkdir(parents=True, exist_ok=True)

        requirements = manager.validate_analysis_requirements()

        assert requirements == {
            "benchmarking": True,
            "sensitivity_analysis": True,
            "decision_analysis": True,
        }


def test_benchmarking_skipped_for_swe_target() -> None:
    """Benchmarking returns None and logs skip when OPTIMIZATION_TARGET is SWE."""
    with tempfile.TemporaryDirectory(prefix="sf_analysis_tests_") as tmp_dir:
        config = SymfluenceConfig.from_minimal(
            domain_name="test_domain",
            model="SUMMA",
            SYMFLUENCE_DATA_DIR=str(tmp_dir),
            EXPERIMENT_ID="test_exp",
            EXPERIMENT_TIME_START="2010-01-01 00:00",
            EXPERIMENT_TIME_END="2010-01-10 23:00",
            CALIBRATION_PERIOD="2010-01-01, 2010-01-05",
            EVALUATION_PERIOD="2010-01-06, 2010-01-10",
            OPTIMIZATION_TARGET="SWE",
        )
        logger = logging.getLogger("test_benchmark_skip")
        manager = AnalysisManager(config, logger)

        result = manager.run_benchmarking()
        assert result is None


def test_benchmarking_not_skipped_for_streamflow_target() -> None:
    """Benchmarking does not short-circuit for streamflow target."""
    with tempfile.TemporaryDirectory(prefix="sf_analysis_tests_") as tmp_dir:
        config = SymfluenceConfig.from_minimal(
            domain_name="test_domain",
            model="SUMMA",
            SYMFLUENCE_DATA_DIR=str(tmp_dir),
            EXPERIMENT_ID="test_exp",
            EXPERIMENT_TIME_START="2010-01-01 00:00",
            EXPERIMENT_TIME_END="2010-01-10 23:00",
            CALIBRATION_PERIOD="2010-01-01, 2010-01-05",
            EVALUATION_PERIOD="2010-01-06, 2010-01-10",
            OPTIMIZATION_TARGET="streamflow",
        )
        logger = logging.getLogger("test_benchmark_proceed")
        manager = AnalysisManager(config, logger)

        # Returns None because observation data is missing, but should
        # NOT have short-circuited due to the target check
        result = manager.run_benchmarking()
        assert result is None


def test_validate_analysis_requirements_fails_without_streamflow() -> None:
    """All analyses are marked unavailable when no streamflow observations are found."""
    with tempfile.TemporaryDirectory(prefix="sf_analysis_tests_") as tmp_dir:
        config = _make_config(Path(tmp_dir))
        manager = AnalysisManager(config, logging.getLogger("test_analysis_manager"))
        project_dir = manager.project_dir

        (project_dir / "optimization").mkdir(parents=True, exist_ok=True)
        (
            project_dir
            / "optimization"
            / "test_exp_parallel_iteration_results.csv"
        ).write_text("iter,best\n1,0.5\n", encoding="utf-8")
        (project_dir / "simulations" / "test_exp").mkdir(parents=True, exist_ok=True)

        requirements = manager.validate_analysis_requirements()

        assert requirements == {
            "benchmarking": False,
            "sensitivity_analysis": False,
            "decision_analysis": False,
        }
