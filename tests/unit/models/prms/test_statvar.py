"""Shared PRMS / GSFLOW statvar reader: one header line per element entry."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from symfluence.core.exceptions import ModelExecutionError
from symfluence.models.prms.statvar import (
    CFS_TO_CMS,
    basin_variable,
    parse_statvar,
    read_statvar_streamflow,
    variable_frame,
)


def _write(path, entries, rows, fmt="{:.6E}"):
    """Write a statvar the way PRMS does: N, N '<name> <element>' lines, rows."""
    lines = [str(len(entries))] + [f"{n} {e}" for n, e in entries]
    for step, (date, values) in enumerate(rows, 1):
        y, m, d = date
        lines.append(f"{step:7d} {y:4d} {m:2d} {d:2d}  0  0  0 " + " ".join(fmt.format(v) for v in values))
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def prms_statvar(tmp_path):
    """PRMS glacier ordering: basin scalars, then 3-HRU arrays, then more scalars."""
    entries = [("basin_cfs", 1), ("basin_actet", 1)]
    entries += [("hru_actet", i) for i in (1, 2, 3)]
    entries += [("basin_storage", 1)]
    entries += [("pkwater_equiv", i) for i in (1, 2, 3)]
    rows = [((2000, 1, 1), [100., .1, 1., 2., 3., 50., 10., 20., 30.]),
            ((2000, 1, 2), [200., .2, 4., 5., 6., 60., 11., 21., 31.])]
    return _write(tmp_path / "statvar.dat", entries, rows)


def test_one_column_per_element_entry(prms_statvar):
    header, df = parse_statvar(prms_statvar)
    assert len(header) == 9
    assert list(df.columns) == [
        "basin_cfs_1", "basin_actet_1", "hru_actet_1", "hru_actet_2", "hru_actet_3",
        "basin_storage_1", "pkwater_equiv_1", "pkwater_equiv_2", "pkwater_equiv_3"]
    assert df.index[1].day == 2
    # Columns after a multi-HRU variable stay aligned
    assert df["basin_storage_1"].tolist() == [50., 60.]
    assert df["pkwater_equiv_3"].tolist() == [30., 31.]


def test_variable_helpers(prms_statvar):
    header, df = parse_statvar(prms_statvar)
    assert basin_variable(header, df, "basin_actet").tolist() == [.1, .2]
    assert basin_variable(header, df, "basin_actet").name == "basin_actet"
    assert variable_frame(header, df, "hru_actet").to_numpy().tolist() == [[1., 2., 3.], [4., 5., 6.]]
    with pytest.raises(ModelExecutionError, match="not a basin scalar"):
        basin_variable(header, df, "hru_actet")
    with pytest.raises(ModelExecutionError, match="no variable"):
        basin_variable(header, df, "basin_cms")


def test_streamflow_after_hru_arrays(tmp_path):
    """basin_cfs listed after per-HRU entries is still read from its own column."""
    entries = [("hru_actet", i) for i in (1, 2, 3, 4)] + [("basin_cfs", 1)]
    sv = _write(tmp_path / "statvar.dat", entries, [((2001, 6, 1), [1., 2., 3., 4., 350.])])
    assert read_statvar_streamflow(sv).iloc[0] == pytest.approx(350 * CFS_TO_CMS)


def test_streamflow_segment_element(tmp_path):
    entries = [("seg_outflow", 2), ("seg_outflow", 5)]
    sv = _write(tmp_path / "statvar.dat", entries, [((2001, 6, 1), [10., 50.])])
    assert read_statvar_streamflow(sv).iloc[0] == pytest.approx(50 * CFS_TO_CMS)
    assert read_statvar_streamflow(sv, element=2).iloc[0] == pytest.approx(10 * CFS_TO_CMS)
    with pytest.raises(ModelExecutionError, match="no element 3"):
        read_statvar_streamflow(sv, element=3)


def test_gsflow_statvars_ordering(tmp_path):
    """The statvar list the GSFLOW preprocessor writes parses column by column."""
    from symfluence.models.gsflow.basin_inputs import statvars

    entries = statvars(SimpleNamespace(nhru=4), glacier=True, cryosphere=True)
    rng = np.random.default_rng(0)
    values = rng.random((3, len(entries)))
    rows = [((2010, 10, d), list(values[d - 1])) for d in (1, 2, 3)]
    sv = _write(tmp_path / "statvar.dat", entries, rows, fmt="{:.15E}")
    header, df = parse_statvar(sv)
    assert header == list(entries)
    np.testing.assert_allclose(df.to_numpy(), values)
    np.testing.assert_allclose(variable_frame(header, df, "snowcov_area").to_numpy(),
                               values[:, [i for i, (n, _) in enumerate(entries) if n == "snowcov_area"]])
    np.testing.assert_allclose(read_statvar_streamflow(sv).to_numpy(), values[:, 0] * CFS_TO_CMS)


def test_gsflow_and_prms_extractors_mean_over_hrus(prms_statvar):
    from symfluence.models.gsflow.extractor import GSFLOWResultExtractor
    from symfluence.models.prms.extractor import PRMSResultExtractor

    et = GSFLOWResultExtractor("GSFLOW").extract_variable(prms_statvar, "et")
    assert et.tolist() == pytest.approx([2., 5.])
    prms = PRMSResultExtractor("PRMS")
    assert prms._extract_from_statvar(prms_statvar, ["hru_actet"]).tolist() == pytest.approx([2., 5.])


def test_malformed_files_raise(tmp_path):
    sv = tmp_path / "statvar.dat"
    sv.write_text("2\nbasin_cfs 1\nbasin_cfs 1\n1 2000 1 1 0 0 0 1 2\n")
    with pytest.raises(ModelExecutionError, match="more than once"):
        parse_statvar(sv)
    sv.write_text("2\nbasin_cfs 1\nbasin_actet 1\n1 2000 1 1 0 0 0 1\n")
    with pytest.raises(ModelExecutionError, match="no data rows"):
        parse_statvar(sv)
    sv.write_text("2\nbasin_cfs 1\n")
    with pytest.raises(ModelExecutionError, match="expected 2 header entries"):
        parse_statvar(sv)


def test_incomplete_trailing_row_skipped(tmp_path):
    sv = tmp_path / "statvar.dat"
    sv.write_text("2\nbasin_cfs 1\nbasin_actet 1\n1 2000 1 1 0 0 0 1 2\n2 2000 1 2 0 0 0 3\n")
    _, df = parse_statvar(sv)
    assert len(df) == 1
