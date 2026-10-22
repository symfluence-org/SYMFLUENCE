import pytest
from symfluence.core.exceptions import ModelExecutionError
from symfluence.models.watflood.runner import WATFLOODRunner

@pytest.mark.parametrize('message', ['PAUSE Hit enter to use the event file values', 'STOP program aborted in read_par_parser @ 1195', 'Fortran runtime error: End of file', 'Error: Program aborted in sub @ 1455'])
def test_zero_exit_native_abort_is_rejected(tmp_path, message):
    log=tmp_path/'stderr.log';log.write_text(message)
    with pytest.raises(ModelExecutionError):
        WATFLOODRunner._check_native_logs(log)

def test_normal_log_is_accepted(tmp_path):
    log=tmp_path/'stdout.log';log.write_text('Finished simulation successfully\n')
    WATFLOODRunner._check_native_logs(log)

def test_documented_normal_fortran_stop_is_accepted(tmp_path):
    log=tmp_path/'stderr.log';log.write_text('STOP CHARM: normal ending\n')
    WATFLOODRunner._check_native_logs(log)

def test_normal_stop_does_not_hide_prior_abort(tmp_path):
    log=tmp_path/'stderr.log';log.write_text('Fortran runtime error: invalid index\nSTOP CHARM: normal ending\n')
    with pytest.raises(ModelExecutionError):
        WATFLOODRunner._check_native_logs(log)
