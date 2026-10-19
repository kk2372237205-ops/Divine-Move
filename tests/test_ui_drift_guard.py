from types import SimpleNamespace

import numpy as np

import pc_app.ui as ui_module
from pc_app.calibration import BoardCalibration, CalibrationAlignment
from pc_app.ui import CALIBRATION_DRIFT_FAILURE_LIMIT, DiagnosticApp


class _Status:
    def __init__(self) -> None:
        self.value = ""

    def set(self, value: str) -> None:
        self.value = value


def test_drift_guard_pauses_for_transient_occlusion_before_invalidating(monkeypatch) -> None:
    """One or two bad corner frames (such as a hand) must not discard calibration."""
    calibration = BoardCalibration(((100, 80), (500, 90), (510, 480), (90, 470)))
    invalidations: list[str] = []
    app = SimpleNamespace(
        calibration=calibration,
        calibration_invalidated=False,
        last_frame=np.zeros((500, 700, 3), dtype=np.uint8),
        calibration_last_checked=0.0,
        calibration_check_interval=1.0,
        calibration_drift_failures=0,
        status_var=_Status(),
        _invalidate_calibration=invalidations.append,
    )
    bad_alignment = CalibrationAlignment(False, "棋盘角点被短暂遮挡")
    monkeypatch.setattr(ui_module, "check_calibration_alignment", lambda *_args: bad_alignment)

    for failure_count in range(1, CALIBRATION_DRIFT_FAILURE_LIMIT):
        assert not DiagnosticApp._monitor_calibration_drift(app, force=True)
        assert app.calibration_drift_failures == failure_count
        assert not invalidations
        assert "请将手移出棋盘" in app.status_var.value

    assert not DiagnosticApp._monitor_calibration_drift(app, force=True)
    assert len(invalidations) == 1


def test_dry_run_disables_empty_board_runtime_guards() -> None:
    """Hands and normal stones are expected during a no-hardware dry run."""
    assert not DiagnosticApp._runtime_board_guards_enabled(SimpleNamespace(dry_run_enabled=True))
    assert DiagnosticApp._runtime_board_guards_enabled(SimpleNamespace(dry_run_enabled=False))
    assert not DiagnosticApp._runtime_board_guards_enabled(
        SimpleNamespace(dry_run_enabled=False, board_clearance_in_progress=True)
    )
