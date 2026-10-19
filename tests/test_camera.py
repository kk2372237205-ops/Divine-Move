import numpy as np

from pc_app.services import camera as camera_module
from pc_app.services.camera import CameraService


class FakeCapture:
    def __init__(self, frames: list[object | None], opened: bool = True) -> None:
        self.frames = frames
        self.opened = opened
        self.released = False
        self.settings: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, object | None]:
        frame = self.frames.pop(0) if self.frames else None
        return frame is not None, frame

    def set(self, prop: int, value: float) -> bool:
        self.settings.append((prop, value))
        return True

    def release(self) -> None:
        self.released = True


def test_start_uses_a_backend_that_returns_a_real_frame(monkeypatch) -> None:
    unavailable = FakeCapture([], opened=False)
    frame = np.ones((4, 5, 3), dtype=np.uint8)
    working = FakeCapture([frame] * (camera_module.CAMERA_WARMUP_FRAMES + 1))
    captures = [unavailable, working]

    def fake_video_capture(index: int, backend: int) -> FakeCapture:
        return captures.pop(0)

    monkeypatch.setattr(camera_module, "CAMERA_BACKENDS", (("first", 1), ("second", 2)))
    monkeypatch.setattr(camera_module.cv2, "VideoCapture", fake_video_capture)

    service = CameraService()
    ok, message = service.start(1)

    assert ok
    assert "second" in message
    assert unavailable.released
    assert service.read() is frame
    assert service.read() is frame
    assert service.frame_width == 5
    assert service.frame_height == 4
    assert "5x4" in message
    assert len(working.settings) == 5


def test_start_rejects_a_camera_that_opens_but_never_returns_a_frame(monkeypatch) -> None:
    captures = [FakeCapture([]), FakeCapture([])]

    def fake_video_capture(index: int, backend: int) -> FakeCapture:
        return captures.pop(0)

    monkeypatch.setattr(camera_module, "CAMERA_BACKENDS", (("first", 1), ("second", 2)))
    monkeypatch.setattr(camera_module.cv2, "VideoCapture", fake_video_capture)

    service = CameraService()
    ok, message = service.start(0)

    assert not ok
    assert "未返回画面" in message


def test_correct_color_balances_channels_without_mutating_input() -> None:
    frame = np.array([[[100, 100, 100], [255, 200, 0]]], dtype=np.uint8)

    corrected = CameraService.correct_color(frame)

    assert corrected is not frame
    assert frame.tolist() == [[[100, 100, 100], [255, 200, 0]]]
    assert corrected[0, 0].tolist() == [80, 100, 115]
    assert corrected[0, 1].tolist() == [204, 200, 0]
