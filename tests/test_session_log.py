import json

from pc_app.session_log import SessionEventLog


def test_session_event_log_appends_structured_events(tmp_path) -> None:
    logger = SessionEventLog(tmp_path)

    logger.record("robot_move_started", move=(7, 7), automatic=True)
    logger.record("advanced_match_safety_stop", reason="视觉确认超时")

    lines = logger.path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert logger.path.parent == tmp_path / "sessions"
    assert [record["event"] for record in records] == ["robot_move_started", "advanced_match_safety_stop"]
    assert records[0]["move"] == [7, 7]
    assert records[1]["reason"] == "视觉确认超时"
