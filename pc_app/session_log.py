"""Append-only, session-scoped evidence logs for advanced-match verification."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import threading
from typing import Any


class SessionEventLog:
    """Write one JSON object per event without rewriting prior session evidence."""

    def __init__(self, logs_root: Path) -> None:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        self.path = logs_root / "sessions" / f"advanced_match_{stamp}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, event: str, **details: Any) -> None:
        payload = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "event": event,
            **details,
        }
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line + "\n")
            stream.flush()
