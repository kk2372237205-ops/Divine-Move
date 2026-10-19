"""Best-effort local speech for visible match results."""

from __future__ import annotations

import os
import subprocess
import sys


def announce(text: str) -> bool:
    """Speak a short local status on Windows without blocking the UI.

    The app only passes fixed game-result strings here.  If Windows speech is
    unavailable, the visual result banner remains the source of truth.
    """
    if sys.platform != "win32" or not text.strip():
        return False
    safe_text = text.replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$speaker.Speak('{safe_text}')"
    )
    try:
        subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except OSError:
        return False
    return True
