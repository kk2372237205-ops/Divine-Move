"""Safe, computer-side diagnostic entry point."""

from .ui import DiagnosticApp


def main() -> None:
    app = DiagnosticApp()
    app.run()
