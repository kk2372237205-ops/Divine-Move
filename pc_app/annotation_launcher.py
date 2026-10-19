"""Standalone offline launcher for the annotation workbench."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk

from .annotation_workbench import AnnotationWorkbench
from .dataset import DatasetManager


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    root = tk.Tk()
    root.withdraw()
    workbench = AnnotationWorkbench(
        root,
        project_root,
        DatasetManager(project_root / "dataset"),
    )

    def close_all() -> None:
        workbench.close()
        if root.winfo_exists():
            root.destroy()

    workbench.window.protocol("WM_DELETE_WINDOW", close_all)
    root.mainloop()


if __name__ == "__main__":
    main()
