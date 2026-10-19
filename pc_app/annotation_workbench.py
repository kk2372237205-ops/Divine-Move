"""Human-in-the-loop annotation window for the fixed Gomoku board.

The workbench keeps image proposals separate from accepted truth.  A proposal
is only copied into the editable grid after the operator presses the explicit
accept button, and a reviewed label is written only after the operator checks
the confirmation box.  This keeps the fast path convenient without silently
turning an image heuristic into ground truth.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .annotation import (
    GridProposal,
    archive_annotation_family,
    build_truth,
    export_truth_family_labels,
    propose_grid_stones,
    write_truth_record,
)
from .calibration import BoardCalibration, detect_board_corners
from .dataset import DatasetManager
from .image_files import read_color_image, write_image
from .labels import (
    BoxOverride,
    box_overrides_from_json,
    draw_grid_annotations,
    fit_grid_box_overrides,
    project_grid_annotations,
)


EMPTY = 0
BLACK = 1
WHITE = 2
BOARD_DISPLAY_SIZE = 500.0
BOARD_DISPLAY_MARGIN = 22.0


def cycle_state(state: int) -> int:
    """Cycle a grid cell through empty, black, and white."""

    if state not in (EMPTY, BLACK, WHITE):
        raise ValueError(f"非法棋子状态：{state}")
    return (state + 1) % 3


def states_to_moves(states: dict[tuple[int, int], int]) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Return sorted black and white moves from editable grid state."""

    black = tuple(sorted((move for move, state in states.items() if state == BLACK), key=lambda move: (move[1], move[0])))
    white = tuple(sorted((move for move, state in states.items() if state == WHITE), key=lambda move: (move[1], move[0])))
    return black, white


def display_grid_geometry(board_size: int) -> tuple[float, float]:
    """Return the display margin and grid spacing for the review canvas."""

    if board_size < 2:
        raise ValueError("棋盘至少需要两条网格线")
    spacing = (BOARD_DISPLAY_SIZE - 2 * BOARD_DISPLAY_MARGIN) / (board_size - 1)
    return BOARD_DISPLAY_MARGIN, spacing


def display_point_to_move(x: float, y: float, board_size: int) -> tuple[int, int] | None:
    """Map a click on the padded review canvas to the nearest board point."""

    margin, spacing = display_grid_geometry(board_size)
    column = round((float(x) - margin) / spacing)
    row = round((float(y) - margin) / spacing)
    if not (0 <= column < board_size and 0 <= row < board_size):
        return None
    if abs(float(x) - (margin + column * spacing)) > spacing * 0.45:
        return None
    if abs(float(y) - (margin + row * spacing)) > spacing * 0.45:
        return None
    return column, row


def stabilize_board_corners(
    corners: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Repair an implausible lower-left grid corner caused by hardware occlusion.

    The picker mechanism can cover the lower-left grid lines.  In that case
    the line detector often returns an inner grid intersection.  The other
    three corners and the visible left-edge slope give a stable estimate for
    the missing lower-left outer intersection.
    """
    top_left, top_right, bottom_right, bottom_left = corners
    right_drop = bottom_right[1] - top_right[1]
    left_drop = bottom_left[1] - top_left[1]
    if right_drop <= 0 or left_drop >= right_drop * 0.92:
        return corners

    repaired_y = bottom_right[1] - (top_right[1] - top_left[1])
    slope_x = (bottom_left[0] - top_left[0]) / max(left_drop, 1.0)
    repaired_x = top_left[0] + slope_x * (repaired_y - top_left[1])
    return top_left, top_right, bottom_right, (repaired_x, repaired_y)


def calibration_for_image(
    image,
    fallback: BoardCalibration,
) -> tuple[BoardCalibration, bool]:
    """Use this image's board geometry when it can be detected reliably.

    Dataset images may have been captured before the camera or board was moved.
    Their label boxes must be projected with their own four corners, rather
    than a newer global calibration file.
    """
    corners = detect_board_corners(image)
    if corners is None:
        return fallback, False
    return BoardCalibration(
        stabilize_board_corners(corners),
        output_size=fallback.output_size,
        board_size=fallback.board_size,
    ), True


def exclude_image_family(dataset_root: Path, image_path: Path, *, reason: str = "") -> Path:
    """Remove one source image family from the active dataset with an archive.

    The raw image, its four generated variants, active YOLO labels, and review
    previews leave the active dataset together.  They are moved to ``excluded``
    instead of being destroyed so an accidental deletion remains recoverable.
    """
    dataset = DatasetManager(Path(dataset_root))
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"原图不存在：{image_path}")
    try:
        image_path.relative_to(dataset.raw_dir)
    except ValueError as error:
        raise ValueError("只能删除 dataset/raw 中的原图") from error

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive_root = dataset.root / "excluded" / f"{stamp}_{image_path.stem}"
    archive_root.mkdir(parents=True, exist_ok=False)
    stem = image_path.stem
    variants = ("brightness", "color", "flip", "blur")
    candidates = [
        image_path,
        *(dataset.augmented_dir / f"{stem}_{suffix}.jpg" for suffix in variants),
        dataset.labels_dir / f"{stem}.txt",
        *(dataset.labels_dir / f"{stem}_{suffix}.txt" for suffix in variants),
        *sorted((dataset.root / "annotation_previews").glob(f"{stem}_reviewed*.jpg")),
    ]
    moved: list[str] = []
    for source in candidates:
        if not source.exists():
            continue
        relative = source.relative_to(dataset.root)
        destination = archive_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        moved.append(relative.as_posix())

    exclusions_path = dataset.root / "annotations" / "exclusions.jsonl"
    exclusions_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "excluded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": image_path.name,
        "reason": reason or "用户在标注工作台删除本图",
        "archive": archive_root.relative_to(dataset.root).as_posix(),
        "files": moved,
    }
    with exclusions_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return archive_root


def load_reviewed_truths(path: Path) -> dict[str, dict[str, object]]:
    """Load the latest reviewed truth for each source image."""

    if not path.exists():
        return {}
    truths: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            image_name = str(payload["image"])
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"真值记录第 {line_number} 行无效：{error}") from error
        if payload.get("status") == "reviewed":
            truths[image_name] = payload
    return truths


class AnnotationWorkbench:
    """Tk window used to review source images and export traceable labels."""

    def __init__(self, parent: tk.Misc, project_root: Path, dataset: DatasetManager) -> None:
        self.parent = parent
        self.project_root = project_root
        self.dataset = dataset
        self.calibration_path = project_root / "calibration" / "board.json"
        self.default_calibration: BoardCalibration | None = None
        self.calibration: BoardCalibration | None = None
        self.image_calibration_detected = False
        self.window = tk.Toplevel(parent)
        self.window.title("标注工作台 - 逐图核验并导出 YOLO")
        self.window.geometry("1400x820")
        self.window.minsize(1240, 700)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.after_idle(self._maximize_window)

        self.raw_paths = sorted(dataset.raw_dir.glob("*.jpg"))
        self.current_index = 0
        self.current_image = None
        self.current_path: Path | None = None
        self.current_proposals: tuple[GridProposal, ...] = ()
        self.states: dict[tuple[int, int], int] = {}
        self.box_overrides: tuple[BoxOverride, ...] = ()
        self.reviewed_truths: dict[str, dict[str, object]] = {}
        self.revision_mode = False
        self.original_photo: ImageTk.PhotoImage | None = None
        self.board_photo: ImageTk.PhotoImage | None = None

        self.status_var = tk.StringVar(value="正在准备图片列表……")
        self.detail_var = tk.StringVar(value="")
        self.annotator_var = tk.StringVar(value="operator")
        self.radius_var = tk.StringVar(value="18.0")
        self.quality_var = tk.StringVar(value="accepted")
        self.notes_var = tk.StringVar(value="")
        self.reviewed_var = tk.BooleanVar(value=False)

        self._build_layout()
        self._load_calibration()
        self._refresh_image_list()
        if not self.raw_paths:
            self.status_var.set("dataset/raw 中暂无 JPG 原图。")

    def _build_layout(self) -> None:
        self.window.columnconfigure(0, weight=0)
        self.window.columnconfigure(1, weight=1)
        self.window.rowconfigure(1, weight=1)

        header = ttk.Frame(self.window, padding=(12, 10, 12, 6))
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="标注工作台", font=("Microsoft YaHei UI", 15, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="操作顺序：采用当前候选 → 对照左图逐点修正 → 勾选确认 → 保存真值并导出标签",
            foreground="#176a98",
        ).grid(row=0, column=1, padx=(18, 0), sticky="w")

        sidebar = ttk.LabelFrame(self.window, text="原图列表", padding=8)
        sidebar.grid(row=1, column=0, sticky="ns", padx=(12, 8), pady=(0, 8))
        sidebar.rowconfigure(0, weight=1)
        self.image_list = tk.Listbox(sidebar, width=38, height=30, activestyle="none", exportselection=False)
        self.image_list.grid(row=0, column=0, sticky="ns")
        self.image_list.bind("<<ListboxSelect>>", self._on_image_selected)
        scroll = ttk.Scrollbar(sidebar, orient="vertical", command=self.image_list.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        horizontal_scroll = ttk.Scrollbar(sidebar, orient="horizontal", command=self.image_list.xview)
        horizontal_scroll.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.image_list.configure(yscrollcommand=scroll.set, xscrollcommand=horizontal_scroll.set)
        ttk.Button(sidebar, text="刷新图片列表", command=self._refresh_image_list).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        content = ttk.Frame(self.window, padding=(0, 0, 12, 8))
        content.grid(row=1, column=1, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(content)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(toolbar, text="上一张", command=self._previous_image).pack(side="left")
        ttk.Button(toolbar, text="下一张", command=self._next_image).pack(side="left", padx=(6, 0))
        ttk.Button(toolbar, text="下一张未完成", command=self._next_unreviewed).pack(side="left", padx=(6, 0))
        ttk.Button(toolbar, text="解锁并修改已完成标注", command=self._edit_reviewed).pack(side="left", padx=(6, 0))
        ttk.Button(toolbar, text="采用当前候选", command=self._accept_proposals).pack(side="left", padx=(16, 0))
        ttk.Button(toolbar, text="贴合棋子框", command=self._fit_box_overrides).pack(side="left", padx=(6, 0))
        ttk.Button(toolbar, text="清空本图编辑", command=self._clear_states).pack(side="left", padx=(6, 0))
        ttk.Button(toolbar, text="删除本图", command=self._delete_current_image).pack(side="left", padx=(6, 0))
        ttk.Label(toolbar, textvariable=self.detail_var, foreground="#52616b").pack(side="left", padx=(16, 0), fill="x", expand=True)

        panes = ttk.Frame(content)
        panes.grid(row=1, column=0, sticky="nsew")
        # Give the editable corrected board more room than the source preview.
        # The source remains large enough to check labels, while every edge
        # intersection on the right stays visible and clickable.
        panes.columnconfigure(0, weight=1, minsize=520)
        panes.columnconfigure(1, weight=0, minsize=int(BOARD_DISPLAY_SIZE) + 16)
        panes.rowconfigure(0, weight=1)

        original_frame = ttk.LabelFrame(panes, text="原图框预览（标签坐标在原始 1920x1080 图像上）", padding=6)
        original_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        original_frame.columnconfigure(0, weight=1)
        original_frame.rowconfigure(0, weight=1)
        self.original_canvas = tk.Canvas(original_frame, width=520, height=390, background="#eef2f1", highlightthickness=0)
        self.original_canvas.grid(row=0, column=0, sticky="nsew")

        board_frame = ttk.LabelFrame(panes, text="右图：同一棋盘的校正视图（点击交点：空→黑→白）", padding=6)
        board_frame.grid(row=0, column=1, sticky="nsew")
        board_frame.columnconfigure(0, weight=1)
        self.board_canvas = tk.Canvas(board_frame, width=int(BOARD_DISPLAY_SIZE), height=int(BOARD_DISPLAY_SIZE), background="#e2b66d", highlightthickness=0)
        self.board_canvas.grid(row=0, column=0, sticky="nsew")
        self.board_canvas.bind("<Button-1>", self._on_board_click)
        legend = ttk.Frame(board_frame)
        legend.grid(row=1, column=0, sticky="w", pady=(5, 0))
        ttk.Label(legend, text="红圈=程序猜测黑棋", foreground="#b52f25").pack(side="left", padx=(2, 10))
        ttk.Label(legend, text="蓝圈=程序猜测白棋", foreground="#176a98").pack(side="left", padx=(0, 10))
        ttk.Label(legend, text="灰色虚线=低置信误报可能性高，不代表有棋子", foreground="#6b6b6b").pack(side="left")

        footer = ttk.LabelFrame(content, text="确认与导出", padding=8)
        footer.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        footer.columnconfigure(7, weight=1)
        ttk.Label(footer, text="审核人").grid(row=0, column=0, sticky="w")
        ttk.Entry(footer, textvariable=self.annotator_var, width=14).grid(row=0, column=1, padx=(5, 14), sticky="w")
        ttk.Label(footer, text="框半径(校正像素)").grid(row=0, column=2, sticky="w")
        ttk.Entry(footer, textvariable=self.radius_var, width=8).grid(row=0, column=3, padx=(5, 14), sticky="w")
        ttk.Label(footer, text="质量").grid(row=0, column=4, sticky="w")
        ttk.Combobox(footer, textvariable=self.quality_var, values=("accepted", "occluded", "uncertain"), state="readonly", width=11).grid(row=0, column=5, padx=(5, 14), sticky="w")
        ttk.Label(footer, text="备注").grid(row=0, column=6, sticky="w")
        ttk.Entry(footer, textvariable=self.notes_var).grid(row=0, column=7, padx=(5, 12), sticky="ew")
        ttk.Checkbutton(footer, text="我已逐点核对黑白类别和框位置", variable=self.reviewed_var).grid(row=1, column=0, columnspan=5, sticky="w", pady=(8, 0))
        ttk.Button(footer, text="保存真值并导出标签", command=self._save_annotation).grid(row=1, column=7, sticky="e", pady=(8, 0))
        ttk.Label(content, textvariable=self.status_var, foreground="#086b5f", wraplength=1050).grid(row=3, column=0, sticky="ew", pady=(8, 0))

    def _maximize_window(self) -> None:
        """Open the workbench maximized while retaining normal window controls."""
        try:
            self.window.state("zoomed")
        except tk.TclError:
            width = self.window.winfo_screenwidth()
            height = self.window.winfo_screenheight()
            self.window.geometry(f"{width}x{height}+0+0")

    def _delete_current_image(self) -> None:
        if self.current_path is None:
            self.status_var.set("当前没有可删除的原图。")
            return
        source = self.current_path
        confirmed = messagebox.askyesno(
            "删除本图",
            f"确定从正式数据集中删除“{source.name}”及其增强图、标签和预览吗？\n"
            "文件会移入 dataset/excluded，可在需要时恢复。",
            parent=self.window,
        )
        if not confirmed:
            return
        try:
            archive_root = exclude_image_family(
                self.dataset.root,
                source,
                reason="用户在标注工作台删除本图",
            )
        except (OSError, ValueError) as error:
            messagebox.showerror("删除失败", str(error), parent=self.window)
            return
        self.reviewed_truths.pop(source.name, None)
        self.current_path = None
        self.current_image = None
        self._refresh_image_list()
        self.status_var.set(f"已删除本图：{source.name} 已移入 {archive_root.relative_to(self.dataset.root)}。")

    def _load_calibration(self) -> None:
        try:
            self.default_calibration = BoardCalibration.from_json(self.calibration_path)
            self.calibration = self.default_calibration
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.default_calibration = None
            self.calibration = None
            self.status_var.set(f"无法载入棋盘标定：{error}")

    def _refresh_image_list(self) -> None:
        previous_name = self.current_path.name if self.current_path is not None else None
        self.raw_paths = sorted(self.dataset.raw_dir.glob("*.jpg"))
        try:
            self.reviewed_truths = load_reviewed_truths(
                self.dataset.root / "annotations" / "truth.jsonl"
            )
        except (OSError, ValueError) as error:
            self.reviewed_truths = {}
            self.status_var.set(f"无法读取现有标注进度：{error}")
        self.image_list.delete(0, "end")
        for path in self.raw_paths:
            self.image_list.insert("end", self._image_list_label(path))
        if self.raw_paths:
            if previous_name:
                matching = [index for index, path in enumerate(self.raw_paths) if path.name == previous_name]
                self.current_index = matching[0] if matching else 0
            else:
                self.current_index = next(
                    (index for index, path in enumerate(self.raw_paths) if path.name not in self.reviewed_truths),
                    0,
                )
            self.image_list.selection_clear(0, "end")
            self.image_list.selection_set(self.current_index)
            self.image_list.see(self.current_index)
            self._select_image(self.current_index)
        else:
            self.current_path = None
            self.status_var.set("dataset/raw 中暂无 JPG 原图。")

    def _on_image_selected(self, _event: tk.Event[tk.Misc]) -> None:
        selection = self.image_list.curselection()
        if selection:
            self._select_image(int(selection[0]))

    def _select_image(self, index: int) -> None:
        if not self.raw_paths:
            return
        self.current_index = max(0, min(index, len(self.raw_paths) - 1))
        self.current_path = self.raw_paths[self.current_index]
        self.image_list.selection_clear(0, "end")
        self.image_list.selection_set(self.current_index)
        self.image_list.see(self.current_index)
        self.reviewed_var.set(False)
        self.notes_var.set("")
        self.states = {}
        self.box_overrides = ()
        self.revision_mode = False
        self.current_proposals = ()
        image = read_color_image(self.current_path)
        if image is None:
            self.current_image = None
            self.status_var.set(f"无法读取图像：{self.current_path.name}")
            return
        self.current_image = image
        self.calibration = self.default_calibration
        self.image_calibration_detected = False
        if self.default_calibration is not None:
            try:
                self.calibration, self.image_calibration_detected = calibration_for_image(
                    image, self.default_calibration
                )
            except (OSError, ValueError, cv2.error):
                self.calibration = self.default_calibration
        calibration_note = (
            "本图已重新定位棋盘四角。"
            if self.image_calibration_detected
            else "未能从本图重新定位棋盘，暂使用当前全局标定。"
        )
        saved_truth = self.reviewed_truths.get(self.current_path.name)
        if saved_truth is not None:
            try:
                self.states = {
                    **{tuple(move): BLACK for move in saved_truth.get("black_moves", [])},
                    **{tuple(move): WHITE for move in saved_truth.get("white_moves", [])},
                }
                self.box_overrides = box_overrides_from_json(saved_truth.get("box_overrides", []))
                self.radius_var.set(str(saved_truth.get("radius", 18.0)))
                self.annotator_var.set(str(saved_truth.get("annotator", "operator")))
                self.quality_var.set(str(saved_truth.get("quality", "accepted")))
                self.notes_var.set(str(saved_truth.get("notes", "")))
            except (TypeError, ValueError) as error:
                self.status_var.set(f"已保存真值无法显示：{error}")
                self.states = {}
            self.status_var.set(
                "本图已经完成人工核验并导出标签，当前显示的是已保存真值。"
                f"{calibration_note} 如发现错误，请点击“修改已完成标注”解锁后逐点修订。"
            )
            self._render()
            return
        if self.calibration is None:
            self.status_var.set("缺少 calibration/board.json，无法生成校正网格。")
        else:
            try:
                self.current_proposals = propose_grid_stones(image, self.calibration)
                low = sum(proposal.confidence < 0.35 for proposal in self.current_proposals)
                self.status_var.set(
                    f"已载入候选：{len(self.current_proposals)} 个圆圈，其中低置信 {low} 个。"
                    f"{calibration_note} 红/蓝只表示程序猜测黑/白，不代表已确认；请点击“采用当前候选”后逐点核验。"
                )
            except (OSError, ValueError) as error:
                self.status_var.set(f"候选生成失败：{error}")
        self._render()

    def _accept_proposals(self) -> None:
        if self.current_path is not None and self.current_path.name in self.reviewed_truths:
            self.status_var.set("本图已有已保存真值；如需纠错，请先点击“修改已完成标注”。")
            return
        self.states = {proposal.move: proposal.class_id + 1 for proposal in self.current_proposals}
        self.box_overrides = ()
        self.reviewed_var.set(False)
        self.status_var.set("已载入程序候选。请把右图每个圆圈与左图同一位置的真实棋子逐一对照；不对的点击交点循环修改，完成后再勾选确认。")
        self._render()

    def _clear_states(self) -> None:
        if self.current_path is not None and self.current_path.name in self.reviewed_truths and not self.revision_mode:
            self.status_var.set("本图已锁定；如需清空后重标，请先点击“修改已完成标注”。")
            return
        self.states = {}
        self.box_overrides = ()
        self.reviewed_var.set(False)
        self.status_var.set("本图编辑已清空；可重新采用候选或点击网格交点添加棋子。")
        self._render()

    def _on_board_click(self, event: tk.Event[tk.Misc]) -> None:
        if self.calibration is None:
            return
        if self.current_path is not None and self.current_path.name in self.reviewed_truths and not self.revision_mode:
            self.status_var.set("本图已锁定；如需修改，请先点击“修改已完成标注”。")
            return
        move = display_point_to_move(event.x, event.y, self.calibration.board_size)
        if move is None:
            return
        self.states[move] = cycle_state(self.states.get(move, EMPTY))
        if self.states[move] == EMPTY:
            del self.states[move]
        self.box_overrides = tuple(item for item in self.box_overrides if item.move in self.states)
        self.reviewed_var.set(False)
        self._render()

    def _fit_box_overrides(self) -> None:
        if self.current_image is None or self.calibration is None:
            self.status_var.set("当前没有可贴合框的图片或棋盘标定。")
            return
        if self.current_path is not None and self.current_path.name in self.reviewed_truths and not self.revision_mode:
            self.status_var.set("本图已锁定；如需修正框位置，请先点击“修改已完成标注”。")
            return
        black, white = states_to_moves(self.states)
        if not black and not white:
            self.status_var.set("请先确认棋子位置，再贴合标注框。")
            return
        self.box_overrides = fit_grid_box_overrides(self.current_image, self.calibration, (*black, *white))
        self.reviewed_var.set(False)
        self.status_var.set(
            f"已按可识别轮廓贴合 {len(self.box_overrides)}/{len(black) + len(white)} 枚棋子框。"
            "请对照左图确认；未识别到轮廓的棋子继续使用格点框。"
        )
        self._render()

    def _edit_reviewed(self) -> None:
        if self.current_path is None or self.current_path.name not in self.reviewed_truths:
            self.status_var.set("当前图片还没有已保存真值，直接采用候选并核对即可。")
            return
        self.revision_mode = True
        self.reviewed_var.set(False)
        self.status_var.set(
            "已进入修订模式：现在可点击右图交点循环修改为空、黑、白。"
            "保存时会备份旧标签并追加新的修订版本。"
        )
        self._render()

    def _previous_image(self) -> None:
        if self.raw_paths:
            self._select_image((self.current_index - 1) % len(self.raw_paths))

    def _next_image(self) -> None:
        if self.raw_paths:
            self._select_image((self.current_index + 1) % len(self.raw_paths))

    def _next_unreviewed(self) -> None:
        if not self.raw_paths:
            return
        for offset in range(1, len(self.raw_paths) + 1):
            index = (self.current_index + offset) % len(self.raw_paths)
            if self.raw_paths[index].name not in self.reviewed_truths:
                self._select_image(index)
                return
        self.status_var.set("全部原图都已有已审核真值记录。")

    def _render(self) -> None:
        self.original_canvas.delete("all")
        self.board_canvas.delete("all")
        if self.current_image is None or self.calibration is None:
            return
        black, white = states_to_moves(self.states)
        try:
            annotations = project_grid_annotations(
                self.calibration,
                self.current_image.shape,
                black,
                white,
                radius=max(1.0, float(self.radius_var.get() or 18.0)),
                box_overrides=self.box_overrides,
            ) if (black or white) else ()
        except (OSError, ValueError):
            annotations = ()
        if annotations:
            preview = draw_grid_annotations(self.current_image, annotations)
        else:
            preview = self.current_image
        self._draw_image(self.original_canvas, preview, max_size=(520, 390), original=True)

        # Keep the real pixels outside the outer grid intersections in the
        # review image. The 520x520 warp remains the source for recognition
        # and label projection; this padded warp is display-only.
        display_size = BOARD_DISPLAY_SIZE
        border = int(round(
            BOARD_DISPLAY_MARGIN
            * self.calibration.output_size
            / (display_size - 2 * BOARD_DISPLAY_MARGIN)
        ))
        padded = self.calibration.warp_padded(self.current_image, padding=border)
        self._draw_image(self.board_canvas, padded, max_size=(int(display_size), int(display_size)), original=False)
        _margin, spacing = display_grid_geometry(self.calibration.board_size)
        board_scale = (display_size - 2 * BOARD_DISPLAY_MARGIN) / float(self.calibration.output_size - 1)
        for index in range(self.calibration.board_size):
            point = BOARD_DISPLAY_MARGIN + index * spacing
            self.board_canvas.create_line(0, point, display_size, point, fill="#65451f", width=1)
            self.board_canvas.create_line(point, 0, point, display_size, fill="#65451f", width=1)
        candidate_by_move = {proposal.move: proposal for proposal in self.current_proposals}
        for move, proposal in candidate_by_move.items():
            if move in self.states:
                continue
            x, y = self.calibration.board_pixel(move)
            x, y = BOARD_DISPLAY_MARGIN + x * board_scale, BOARD_DISPLAY_MARGIN + y * board_scale
            color = "#777777" if proposal.confidence < 0.35 else ("#e34b3b" if proposal.class_id == 0 else "#1976a8")
            radius = 9 if proposal.confidence >= 0.35 else 12
            options = {"outline": color, "width": 2}
            if proposal.confidence < 0.35:
                options["dash"] = (4, 2)
            self.board_canvas.create_oval(x - radius, y - radius, x + radius, y + radius, **options)
        for move, state in self.states.items():
            x, y = self.calibration.board_pixel(move)
            x, y = BOARD_DISPLAY_MARGIN + x * board_scale, BOARD_DISPLAY_MARGIN + y * board_scale
            if state == BLACK:
                self.board_canvas.create_oval(x - 12, y - 12, x + 12, y + 12, fill="#101010", outline="#ffffff", width=1)
            elif state == WHITE:
                self.board_canvas.create_oval(x - 12, y - 12, x + 12, y + 12, fill="#f8f8f8", outline="#101010", width=2)
        self.detail_var.set(
            f"第 {self.current_index + 1}/{len(self.raw_paths)} 张："
            f"总进度 {len(self.reviewed_truths)}/{len(self.raw_paths)}，"
            f"本图 {len(self.states)} 颗（黑 {len(black)} / 白 {len(white)}）"
        )

    def _draw_image(self, canvas: tk.Canvas, image, *, max_size: tuple[int, int], original: bool) -> None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail(max_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        if original:
            self.original_photo = photo
        else:
            self.board_photo = photo
        canvas.update_idletasks()
        canvas_width = max(int(canvas.winfo_width()), pil.width)
        canvas_height = max(int(canvas.winfo_height()), pil.height)
        x = max(0, (canvas_width - pil.width) // 2)
        y = max(0, (canvas_height - pil.height) // 2)
        canvas.create_image(x, y, image=photo, anchor="nw")

    def _save_annotation(self) -> None:
        if self.current_path is None or self.current_image is None or self.calibration is None:
            messagebox.showerror("无法保存", "当前没有可标注图片或棋盘标定。", parent=self.window)
            return
        if not self.reviewed_var.get():
            messagebox.showwarning("需要确认", "请逐点核对黑白类别和框位置，并勾选确认框。", parent=self.window)
            return
        existing_truth = self.reviewed_truths.get(self.current_path.name)
        if existing_truth is not None and not self.revision_mode:
            messagebox.showwarning(
                "图片已锁定",
                "本图已经存在已审核真值。如需纠错，请先点击“修改已完成标注”。",
                parent=self.window,
            )
            return
        black, white = states_to_moves(self.states)
        if not black and not white:
            messagebox.showwarning("没有棋子", "本图至少需要一个黑棋或白棋标注。", parent=self.window)
            return
        try:
            radius = float(self.radius_var.get())
            truth = build_truth(
                self.current_path,
                self.calibration,
                black,
                white,
                radius=radius,
                annotator=self.annotator_var.get().strip() or "operator",
                quality=self.quality_var.get(),
                notes=self.notes_var.get().strip(),
                revision=(int(existing_truth.get("revision", 1)) + 1) if existing_truth is not None else 1,
                box_overrides=self.box_overrides,
            )
            archive_path = None
            if existing_truth is not None:
                archive_path = archive_annotation_family(
                    self.dataset.root,
                    self.current_path,
                    reason="用户进入修订模式后重新确认黑白类别/位置",
                )
            label_paths = export_truth_family_labels(
                truth,
                self.current_path,
                self.calibration,
                self.dataset.root,
                overwrite=existing_truth is not None,
            )
            label_path = label_paths[0]
            log_path = self.dataset.root / "annotations" / "truth.jsonl"
            write_truth_record(log_path, truth)
            preview = draw_grid_annotations(
                self.current_image,
                project_grid_annotations(
                    self.calibration,
                    self.current_image.shape,
                    black,
                    white,
                    radius=truth.radius,
                    box_overrides=truth.box_overrides,
                ),
            )
            preview_path = self._unique_path(self.dataset.root / "annotation_previews" / f"{self.current_path.stem}_reviewed.jpg")
            write_image(preview_path, preview, [cv2.IMWRITE_JPEG_QUALITY, 95])
        except FileExistsError as error:
            messagebox.showerror("拒绝覆盖", str(error), parent=self.window)
            return
        except (OSError, ValueError) as error:
            messagebox.showerror("保存失败", str(error), parent=self.window)
            return
        revision_text = f"修订版 {truth.revision}" if truth.revision > 1 else "首版"
        archive_text = f"旧文件已备份到 {archive_path.name}" if archive_path is not None else ""
        self.status_var.set(
            f"已保存{revision_text}真值并导出 5 份标签：{label_path.name}；预览：{preview_path.name}。{archive_text}"
        )
        self.reviewed_truths[self.current_path.name] = truth.as_json()
        self.image_list.delete(self.current_index)
        self.image_list.insert(self.current_index, self._image_list_label(self.current_path))
        self.image_list.selection_set(self.current_index)
        self.revision_mode = False
        self.reviewed_var.set(False)

    def _image_list_label(self, path: Path) -> str:
        state = "已完成" if path.name in self.reviewed_truths else "待标注"
        return f"[{state}] {path.name}"

    @staticmethod
    def _unique_path(path: Path) -> Path:
        if not path.exists():
            return path
        for index in range(1, 10000):
            candidate = path.with_name(f"{path.stem}_{index:02d}{path.suffix}")
            if not candidate.exists():
                return candidate
        raise OSError(f"无法为预览生成不重复文件名：{path}")

    def close(self) -> None:
        if self.window.winfo_exists():
            self.window.destroy()


__all__ = [
    "AnnotationWorkbench",
    "BLACK",
    "BOARD_DISPLAY_MARGIN",
    "BOARD_DISPLAY_SIZE",
    "EMPTY",
    "WHITE",
    "cycle_state",
    "display_grid_geometry",
    "exclude_image_family",
    "display_point_to_move",
    "load_reviewed_truths",
    "states_to_moves",
]
