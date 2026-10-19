"""Tkinter diagnostic UI with an explicit, opt-in hardware debug panel."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from datetime import datetime
from pathlib import Path
import shutil
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageTk

from .calibration import BoardCalibration, check_calibration_alignment, detect_board_corners
from .board import GomokuBoard, Stone
from .coordinates import board_to_competition
from .dataset import DatasetManager
from .annotation_workbench import AnnotationWorkbench
from .foundation_calibration import FOUNDATION_TARGETS, FoundationHardwareCalibration, target_label
from .gameflow import AutonomousGomokuController, GameFlowUpdate, GamePhase
from .motion import ArmBoardCalibration, MotionPlan, MotionPlanner
from .services.camera import CameraService
from .services.hardware import HardwareCancelled, HardwareDebugError, HardwareDebugSession, HardwareReply
from .services.ports import list_serial_ports
from .session_log import SessionEventLog
from .voice import announce as announce_voice
from .vision import (
    BoardObstruction,
    BoardStoneRecognizer,
    BoardVisionResult,
    StableBoardTracker,
    detect_board_obstructions,
    detect_gameplay_occlusion,
)


DIRECT_MOVE_MIN_Z = 100.0
PREVIEW_STALL_FRAME_LIMIT = 90
PREVIEW_MAX_RECOVERY_ATTEMPTS = 2
# A hand briefly covering a board edge can make contour detection choose a
# completely unrelated quadrilateral.  Do not invalidate a usable board
# calibration on one such frame; recognition remains paused until the next
# checks confirm that the geometry is sound again.
CALIBRATION_DRIFT_FAILURE_LIMIT = 3
GAME_ACTION_TIMEOUT_MS = 90_000
GAME_VISUAL_CONFIRMATION_TIMEOUT_MS = 25_000


class DiagnosticApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("五子棋机器人")
        self.root.geometry("1320x840")
        self.root.minsize(1080, 720)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after_idle(self._show_startup_window)

        self.camera = CameraService()
        self.hardware = HardwareDebugSession(port="")
        self.hardware_busy = False
        self.hardware_cancel_event = threading.Event()
        self.closing = False
        self.camera_image: ImageTk.PhotoImage | None = None
        self.preview_running = False
        self.preview_generation = 0
        self.preview_after_id: str | None = None
        self.preview_failures = 0
        self.preview_last_sample: np.ndarray | None = None
        self.preview_stalled_frames = 0
        self.preview_recovery_attempts = 0
        self.last_frame = None
        self.calibration: BoardCalibration | None = None
        self.calibration_invalidated = False
        self.reference_invalidated = True
        self.calibration_last_checked = 0.0
        self.calibration_check_interval = 1.0
        self.calibration_drift_failures = 0
        self.project_root = Path(__file__).resolve().parents[1]
        self.session_log = SessionEventLog(self.project_root / "logs")
        self.calibration_path = self.project_root / "calibration" / "board.json"
        self.advanced_arm_calibration_path = self.project_root / "calibration" / "arm_board.json"
        self.foundation_calibration_path = self.project_root / "calibration" / "foundation_hardware.json"
        self.dataset = DatasetManager(self.project_root / "dataset")
        self.recognizer = BoardStoneRecognizer(self.project_root / "calibration" / "empty_board.png")
        self.board_tracker = StableBoardTracker(required_frames=3)
        self.last_vision_result: BoardVisionResult | None = None
        self.board_obstructions: tuple[BoardObstruction, ...] = ()
        self.corner_overlay_until = 0.0
        self.obstruction_last_checked = 0.0
        self.obstruction_check_interval = 0.6
        self.dataset_lighting_var = tk.StringVar(value="正常")
        self.dataset_edge_var = tk.BooleanVar(value=False)
        self.dataset_scene_var = tk.StringVar(value="")
        self.dataset_auto_id_var = tk.BooleanVar(value=True)
        self.dataset_notes_var = tk.StringVar(value="")
        self.dataset_batch_status_var = tk.StringVar(value="采集批次：尚未开始")
        self.dataset_batch_active = False
        self.dataset_batch_files: list[str] = []
        self.dataset_batch_counter = 0
        self.annotation_workbench: AnnotationWorkbench | None = None
        self.foundation_calibration = self._load_foundation_calibration()
        self.advanced_arm_calibration = self._load_advanced_arm_calibration()
        self.game_controller = AutonomousGomokuController(
            planner=MotionPlanner(self.advanced_arm_calibration)
        )
        self.dry_run_enabled = False
        self.dry_run_last_recognition = 0.0
        self.game_suggested_move: tuple[int, int] | None = None
        self.game_motion_plan: MotionPlan | None = None
        self.game_action_in_progress = False
        self.game_waiting_visual_confirmation = False
        self.game_finished = False
        self.game_pending_move: tuple[int, int] | None = None
        self.game_action_timeout_after_id: str | None = None
        self.game_confirmation_timeout_after_id: str | None = None
        # None means this app has not sent a turntable position command yet.
        # It avoids claiming knowledge of a position retained across restarts.
        self.turntable_duty: int | None = None
        self.foundation_target_labels = {target_label(key): key for key, _, _ in FOUNDATION_TARGETS}
        self.foundation_target_var = tk.StringVar(value=next(iter(self.foundation_target_labels)))
        self.foundation_target_xyz_vars = [tk.StringVar() for _ in range(3)]
        self.foundation_pickup_vars = [tk.StringVar(value=f"{value:g}") for value in self.foundation_calibration.pickup_point]
        self.foundation_standby_vars = [tk.StringVar(value=f"{value:g}") for value in self.foundation_calibration.standby_point]
        self.foundation_safe_z_var = tk.StringVar(value=f"{self.foundation_calibration.safe_z:g}")
        self.foundation_summary_var = tk.StringVar()
        self.foundation_measurement_var = tk.StringVar(
            value="视觉五点测量：先完成五点落子并保持棋盘、相机不动；测量只生成建议，不会改写坐标。"
        )

        self.status_var = tk.StringVar(value="诊断模式已就绪：硬件默认断开。")
        self.camera_var = tk.StringVar(value="1")
        self.calibration_var = tk.StringVar(value="棋盘尚未标定")
        self.dataset_var = tk.StringVar(value="原图 0 张，增强图 0 张")
        self.vision_var = tk.StringVar(value="视觉识别：先标定棋盘，再保存无棋子的空棋盘基准图")
        self.servo_var = tk.StringVar(value="取子机构舵机调试台：未动作（0°接子 / 90°取子）")
        self.arm_var = tk.StringVar(value="机械臂：未选择调试端口")
        self.arm_debug_var = tk.StringVar(value="机械臂调试台：未选择调试端口")
        self.game_phase_var = tk.StringVar(value="干运行未启动")
        self.game_message_var = tk.StringVar(value="视觉、AI 与动作规划均处于断开状态。")
        self.game_plan_var = tk.StringVar(value="不会发送机械臂、转盘或气泵命令。")
        self.match_result_var = tk.StringVar(value="等待对局开始")
        self.match_voice_enabled_var = tk.BooleanVar(value=True)
        self.game_hardware_enabled_var = tk.BooleanVar(value=False)
        # This is intentionally separate from the one-shot supervised control.
        # A team may explicitly arm it once at the beginning of a match, after
        # which every camera-confirmed black turn is executed without a click.
        self.game_auto_execution_var = tk.BooleanVar(value=False)

        self._style()
        self._build_layout()
        self._update_dataset_status()
        self._refresh_foundation_table()
        self._refresh_ports()
        self._refresh_camera_indexes()
        self._record_session_event("application_started", hardware_connected=False)

    def _record_session_event(self, event: str, **details) -> None:
        """Keep evidence logging best-effort so a disk problem cannot move hardware."""
        session_log = getattr(self, "session_log", None)
        if session_log is None:
            return
        try:
            session_log.record(event, **details)
        except OSError:
            # The visible safety gates must still be able to stop an action if
            # an evidence drive becomes unavailable.
            pass

    def _style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Header.TLabel", font=("Microsoft YaHei UI", 16, "bold"), foreground="#123042")
        style.configure("Muted.TLabel", foreground="#52616b")
        style.configure("Status.TLabel", foreground="#086b5f")
        style.configure(
            "MatchResult.TLabel",
            font=("Microsoft YaHei UI", 18, "bold"),
            foreground="#075f45",
            background="#e4f4ed",
            padding=(14, 8),
            anchor="center",
        )
        style.configure(
            "MatchPending.TLabel",
            font=("Microsoft YaHei UI", 13, "bold"),
            foreground="#335365",
            background="#edf3f6",
            padding=(12, 7),
            anchor="center",
        )
        style.configure("TButton", padding=(10, 6))
        style.configure("Danger.TButton", foreground="#8a3b12")

    def _show_startup_window(self) -> None:
        """Show the diagnostic window maximized when launched from Explorer."""
        self.root.update_idletasks()
        self.root.deiconify()
        try:
            self.root.state("zoomed")
        except tk.TclError:
            # Fallback for window managers that do not support Tk's zoomed state.
            width, height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            self.root.geometry(f"{width}x{height}+0+0")
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(400, lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()

    def _load_foundation_calibration(self) -> FoundationHardwareCalibration:
        if not self.foundation_calibration_path.exists():
            return FoundationHardwareCalibration()
        try:
            return FoundationHardwareCalibration.from_json(self.foundation_calibration_path)
        except (OSError, ValueError, KeyError):
            return FoundationHardwareCalibration()

    def _load_advanced_arm_calibration(self) -> ArmBoardCalibration | None:
        """Load only a previously camera-confirmed four-corner calibration."""
        if not self.advanced_arm_calibration_path.exists():
            return None
        try:
            return ArmBoardCalibration.from_json(self.advanced_arm_calibration_path)
        except (OSError, ValueError, KeyError):
            return None

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=3)
        container.columnconfigure(1, weight=2)
        container.rowconfigure(1, weight=1)

        header = ttk.Frame(container)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="五子棋机器人", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text=" ", style="Muted.TLabel").grid(row=0, column=1, sticky="e")
        ttk.Label(container, textvariable=self.status_var, style="Status.TLabel", wraplength=1100).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(12, 0)
        )

        vision = ttk.LabelFrame(container, text="摄像头调试台", padding=12)
        vision.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        vision.columnconfigure(0, weight=1)
        vision.rowconfigure(1, weight=1)

        controls = ttk.Frame(vision)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(controls, text="摄像头编号").pack(side="left")
        self.camera_combo = ttk.Combobox(controls, textvariable=self.camera_var, state="readonly", width=8)
        self.camera_combo.pack(side="left", padx=(8, 8))
        self.refresh_camera_button = ttk.Button(controls, text="刷新列表", command=self._refresh_camera_indexes)
        self.refresh_camera_button.pack(side="left")
        self.start_camera_button = ttk.Button(controls, text="启动预览", command=self._start_camera)
        self.start_camera_button.pack(side="left", padx=(8, 0))
        self.stop_camera_button = ttk.Button(controls, text="停止预览", command=self._stop_camera, state="disabled")
        self.stop_camera_button.pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="自动找棋盘", command=self._calibrate_board).pack(side="left", padx=(8, 0))

        self.preview = ttk.Label(vision, text="选择摄像头后点击“启动预览”", anchor="center", style="Muted.TLabel")
        self.preview.grid(row=1, column=0, sticky="nsew")

        right = ttk.Frame(container)
        right.grid(row=1, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        tabs = ttk.Notebook(right)
        tabs.grid(row=0, column=0, sticky="nsew")
        setup_tab = ttk.Frame(tabs, padding=8)
        dataset_tab = ttk.Frame(tabs, padding=8)
        tuning_tab = ttk.Frame(tabs, padding=8)
        game_tab = ttk.Frame(tabs, padding=8)
        tabs.add(setup_tab, text="基础调试")
        tabs.add(dataset_tab, text="批量采集")
        tabs.add(tuning_tab, text="五黑点坐标微调")
        tabs.add(game_tab, text="进阶赛干运行")
        setup_tab.columnconfigure(0, weight=1)

        ports = ttk.LabelFrame(setup_tab, text="串口与硬件连接", padding=12)
        ports.grid(row=0, column=0, sticky="ew")
        ports.columnconfigure(0, weight=1)
        port_controls = ttk.Frame(ports)
        port_controls.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Label(port_controls, text="调试端口").pack(side="left")
        self.hardware_port_var = tk.StringVar(value="")
        self.hardware_port_combo = ttk.Combobox(port_controls, textvariable=self.hardware_port_var, state="readonly", width=10)
        self.hardware_port_combo.pack(side="left", padx=(8, 8))
        ttk.Button(port_controls, text="刷新串口", command=self._refresh_ports).pack(side="left")
        self.hardware_connect_button = ttk.Button(port_controls, text="连接并进入调试", command=self._connect_hardware)
        self.hardware_connect_button.pack(side="left", padx=(8, 0))
        ttk.Button(port_controls, text="退出调试", command=self._disconnect_hardware).pack(side="left", padx=(8, 0))
        self.port_text = tk.Text(ports, height=5, wrap="word", relief="flat", background="#f5f7f7", state="disabled")
        self.port_text.grid(row=1, column=0, columnspan=2, sticky="ew")

        hardware = ttk.LabelFrame(setup_tab, text="硬件调试台", padding=12)
        hardware.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        hardware.columnconfigure(0, weight=1)
        hardware.columnconfigure(1, weight=1)
        ttk.Label(hardware, textvariable=self.servo_var, style="Muted.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Button(hardware, text="转盘到接子位（0°）", command=self._turntable_receive).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(hardware, text="转盘到取子位（90°）", command=self._turntable_pickup).grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=(6, 0))
        ttk.Label(hardware, textvariable=self.arm_debug_var, style="Muted.TLabel").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        ttk.Button(hardware, text="机械臂取子", command=self._pickup_hardware).grid(
            row=3, column=0, sticky="ew", pady=(4, 0)
        )
        ttk.Button(hardware, text="机械臂放子", command=self._drop_hardware).grid(
            row=3, column=1, sticky="ew", padx=(6, 0), pady=(4, 0)
        )
        ttk.Button(hardware, text="停止当前动作", command=lambda: self._pump(2, "停止")).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(4, 0)
        )
        ttk.Button(
            hardware,
            text="机械臂调试",
            command=self._debug_arm_to_center_target,
        ).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(hardware, text="初始化机械臂", command=self._initialize_hardware).grid(row=6, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(hardware, text="连续完成五黑点", command=self._place_all_foundation_targets).grid(row=6, column=1, sticky="ew", padx=(6, 0), pady=(6, 0))
        ttk.Label(hardware, text="空载坐标移动（mm）", style="Muted.TLabel").grid(row=7, column=0, columnspan=2, sticky="w", pady=(10, 0))
        coordinate_controls = ttk.Frame(hardware)
        coordinate_controls.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.hardware_coordinate_vars = [
            tk.StringVar(value=f"{value:g}") for value in self.foundation_calibration.standby_point
        ]
        for label, variable in zip(("X", "Y", "Z"), self.hardware_coordinate_vars):
            ttk.Label(coordinate_controls, text=label).pack(side="left")
            ttk.Entry(coordinate_controls, textvariable=variable, width=8).pack(side="left", padx=(3, 8))
        ttk.Button(coordinate_controls, text="执行空载移动", command=self._move_hardware).pack(side="left", padx=(8, 0))

        machine_status = ttk.LabelFrame(setup_tab, text="机器状态", padding=12)
        machine_status.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(machine_status, textvariable=self.servo_var, style="Muted.TLabel").pack(anchor="w")
        ttk.Label(machine_status, textvariable=self.arm_var, style="Muted.TLabel").pack(anchor="w", pady=(4, 0))

        self._build_dataset_batch_tab(dataset_tab)
        self._build_five_point_tuning_tab(tuning_tab)
        self._build_game_dry_run_tab(game_tab)

    def _build_game_dry_run_tab(self, tab: ttk.Frame) -> None:
        """Build the competition workflow in the order an operator uses it."""
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)

        overview = ttk.LabelFrame(tab, text="进阶赛操作台", padding=10)
        overview.grid(row=0, column=0, sticky="ew")
        overview.columnconfigure(0, weight=1)
        overview.columnconfigure(1, weight=1)

        prepare = ttk.LabelFrame(overview, text="1. 准备棋盘", padding=8)
        prepare.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(prepare, text="保存空盘基准", command=self._capture_empty_board).pack(side="left")
        ttk.Button(prepare, text="识别当前棋局", command=lambda: self._recognize_board(False)).pack(side="left", padx=(8, 0))

        match = ttk.LabelFrame(overview, text="2. 开始或重置", padding=8)
        match.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        ttk.Button(match, text="开始对局", command=self._start_dry_run).pack(side="left")
        ttk.Button(match, text="停止对局", command=self._stop_dry_run).pack(side="left", padx=(8, 0))
        ttk.Button(match, text="准备下一局", command=self._prepare_next_match).pack(side="left", padx=(8, 0))

        execution = ttk.LabelFrame(overview, text="3. 执行控制", padding=8)
        execution.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Checkbutton(
            execution,
            text="受监督执行",
            variable=self.game_hardware_enabled_var,
            command=self._on_game_execution_toggle,
        ).pack(side="left")
        ttk.Button(execution, text="执行本次黑棋", command=self._execute_game_plan).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(
            execution,
            text="比赛自动执行",
            variable=self.game_auto_execution_var,
            command=self._on_game_auto_execution_toggle,
        ).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(execution, text="胜负语音", variable=self.match_voice_enabled_var).pack(side="left", padx=(8, 0))
        ttk.Button(
            execution,
            text="紧急停止本手",
            command=self._emergency_stop_game_action,
            style="Danger.TButton",
        ).pack(side="left", padx=(8, 0))
        ttk.Label(overview, textvariable=self.game_phase_var, style="Status.TLabel").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        ttk.Label(overview, textvariable=self.game_message_var, style="Muted.TLabel", wraplength=570).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(3, 0)
        )

        board_frame = ttk.LabelFrame(tab, text="实时 13×13 棋局（竞赛坐标：右上为 (1,1)）", padding=8)
        board_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.match_result_label = ttk.Label(
            board_frame,
            textvariable=self.match_result_var,
            style="MatchPending.TLabel",
        )
        self.match_result_label.pack(fill="x", padx=4, pady=(0, 8))
        self.game_canvas = tk.Canvas(board_frame, width=330, height=330, background="#deb887", highlightthickness=0)
        self.game_canvas.pack(anchor="center")

        plan_frame = ttk.LabelFrame(tab, text="AI 建议与动作计划", padding=10)
        plan_frame.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        ttk.Label(plan_frame, textvariable=self.game_plan_var, style="Muted.TLabel", justify="left", wraplength=570).pack(anchor="w")
        self._draw_game_board(GomokuBoard())

    def _build_dataset_batch_tab(self, dataset_tab: ttk.Frame) -> None:
        """Build the visible multi-capture workflow without changing the old controls."""

        dataset_tab.columnconfigure(0, weight=1)
        dataset_tab.rowconfigure(2, weight=1)
        intro = ttk.LabelFrame(dataset_tab, text="批量采集说明", padding=12)
        intro.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            intro,
            text=(
                "先启动左侧摄像头预览，并确认画面为 1920x1080。默认不需要手写编号，点击“开始新批次”会自动生成批次号。\n"
                "之后每摆好一个新的棋子布局，只需点击一次“保存当前帧到批次”；全部完成后点击“结束本批次”。\n"
                "每张图片会自动记录批次序号；文件名、光照档位、边缘标记和备注都会写入 manifest。\n"
                "本批采集期间请固定相机；如果相机被移动或碰到，先结束批次，重新“自动找棋盘”后再开始新批次。"
            ),
            justify="left",
            wraplength=650,
            style="Muted.TLabel",
        ).pack(anchor="w")

        metadata = ttk.LabelFrame(dataset_tab, text="本批次条件", padding=12)
        metadata.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        metadata.columnconfigure(7, weight=1)
        ttk.Label(metadata, text="采集光照").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            metadata,
            textvariable=self.dataset_lighting_var,
            values=("低光", "正常", "高光"),
            state="readonly",
            width=9,
        ).grid(row=0, column=1, padx=(6, 18), sticky="w")
        ttk.Checkbutton(metadata, text="含棋盘边缘落子", variable=self.dataset_edge_var).grid(
            row=0, column=2, padx=(0, 18), sticky="w"
        )
        ttk.Label(metadata, text="场景编号").grid(row=0, column=3, sticky="w")
        ttk.Entry(metadata, textvariable=self.dataset_scene_var, width=16).grid(row=0, column=4, padx=(6, 18), sticky="w")
        ttk.Checkbutton(metadata, text="自动编号", variable=self.dataset_auto_id_var).grid(row=0, column=5, sticky="w")
        ttk.Label(metadata, text="备注").grid(row=0, column=6, sticky="w")
        ttk.Entry(metadata, textvariable=self.dataset_notes_var).grid(row=0, column=7, padx=(6, 0), sticky="ew")

        actions = ttk.Frame(dataset_tab)
        actions.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        actions.columnconfigure(0, weight=1)
        actions.rowconfigure(1, weight=1)
        action_bar = ttk.Frame(actions)
        action_bar.grid(row=0, column=0, sticky="ew")
        ttk.Button(action_bar, text="开始新批次", command=self._begin_dataset_batch).pack(side="left")
        ttk.Button(action_bar, text="保存当前帧到批次", command=self._save_dataset_frame_to_batch).pack(side="left", padx=(8, 0))
        ttk.Button(action_bar, text="结束本批次", command=self._finish_dataset_batch).pack(side="left", padx=(8, 0))
        ttk.Button(action_bar, text="批量生成四种增强图", command=self._augment_dataset).pack(side="left", padx=(8, 0))
        ttk.Button(action_bar, text="打开标注工作台", command=self._open_annotation_workbench).pack(side="left", padx=(8, 0))
        ttk.Label(action_bar, textvariable=self.dataset_batch_status_var, style="Muted.TLabel", wraplength=360).pack(
            side="left", padx=(12, 0), fill="x", expand=True
        )

        files_frame = ttk.LabelFrame(actions, text="本批次已保存原图（完整列表）", padding=8)
        files_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        files_frame.columnconfigure(0, weight=1)
        files_frame.rowconfigure(0, weight=1)
        self.dataset_batch_list = tk.Listbox(files_frame, height=16, activestyle="none")
        self.dataset_batch_list.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(files_frame, orient="vertical", command=self.dataset_batch_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.dataset_batch_list.configure(yscrollcommand=scrollbar.set)

    def _build_foundation_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)

        summary = ttk.LabelFrame(tab, text="基础赛五黑点取放", padding=12)
        summary.grid(row=0, column=0, sticky="ew")
        ttk.Label(summary, textvariable=self.foundation_summary_var, style="Muted.TLabel", wraplength=540).pack(anchor="w")

        base = ttk.LabelFrame(tab, text="安全基础坐标（mm）", padding=12)
        base.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        for row, (title, variables) in enumerate(
            (("取子点", self.foundation_pickup_vars), ("待命点", self.foundation_standby_vars))
        ):
            ttk.Label(base, text=title).grid(row=row, column=0, sticky="w", pady=(0, 6))
            for column, (axis, variable) in enumerate(zip(("X", "Y", "Z"), variables), start=1):
                ttk.Label(base, text=axis).grid(row=row, column=column * 2 - 1, sticky="e", padx=(8, 3), pady=(0, 6))
                ttk.Entry(base, textvariable=variable, width=8).grid(row=row, column=column * 2, pady=(0, 6))
        ttk.Label(base, text="安全Z").grid(row=2, column=0, sticky="w")
        ttk.Entry(base, textvariable=self.foundation_safe_z_var, width=8).grid(row=2, column=2, sticky="w")
        ttk.Button(base, text="保存基础坐标", command=self._save_foundation_base).grid(row=2, column=6, sticky="e")

        points = ttk.LabelFrame(tab, text="五个黑圆点标定", padding=12)
        points.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        points.columnconfigure(0, weight=1)
        points.rowconfigure(0, weight=1)
        self.foundation_tree = ttk.Treeview(
            points,
            columns=("board", "x", "y", "z", "status"),
            show="tree headings",
            height=5,
        )
        self.foundation_tree.heading("#0", text="黑点")
        self.foundation_tree.heading("board", text="棋盘坐标")
        for axis in ("x", "y", "z"):
            self.foundation_tree.heading(axis, text=axis.upper())
        self.foundation_tree.heading("status", text="状态")
        self.foundation_tree.column("#0", width=100, stretch=True)
        self.foundation_tree.column("board", width=80, anchor="center")
        for axis in ("x", "y", "z"):
            self.foundation_tree.column(axis, width=64, anchor="center")
        self.foundation_tree.column("status", width=70, anchor="center")
        self.foundation_tree.grid(row=0, column=0, columnspan=9, sticky="nsew")

        ttk.Label(points, text="当前黑点").grid(row=1, column=0, sticky="w", pady=(10, 0))
        selector = ttk.Combobox(
            points,
            textvariable=self.foundation_target_var,
            values=tuple(self.foundation_target_labels),
            state="readonly",
            width=20,
        )
        selector.grid(row=1, column=1, columnspan=2, sticky="w", pady=(10, 0))
        selector.bind("<<ComboboxSelected>>", self._load_selected_foundation_target)
        for index, (axis, variable) in enumerate(zip(("X", "Y", "Z"), self.foundation_target_xyz_vars)):
            column = 3 + index * 2
            ttk.Label(points, text=axis).grid(row=1, column=column, sticky="e", padx=(6, 2), pady=(10, 0))
            ttk.Entry(points, textvariable=variable, width=7).grid(row=1, column=column + 1, sticky="w", pady=(10, 0))

        commands = ttk.Frame(points)
        commands.grid(row=2, column=0, columnspan=9, sticky="ew", pady=(10, 0))
        ttk.Button(commands, text="保存当前黑点", command=self._save_foundation_target).pack(side="left")
        ttk.Button(commands, text="空载到达当前黑点", command=self._move_foundation_target).pack(side="left", padx=(8, 0))
        ttk.Button(commands, text="取放到当前黑点", command=self._place_foundation_target).pack(side="left", padx=(8, 0))
        ttk.Button(commands, text="连续完成五黑点", command=self._place_all_foundation_targets).pack(side="left", padx=(8, 0))

    def _build_five_point_tuning_tab(self, tab: ttk.Frame) -> None:
        """Build the compact per-point coordinate editor used during placement tuning."""
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)

        instruction = ttk.LabelFrame(tab, text="五黑点坐标微调", padding=12)
        instruction.grid(row=0, column=0, sticky="ew")
        ttk.Label(instruction, textvariable=self.foundation_summary_var, style="Status.TLabel", wraplength=560).pack(
            anchor="w"
        )
        ttk.Label(instruction, textvariable=self.foundation_measurement_var, style="Muted.TLabel", wraplength=620, justify="left").pack(
            anchor="w", pady=(6, 0)
        )

        points = ttk.LabelFrame(tab, text="已保存的落子坐标（mm）", padding=12)
        points.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        points.columnconfigure(0, weight=1)
        points.rowconfigure(0, weight=1)
        self.foundation_tree = ttk.Treeview(
            points,
            columns=("board", "x", "y", "z", "status"),
            show="tree headings",
            height=5,
        )
        self.foundation_tree.heading("#0", text="黑点")
        self.foundation_tree.heading("board", text="棋盘坐标")
        for axis in ("x", "y", "z"):
            self.foundation_tree.heading(axis, text=axis.upper())
        self.foundation_tree.heading("status", text="状态")
        self.foundation_tree.column("#0", width=100, stretch=True)
        self.foundation_tree.column("board", width=80, anchor="center")
        for axis in ("x", "y", "z"):
            self.foundation_tree.column(axis, width=72, anchor="center")
        self.foundation_tree.column("status", width=70, anchor="center")
        self.foundation_tree.grid(row=0, column=0, sticky="nsew")
        self.foundation_tree.bind("<<TreeviewSelect>>", self._select_foundation_tree_target)

        editor = ttk.Frame(points)
        editor.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(editor, text="当前黑点").pack(side="left")
        selector = ttk.Combobox(
            editor,
            textvariable=self.foundation_target_var,
            values=tuple(self.foundation_target_labels),
            state="readonly",
            width=18,
        )
        selector.pack(side="left", padx=(8, 12))
        selector.bind("<<ComboboxSelected>>", self._load_selected_foundation_target)
        for axis, variable in zip(("X", "Y", "Z"), self.foundation_target_xyz_vars):
            ttk.Label(editor, text=axis).pack(side="left")
            ttk.Entry(editor, textvariable=variable, width=8).pack(side="left", padx=(3, 8))

        actions = ttk.Frame(points)
        actions.grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Button(actions, text="保存当前点", command=self._save_foundation_target).pack(side="left")
        ttk.Button(actions, text="空载验证当前点", command=self._move_foundation_target).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="取放验证当前点", command=self._place_foundation_target).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="视觉测量已落五点（只读）", command=self._measure_foundation_visual_offsets).pack(side="left", padx=(8, 0))

    def _set_port_text(self, content: str) -> None:
        self.port_text.configure(state="normal")
        self.port_text.delete("1.0", "end")
        self.port_text.insert("1.0", content)
        self.port_text.configure(state="disabled")

    def _set_arm_status(self, status: str) -> None:
        """Keep the compact arm-debug heading and machine status in sync."""
        self.arm_var.set(f"机械臂：{status}")
        self.arm_debug_var.set(f"机械臂调试台：{status}")

    def _refresh_ports(self) -> None:
        ports, error = list_serial_ports()
        if error:
            self._set_port_text(error)
            return
        if not ports:
            self._set_port_text("没有发现串口。请检查 USB 线和 CH340 驱动。")
            return
        lines = []
        devices = []
        for port in ports:
            tag = "  [USB-SERIAL CH340]" if "CH340" in port.description.upper() else ""
            devices.append(port.device)
            lines.append(f"{port.device}: {port.description}{tag}")
        self.hardware_port_combo["values"] = devices
        current_port = self.hardware_port_var.get().strip().upper()
        if current_port not in {device.upper() for device in devices}:
            # COM numbers change when the robot is connected to another USB port
            # or computer, so use the first currently enumerated port rather
            # than retaining a historical port number.
            selected_port = devices[0]
            self.hardware_port_var.set(selected_port)
            self._set_arm_status(f"未连接 {selected_port}")
        self._set_port_text("\n".join(lines))

    def _refresh_camera_indexes(self) -> None:
        indexes = CameraService.available_indexes()
        values = [str(index) for index in indexes] or ["0"]
        self.camera_combo["values"] = values
        if self.camera_var.get() not in values:
            self.camera_var.set(values[0])
        if indexes:
            self.status_var.set(f"发现摄像头编号：{', '.join(values)}。点击“启动预览”后才会打开。")
        else:
            self.status_var.set("暂未检测到可用摄像头；可检查 USB 摄像头连接后刷新。")

    def _set_preview_controls(self, running: bool) -> None:
        """Keep camera controls consistent with the single preview loop."""
        state = "disabled" if running else "normal"
        self.start_camera_button.configure(state=state)
        self.refresh_camera_button.configure(state=state)
        self.camera_combo.configure(state="disabled" if running else "readonly")
        self.stop_camera_button.configure(state="normal" if running else "disabled")

    def _reset_preview_watchdog(self) -> None:
        self.preview_last_sample = None
        self.preview_stalled_frames = 0

    def _preview_is_stalled(self, frame: np.ndarray) -> bool:
        """Detect a driver returning the same captured image repeatedly.

        The comparison uses a small raw BGR sample and requires three seconds
        of exact repeats, so a normal still scene is not mistaken for a frozen
        preview merely because nothing in view is moving.
        """
        sample = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        if self.preview_last_sample is None or not np.array_equal(sample, self.preview_last_sample):
            self.preview_last_sample = sample
            self.preview_stalled_frames = 0
            return False
        self.preview_stalled_frames += 1
        return self.preview_stalled_frames >= PREVIEW_STALL_FRAME_LIMIT

    def _recover_stalled_preview(self, reason: str) -> None:
        """Reopen a camera once its driver has stopped producing fresh frames."""
        index = self.camera.index
        if index is None:
            return
        if self.preview_recovery_attempts >= PREVIEW_MAX_RECOVERY_ATTEMPTS:
            self.preview_running = False
            self.camera.stop()
            self._set_preview_controls(False)
            self.preview.configure(image="", text="摄像头画面已停滞，请重新启动预览")
            self.camera_image = None
            self.status_var.set(f"{reason}；已自动重开两次仍无新画面，请检查摄像头连接后重新启动预览。")
            return

        self.preview_recovery_attempts += 1
        self.preview_generation += 1
        generation = self.preview_generation
        self.camera.stop()
        ok, message = self.camera.start(index)
        if not ok:
            self.preview_running = False
            self._set_preview_controls(False)
            self.preview.configure(image="", text="摄像头自动恢复失败")
            self.camera_image = None
            self.status_var.set(f"{reason}；自动重开失败：{message}")
            return

        self._reset_preview_watchdog()
        self.preview_failures = 0
        self.last_frame = None
        self.calibration_invalidated = True
        self.calibration = None
        self.reference_invalidated = True
        self.recognizer.reset_reference_cache()
        self.board_tracker.reset()
        self.board_obstructions = ()
        self.corner_overlay_until = 0.0
        self.obstruction_last_checked = 0.0
        self.status_var.set(f"{reason}；已自动重新打开摄像头。请重新点击“自动找棋盘”。")
        self.preview_after_id = self.root.after(100, lambda: self._update_preview(generation))

    def _start_camera(self) -> None:
        # Starting a second capture would close and reopen the active camera
        # while also scheduling a second Tk preview loop.  Both reads then
        # compete for the same device and make the interface appear frozen.
        if self.preview_running or self.camera.index is not None:
            self.preview_running = True
            self._set_preview_controls(True)
            self.status_var.set("摄像头预览已经启动，无需重复点击“启动预览”。")
            return
        try:
            index = int(self.camera_var.get())
        except ValueError:
            self.status_var.set("摄像头编号无效。")
            return
        ok, message = self.camera.start(index)
        self.status_var.set(message)
        if not ok:
            self.preview_running = False
            self._set_preview_controls(False)
            return
        self.preview_running = True
        self.preview_generation += 1
        self._set_preview_controls(True)
        self.preview_failures = 0
        self.preview_recovery_attempts = 0
        self._reset_preview_watchdog()
        # The physical view can change between camera starts. Require a
        # fresh calibration before this run can save a reference image.
        self.calibration_invalidated = True
        self.calibration = None
        self.reference_invalidated = True
        self.recognizer.reset_reference_cache()
        self.calibration_last_checked = 0.0
        self.board_tracker.reset()
        self.last_vision_result = None
        self.board_obstructions = ()
        self.corner_overlay_until = 0.0
        self.obstruction_last_checked = 0.0
        self._update_preview(self.preview_generation)

    def _stop_camera(self) -> None:
        self.preview_running = False
        self.preview_generation += 1
        if self.preview_after_id is not None:
            try:
                self.root.after_cancel(self.preview_after_id)
            except tk.TclError:
                pass
            self.preview_after_id = None
        self.camera.stop()
        self._set_preview_controls(False)
        self.preview.configure(image="", text="摄像头预览已停止")
        self.camera_image = None
        self.preview_failures = 0
        self.preview_recovery_attempts = 0
        self._reset_preview_watchdog()
        self.last_frame = None
        self.calibration_invalidated = True
        self.calibration = None
        self.reference_invalidated = True
        self.recognizer.reset_reference_cache()
        self.board_obstructions = ()
        self.corner_overlay_until = 0.0
        self.obstruction_last_checked = 0.0
        self.status_var.set(f"摄像头预览已停止；{self._hardware_connection_text()}")

    def _draw_board_corner_overlay(self, frame):
        """Draw the four camera-detected board corners on a display-only frame."""
        if self.calibration is None or self.calibration_invalidated:
            return frame
        obstructions = getattr(self, "board_obstructions", ())
        if not obstructions and time.monotonic() >= getattr(self, "corner_overlay_until", 0.0):
            return frame
        display = frame.copy()
        points = np.asarray(self.calibration.corners, dtype=np.float32)
        polygon = np.round(points).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(display, [polygon], True, (0, 210, 0), 4, cv2.LINE_AA)
        # OpenCV's built-in Hershey font cannot render Chinese.  Numbered
        # markers remain readable on every Windows installation: 1=左上、2=右上、
        # 3=右下、4=左下。
        labels = ("1", "2", "3", "4")
        for label, (x, y) in zip(labels, points):
            center = (int(round(float(x))), int(round(float(y))))
            cv2.circle(display, center, 10, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(display, center, 14, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, label, (center[0] + 12, center[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 210, 0), 2, cv2.LINE_AA)
        if obstructions:
            inverse = np.linalg.inv(self.calibration.matrix)
            for obstruction in obstructions:
                x, y, width, height = obstruction.x, obstruction.y, obstruction.width, obstruction.height
                warped_box = np.asarray(
                    ((x, y), (x + width, y), (x + width, y + height), (x, y + height)),
                    dtype=np.float32,
                ).reshape((-1, 1, 2))
                original_box = cv2.perspectiveTransform(warped_box, inverse)
                cv2.polylines(display, [np.round(original_box).astype(np.int32)], True, (0, 0, 255), 3, cv2.LINE_AA)
                label_point = tuple(np.round(original_box[0, 0]).astype(int))
                cv2.putText(display, "OBS", label_point, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return display

    def _refresh_board_obstruction_overlay(self, *, force: bool = False) -> None:
        """Keep red obstruction boxes tied to the live frame, not an old one."""
        if self.calibration is None or self.calibration_invalidated or self.last_frame is None:
            return
        now = time.monotonic()
        if not force and now - self.obstruction_last_checked < self.obstruction_check_interval:
            return
        self.obstruction_last_checked = now
        try:
            clearance = detect_board_obstructions(self.last_frame, self.calibration)
        except ValueError:
            return
        previous = self.board_obstructions
        self.board_obstructions = clearance.obstructions
        if clearance.obstructions:
            self.corner_overlay_until = float("inf")
            if clearance.obstructions != previous:
                count = len(clearance.obstructions)
                self.calibration_var.set(f"棋盘已标定，但检测到 {count} 处异物或遗留棋子")
                self.status_var.set(f"棋盘当前发现 {count} 处异物或遗留棋子；红框标记会随实时画面更新。")
            return
        if previous:
            self.corner_overlay_until = now + 10.0
            self.calibration_var.set("棋盘已标定且无异物：外角标记将在 10 秒后自动隐藏")
            self.status_var.set("棋盘异物已清除；过期红框已移除，外角标记将在 10 秒后隐藏。")

    def _update_preview(self, generation: int | None = None) -> None:
        if generation is None:
            generation = self.preview_generation
        if not self.preview_running or generation != self.preview_generation or self.camera.index is None:
            return
        frame = self.camera.read()
        if frame is None:
            self.preview_failures += 1
            if self.preview_failures == 15:
                self._recover_stalled_preview(f"摄像头 {self.camera.index} 连续读取画面失败")
                return
        else:
            self.preview_failures = 0
            if self._preview_is_stalled(frame):
                self._recover_stalled_preview("摄像头画面连续三秒未刷新")
                return
            self.last_frame = frame.copy()
            if self._runtime_board_guards_enabled():
                self._monitor_calibration_drift()
                self._refresh_board_obstruction_overlay()
            else:
                # In a human-supervised game, stones and hands are expected
                # to move inside the board.  The clearance checker is for an
                # empty board only, so do not paint normal moves as OBS or
                # run a costly geometry check on every preview frame.
                self.board_obstructions = ()
                self.corner_overlay_until = 0.0
            now = time.monotonic()
            if self.dry_run_enabled and now - self.dry_run_last_recognition >= 0.20:
                self.dry_run_last_recognition = now
                self._recognize_board(automatic=True)
            max_width, max_height = 700, 500
            height, width = frame.shape[:2]
            ratio = min(max_width / width, max_height / height, 1.0)
            frame = self.camera.correct_color(frame)
            frame = self._draw_board_corner_overlay(frame)
            frame = cv2.resize(frame, (int(width * ratio), int(height * ratio)))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            self.camera_image = ImageTk.PhotoImage(image=image)
            self.preview.configure(image=self.camera_image, text="")
        if self.preview_running and generation == self.preview_generation and self.camera.index is not None:
            self.preview_after_id = self.root.after(33, lambda: self._update_preview(generation))

    def _calibrate_board(self) -> None:
        if self.last_frame is None:
            self.status_var.set("请先启动摄像头预览，再进行棋盘标定。")
            return
        height, width = self.last_frame.shape[:2]
        if width < 1920 or height < 1080:
            self.calibration_var.set(f"当前画面只有 {width}x{height}，不能用于比赛标定")
            self.status_var.set("棋盘标定要求摄像头实际输出至少 1920x1080。")
            return
        corners = detect_board_corners(self.last_frame)
        if corners is None:
            self.calibration = None
            self.calibration_var.set("自动找棋盘失败：请调整摄像头，使棋盘完整无遮挡")
            self.status_var.set("未找到稳定的四边形棋盘区域。")
            return
        candidate = BoardCalibration(corners)
        try:
            clearance = detect_board_obstructions(self.last_frame, candidate)
        except ValueError as error:
            self.status_var.set(f"棋盘异物检测失败：{error}")
            return
        backup = self._backup_calibration_file(self.calibration_path, "board")
        self.calibration = candidate
        self.board_obstructions = clearance.obstructions
        self.corner_overlay_until = float("inf") if not clearance.clear else time.monotonic() + 10.0
        self.obstruction_last_checked = time.monotonic()
        self.calibration_invalidated = False
        self.calibration.to_json(self.calibration_path)
        self.reference_invalidated = True
        self.recognizer.reset_reference_cache()
        self.calibration_last_checked = time.monotonic()
        self.calibration_drift_failures = 0
        self.board_tracker.reset()
        self.last_vision_result = None
        DiagnosticApp._record_session_event(
            self,
            "board_calibrated",
            clear=clearance.clear,
            obstruction_count=len(clearance.obstructions),
            image_size=[width, height],
        )
        backup_note = f"；旧标定已备份为 {backup.name}" if backup is not None else ""
        if clearance.clear:
            self.calibration_var.set("棋盘已标定且无异物：外角标记将在 10 秒后自动隐藏")
            self.vision_var.set("视觉识别：棋盘已标定且已确认无异物，请保存空棋盘基准图")
            self.status_var.set(f"棋盘标定成功且未发现异物，已保存到 {self.calibration_path}{backup_note}。外角标记将在 10 秒后隐藏。")
        else:
            count = len(clearance.obstructions)
            self.calibration_var.set(f"棋盘已标定，但检测到 {count} 处异物或遗留棋子")
            self.vision_var.set("视觉识别：请清空棋盘异物或遗留棋子后，重新点击“自动找棋盘”")
            self.status_var.set(f"棋盘标定完成，但发现 {count} 处异物或遗留棋子；红框已标出位置。请清空后重新自动找棋盘。")

    def _backup_calibration_file(self, path: Path, label: str) -> Path | None:
        """Preserve an active calibration artifact before replacing it."""
        if not path.exists():
            return None
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        history = self.project_root / "calibration" / "history" / f"before_{label}_{stamp}"
        history.mkdir(parents=True, exist_ok=False)
        destination = history / path.name
        shutil.copy2(path, destination)
        return destination

    def _save_current_dataset_frame(self, *, batch_index: int | None = None) -> Path:
        if self.last_frame is None:
            raise ValueError("没有可保存的画面，请先启动摄像头预览。")
        height, width = self.last_frame.shape[:2]
        if width < 1920 or height < 1080:
            raise ValueError(f"当前画面只有 {width}x{height}，请重新启动1080P摄像头后再采集")
        lighting = {"低光": "low", "正常": "medium", "高光": "high"}[self.dataset_lighting_var.get()]
        scene_id = self.dataset_scene_var.get().strip()
        notes = self.dataset_notes_var.get().strip()
        if batch_index is not None:
            frame_note = f"batch_frame={batch_index:02d}"
            notes = f"{notes}; {frame_note}" if notes else frame_note
        path = self.dataset.save_frame(
            self.camera.correct_color(self.last_frame),
            lighting=lighting,
            edge_scene=self.dataset_edge_var.get(),
            scene_id=scene_id,
            notes=notes,
        )
        self._update_dataset_status()
        return path

    def _begin_dataset_batch(self) -> None:
        self.dataset_batch_active = True
        self.dataset_batch_files = []
        self.dataset_batch_counter = 0
        if self.dataset_auto_id_var.get() or not self.dataset_scene_var.get().strip():
            self.dataset_scene_var.set("batch_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        if hasattr(self, "dataset_batch_list"):
            self.dataset_batch_list.delete(0, "end")
        self.dataset_batch_status_var.set(f"采集批次：{self.dataset_scene_var.get()}；已开始，摆好新布局后保存")
        self.status_var.set(f"采集批次 {self.dataset_scene_var.get()} 已开始；每张会自动记录序号。")

    def _save_dataset_frame_to_batch(self) -> None:
        if not self.dataset_batch_active:
            self._begin_dataset_batch()
        try:
            next_index = self.dataset_batch_counter + 1
            path = self._save_current_dataset_frame(batch_index=next_index)
        except (OSError, ValueError) as error:
            self.status_var.set(f"批次保存失败：{error}")
            return
        self.dataset_batch_counter = next_index
        self.dataset_batch_files.append(path.name)
        if hasattr(self, "dataset_batch_list"):
            self.dataset_batch_list.insert("end", f"第 {next_index:02d} 张：{path.name}")
        recent = "、".join(self.dataset_batch_files[-3:])
        if len(self.dataset_batch_files) > 3:
            recent = "...、" + recent
        self.dataset_batch_status_var.set(f"采集批次：已保存 {len(self.dataset_batch_files)} 张；最近：{recent}")
        self.status_var.set(f"已加入当前批次：{path.name}。可以继续改变布局后保存。")

    def _finish_dataset_batch(self) -> None:
        if not self.dataset_batch_active:
            self.status_var.set("当前没有进行中的采集批次。")
            return
        count = len(self.dataset_batch_files)
        self.dataset_batch_active = False
        self.dataset_batch_status_var.set(f"采集批次 {self.dataset_scene_var.get()}：已结束，共保存 {count} 张；可开始下一批次")
        self.status_var.set(f"本批次已结束，共保存 {count} 张原图；可继续采集其他光照或布局。")

    def _augment_dataset(self) -> None:
        try:
            outputs = self.dataset.augment_all_raw()
            self._update_dataset_status()
            self.status_var.set(f"已生成 {len(outputs)} 张增强图：亮度、颜色、翻转、模糊各一套。")
        except OSError as error:
            self.status_var.set(f"生成增强图失败：{error}")

    def _open_annotation_workbench(self) -> None:
        """Open one review window; it never connects to camera or hardware."""

        if self.annotation_workbench is not None:
            try:
                if self.annotation_workbench.window.winfo_exists():
                    self.annotation_workbench.window.lift()
                    self.annotation_workbench.window.focus_force()
                    return
            except tk.TclError:
                self.annotation_workbench = None
        self.annotation_workbench = AnnotationWorkbench(self.root, self.project_root, self.dataset)
        self.status_var.set("标注工作台已打开：请逐图核验，只有明确勾选确认后才会导出标签。")

    def _update_dataset_status(self) -> None:
        stats = self.dataset.stats()
        self.dataset_var.set(
            f"原图 {stats.raw}/200，增强图 {stats.augmented}/800，标注 {stats.labeled}/20；"
            f"光照 低{stats.low_light}/中{stats.medium_light}/高{stats.high_light}，边缘场景 {stats.edge_scenes}"
        )

    def _foundation_key(self) -> str:
        try:
            return self.foundation_target_labels[self.foundation_target_var.get()]
        except KeyError as error:
            raise ValueError("请选择一个有效黑点") from error

    @staticmethod
    def _point_from_vars(variables: list[tk.StringVar], name: str) -> tuple[float, float, float]:
        try:
            point = tuple(float(variable.get()) for variable in variables)
        except ValueError as error:
            raise ValueError(f"{name}坐标必须全部为数字") from error
        if len(point) != 3:
            raise ValueError(f"{name}必须包含 X/Y/Z")
        return point  # type: ignore[return-value]

    def _apply_foundation_base_inputs(self) -> None:
        pickup = self._point_from_vars(self.foundation_pickup_vars, "取子点")
        standby = self._point_from_vars(self.foundation_standby_vars, "待命点")
        try:
            safe_z = float(self.foundation_safe_z_var.get())
        except ValueError as error:
            raise ValueError("安全Z必须是数字") from error
        self.foundation_calibration = FoundationHardwareCalibration(
            pickup_point=pickup,
            release_point=self.foundation_calibration.release_point,
            standby_point=standby,
            safe_z=safe_z,
            turntable_receive_duty=self.foundation_calibration.turntable_receive_duty,
            turntable_pickup_duty=self.foundation_calibration.turntable_pickup_duty,
            targets=dict(self.foundation_calibration.targets),
        )

    def _save_foundation_base(self) -> None:
        try:
            self._apply_foundation_base_inputs()
            self.foundation_calibration.to_json(self.foundation_calibration_path)
        except (OSError, ValueError) as error:
            self.status_var.set(f"保存基础坐标失败：{error}")
            return
        self._refresh_foundation_table()
        pickup = self.foundation_calibration.pickup_point
        self.status_var.set(
            f"基础坐标已保存：取子点 X={pickup[0]:g}、Y={pickup[1]:g}、Z={pickup[2]:g}。"
        )

    def _save_foundation_target(self) -> None:
        try:
            self._apply_foundation_base_inputs()
            key = self._foundation_key()
            point = self._point_from_vars(self.foundation_target_xyz_vars, target_label(key))
            self.foundation_calibration.set_target(key, point)
            self.foundation_calibration.to_json(self.foundation_calibration_path)
        except (OSError, ValueError) as error:
            self.status_var.set(f"保存黑点坐标失败：{error}")
            return
        self._refresh_foundation_table()
        self.status_var.set(f"已保存 {target_label(key)} 的机械臂坐标。")

    def _load_selected_foundation_target(self, _event=None) -> None:
        try:
            key = self._foundation_key()
        except ValueError:
            return
        point = self.foundation_calibration.targets.get(key)
        values = ("", "", "") if point is None else tuple(f"{value:g}" for value in point)
        for variable, value in zip(self.foundation_target_xyz_vars, values):
            variable.set(value)

    def _select_foundation_tree_target(self, _event=None) -> None:
        selected = self.foundation_tree.selection()
        if not selected:
            return
        key = selected[0]
        try:
            self.foundation_target_var.set(target_label(key))
        except KeyError:
            return
        self._load_selected_foundation_target()

    def _refresh_foundation_table(self) -> None:
        if not hasattr(self, "foundation_tree"):
            return
        for item in self.foundation_tree.get_children():
            self.foundation_tree.delete(item)
        for key, board_move, label in FOUNDATION_TARGETS:
            point = self.foundation_calibration.targets.get(key)
            coordinates = ("", "", "") if point is None else tuple(f"{value:.1f}" for value in point)
            self.foundation_tree.insert(
                "",
                "end",
                iid=key,
                text=label,
                values=(str(board_move), *coordinates, "已标定" if point is not None else "待标定"),
            )
        if self.foundation_calibration.complete:
            self.foundation_summary_var.set("五个黑圆点均已保存，可按需选择其中一个点进行微调。")
        else:
            self.foundation_summary_var.set(
                "尚缺：" + "、".join(self.foundation_calibration.missing_labels) + "。先逐点空载到达并保存实测坐标。"
            )
        self._load_selected_foundation_target()

    def _measure_foundation_visual_offsets(self) -> None:
        """Report five-point placement errors without changing any coordinates."""
        if self.last_frame is None:
            self.status_var.set("无法测量：请先启动摄像头预览。")
            return
        calibration = self._current_calibration()
        if calibration is None:
            self.status_var.set("无法测量：请先完成棋盘标定。")
            return
        if not self.recognizer.has_reference:
            self.status_var.set("无法测量：未找到空棋盘基准图。")
            return
        expected = tuple(board_move for _, board_move, _ in FOUNDATION_TARGETS)
        try:
            measurements = self.recognizer.measure_black_stone_centers(
                self.last_frame,
                calibration,
                expected,
            )
        except (FileNotFoundError, OSError, ValueError) as error:
            self.foundation_measurement_var.set(f"视觉五点测量失败：{error}")
            self.status_var.set(f"视觉五点测量失败：{error}")
            return

        by_move = {item.move: item for item in measurements}
        spacing = (calibration.output_size - 1) / (calibration.board_size - 1)
        lines = ["视觉五点误差（+X 向右，+Y 向下；候选值尚未保存）："]
        for key, move, label in FOUNDATION_TARGETS:
            measurement = by_move[move]
            pixel_x, pixel_y = measurement.offset_pixel
            offset_cells = (pixel_x / spacing, pixel_y / spacing)
            candidate = self._foundation_visual_candidate(key, move, offset_cells)
            if candidate is None:
                candidate_note = "（缺少四角机械臂标定，未给出 mm 建议）"
            else:
                candidate_note = f" → 建议 X={candidate[0]:.1f}, Y={candidate[1]:.1f}"
            lines.append(
                f"{label} {move}：图像 Δ=({pixel_x:+.1f}, {pixel_y:+.1f})px"
                f" / ({offset_cells[0]:+.2f}, {offset_cells[1]:+.2f})格{candidate_note}"
            )
        self.foundation_measurement_var.set("\n".join(lines))
        self.status_var.set("视觉五点测量完成：已生成候选修正值，尚未保存或发送任何机械臂命令。")

    def _foundation_visual_candidate(
        self,
        key: str,
        move: tuple[int, int],
        offset_cells: tuple[float, float],
    ) -> tuple[float, float, float] | None:
        """Convert a camera-grid error to a candidate correction in arm mm."""
        arm = self.advanced_arm_calibration
        point = self.foundation_calibration.targets.get(key)
        if arm is None or point is None:
            return None
        column, row = move
        divisor = float(arm.board_size - 1)
        u, v = column / divisor, row / divisor
        # Local derivatives of the calibrated bilinear board-to-arm mapping,
        # expressed as mm per one camera grid cell.
        per_column = tuple(
            ((1 - v) * (arm.point_12_0[index] - arm.point_0_0[index])
             + v * (arm.point_12_12[index] - arm.point_0_12[index]))
            / divisor
            for index in range(3)
        )
        per_row = tuple(
            ((1 - u) * (arm.point_0_12[index] - arm.point_0_0[index])
             + u * (arm.point_12_12[index] - arm.point_12_0[index]))
            / divisor
            for index in range(3)
        )
        correction = tuple(
            -(offset_cells[0] * per_column[index] + offset_cells[1] * per_row[index])
            for index in range(3)
        )
        return (
            point[0] + correction[0],
            point[1] + correction[1],
            point[2],
        )

    def _selected_foundation_point(self) -> tuple[str, tuple[float, float, float]]:
        self._apply_foundation_base_inputs()
        key = self._foundation_key()
        try:
            point = self.foundation_calibration.targets[key]
        except KeyError as error:
            raise ValueError(f"{target_label(key)}尚未保存坐标") from error
        return key, point

    def _move_foundation_target(self) -> None:
        try:
            key, point = self._selected_foundation_point()
        except ValueError as error:
            self.status_var.set(str(error))
            return
        self._hardware_reply(
            f"空载到达{target_label(key)}",
            lambda: self.hardware.move_via_approach(point, approach_z=self.foundation_calibration.safe_z),
        )

    def _place_foundation_target(self) -> None:
        try:
            key, point = self._selected_foundation_point()
        except ValueError as error:
            self.status_var.set(str(error))
            return
        calibration = self.foundation_calibration
        self._hardware_reply(
            f"取放到{target_label(key)}",
            lambda: self.hardware.pick_and_place(
                calibration.pickup_point,
                point,
                calibration.standby_point,
                safe_z=calibration.safe_z,
                turntable_receive_duty=calibration.turntable_receive_duty,
                turntable_pickup_duty=calibration.turntable_pickup_duty,
                cancel_requested=self._hardware_cancel_requested,
            ),
            on_success=lambda _reply: self._set_turntable_state("0°接子", calibration.turntable_receive_duty),
        )

    def _debug_arm_to_center_target(self) -> None:
        """Pick one piece and place it on the saved centre foundation point."""
        calibration = self.foundation_calibration
        try:
            center_point = calibration.targets["center"]
        except KeyError:
            self.status_var.set("中心黑点尚未保存坐标，无法执行机械臂调试。")
            return
        self._hardware_reply(
            "机械臂调试",
            lambda: self.hardware.pick_and_place(
                calibration.pickup_point,
                center_point,
                calibration.standby_point,
                safe_z=calibration.safe_z,
                turntable_receive_duty=calibration.turntable_receive_duty,
                turntable_pickup_duty=calibration.turntable_pickup_duty,
                cancel_requested=self._hardware_cancel_requested,
            ),
            on_success=lambda _reply: self._finish_center_arm_debug(calibration.turntable_receive_duty),
        )

    def _finish_center_arm_debug(self, receive_duty: int) -> None:
        self._set_turntable_state("0°接子", receive_duty)
        self.status_var.set("机械臂调试完成：棋子已放到五黑点中心，并已恢复待命。")

    def _place_all_foundation_targets(self) -> None:
        try:
            self._apply_foundation_base_inputs()
            targets = self.foundation_calibration.ordered_targets()
        except ValueError as error:
            self.status_var.set(str(error))
            return
        calibration = self.foundation_calibration

        def run_all() -> HardwareReply:
            started = time.monotonic()
            for index, (key, point) in enumerate(targets, start=1):
                self.root.after(
                    0,
                    lambda index=index, key=key: self.status_var.set(
                        f"连续五黑点 {index}/5：正在执行 {target_label(key)}"
                    ),
                )
                self.hardware.pick_and_place(
                    calibration.pickup_point,
                    point,
                    calibration.standby_point,
                    safe_z=calibration.safe_z,
                    turntable_receive_duty=calibration.turntable_receive_duty,
                    turntable_pickup_duty=calibration.turntable_pickup_duty,
                    cancel_requested=self._hardware_cancel_requested,
                )
            elapsed = time.monotonic() - started
            return HardwareReply("FOUNDATION_FIVE_POINTS", f"FOUNDATION_FIVE_OK:{elapsed:.2f}s")

        self._hardware_reply(
            "连续五黑点取放",
            run_all,
            on_success=lambda _reply: self._set_turntable_state("0°接子", calibration.turntable_receive_duty),
        )

    def _current_calibration(self) -> BoardCalibration | None:
        if self.calibration_invalidated:
            return None
        if self.calibration is not None:
            return self.calibration
        if not self.calibration_path.exists():
            return None
        try:
            self.calibration = BoardCalibration.from_json(self.calibration_path)
        except (OSError, ValueError, KeyError):
            return None
        return self.calibration

    def _monitor_calibration_drift(self, *, force: bool = False) -> bool:
        """Stop recognition immediately when the live board leaves calibration."""
        calibration = self.calibration
        if calibration is None or self.calibration_invalidated or self.last_frame is None:
            return not self.calibration_invalidated
        now = time.monotonic()
        if not force and now - self.calibration_last_checked < self.calibration_check_interval:
            # A failed check is an uncertain state: do not let a hand or an
            # occluding object be interpreted as a new board position while
            # waiting for the next geometry check.
            return self.calibration_drift_failures == 0
        self.calibration_last_checked = now
        result = check_calibration_alignment(self.last_frame, calibration)
        if result.aligned:
            self.calibration_drift_failures = 0
            return True
        self.calibration_drift_failures += 1
        if self.calibration_drift_failures < CALIBRATION_DRIFT_FAILURE_LIMIT:
            self.status_var.set(
                "棋盘角点暂时不稳定，已暂停识别并等待复核 "
                f"({self.calibration_drift_failures}/{CALIBRATION_DRIFT_FAILURE_LIMIT})；"
                "请将手移出棋盘。"
            )
            return False
        self._invalidate_calibration(f"{result.reason}。已停止自动识别，需重新点击“自动找棋盘”并保存空盘基准")
        return False

    def _runtime_board_guards_enabled(self) -> bool:
        """Keep empty-board and drift guards out of a live dry-run game.

        Dry run has no hardware I/O.  Its stable-three-frame recognizer is
        the appropriate gate while a person is deliberately reaching into
        the board.  A camera/support move still requires ending the game and
        calibrating again before any hardware workflow is used.
        """
        return not self.dry_run_enabled

    def _invalidate_calibration(self, reason: str) -> None:
        self.calibration_invalidated = True
        self.calibration = None
        self.reference_invalidated = True
        self.recognizer.reset_reference_cache()
        self.board_tracker.reset()
        self.last_vision_result = None
        self.board_obstructions = ()
        self.corner_overlay_until = 0.0
        self.obstruction_last_checked = 0.0
        self.calibration_var.set("标定已失效：请重新自动找棋盘")
        self.vision_var.set("视觉识别已停止：相机/棋盘几何发生变化")
        self.status_var.set(f"安全停止：{reason}")

    def _capture_empty_board(self) -> None:
        if self.last_frame is None:
            self.status_var.set("请先启动摄像头预览，再保存空棋盘基准图。")
            return
        if self.hardware_busy:
            self.status_var.set("硬件动作尚未结束，请等待转盘或机械臂回执后再保存空盘基准。")
            return
        receive_duty = self.foundation_calibration.turntable_receive_duty
        must_return_turntable = self.hardware.connected and self.turntable_duty != receive_duty
        if must_return_turntable:
            self.status_var.set("保存空盘前正在将转盘回料仓位，请等待回执……")
            self._hardware_reply(
                "保存空盘前转盘回料仓位",
                lambda: self.hardware.turntable(receive_duty),
                on_success=lambda _reply: DiagnosticApp._finish_empty_board_turntable_return(self, receive_duty),
            )
            return
        if self.turntable_duty is not None and self.turntable_duty != receive_duty:
            self.status_var.set("转盘未在料仓位且调试串口未连接；请先连接 DEBUG 后再保存空盘基准。")
            return
        self._save_empty_board_reference()

    def _finish_empty_board_turntable_return(self, receive_duty: int) -> None:
        DiagnosticApp._set_turntable_state(self, "料仓位", receive_duty)
        DiagnosticApp._record_session_event(self, "turntable_returned_for_empty_board", duty=receive_duty)
        self._save_empty_board_reference()

    def _save_empty_board_reference(self) -> None:
        """Persist the visual baseline only after the tray is safe or unavailable."""
        calibration = self._current_calibration()
        if calibration is None:
            self.status_var.set("请先完成四角棋盘标定，再保存空棋盘基准图。")
            return
        try:
            self._backup_calibration_file(self.recognizer.reference_path, "empty_board")
            path = self.recognizer.capture_empty_board(self.last_frame, calibration)
        except (OSError, ValueError) as error:
            self.status_var.set(f"空棋盘基准图保存失败：{error}")
            return
        self.board_tracker.reset()
        self.reference_invalidated = False
        self.last_vision_result = None
        self.vision_var.set("视觉识别：空棋盘基准已保存。落子后可点击识别，或开启自动识别。")
        self.status_var.set(f"空棋盘基准图已保存：{path.name}。{self._hardware_connection_text()}")

    def _recognize_board(self, automatic: bool = False) -> None:
        if self.last_frame is None:
            if not automatic:
                self.status_var.set("请先启动摄像头预览，再识别棋局。")
            return
        calibration = self._current_calibration()
        if calibration is None:
            if not automatic:
                self.status_var.set("请先完成四角棋盘标定，再识别棋局。")
            return
        if self._runtime_board_guards_enabled() and not self._monitor_calibration_drift(force=not automatic):
            return
        if self.reference_invalidated:
            if not automatic:
                self.status_var.set("请在当前标定下重新保存空棋盘基准图。")
            return
        if not self.recognizer.has_reference:
            if not automatic:
                self.status_var.set("请清空棋盘后点击“保存空棋盘基准”。")
            return
        if self.dry_run_enabled:
            if self.game_finished:
                # Keep showing the final board, but never turn post-game
                # movement into another plan or an obstruction warning.
                return
            if self.game_action_in_progress:
                # A moving arm is not a chess position.  Keep the last
                # camera-confirmed board unchanged until the full guarded
                # pick-and-place sequence has returned to standby.
                self.board_tracker.reset()
                self.vision_var.set("机械臂动作进行中：棋局识别已冻结，等待回待命位后再进行视觉复核。")
                return
            try:
                occlusion = detect_gameplay_occlusion(self.last_frame, calibration)
            except ValueError as error:
                self.status_var.set(f"棋盘遮挡检查失败：{error}")
                return
            if occlusion is not None:
                self.board_tracker.reset()
                self.vision_var.set(
                    "视觉识别临时冻结：机械臂、手或其他大物体正在遮挡棋盘；"
                    "移出棋盘后将重新等待三帧稳定画面。"
                )
                return
        try:
            result = self.recognizer.detect(self.last_frame, calibration)
        except (FileNotFoundError, OSError, ValueError) as error:
            self.status_var.set(f"棋局识别失败：{error}")
            return

        self.last_vision_result = result
        changed = self.board_tracker.update(result)
        detected = len(result.detections)
        stable_count = self.board_tracker.consecutive_frames
        if stable_count < self.board_tracker.required_frames:
            self.vision_var.set(
                f"视觉识别：当前看到 {detected} 颗棋子，正在确认 {stable_count}/{self.board_tracker.required_frames} 帧"
            )
            return

        self.vision_var.set(f"视觉识别：稳定棋局 {detected} 颗棋子，已连续确认 {stable_count} 帧")
        if changed:
            self.status_var.set(f"视觉棋局已更新：当前稳定识别到 {detected} 颗棋子。")
            if self.dry_run_enabled:
                update = self.game_controller.observe(self.board_tracker.stable_stones)
                self._apply_gameflow_update(update)
                self._check_game_visual_confirmation(update)

    def _start_dry_run(self) -> None:
        """Enable a visual game loop; hardware remains opt-in per move."""
        if self.game_action_in_progress:
            self.status_var.set("当前黑棋动作尚未结束，不能重置对局。")
            return
        self.dry_run_enabled = True
        self.dry_run_last_recognition = 0.0
        self.game_controller.reset()
        self.board_tracker.reset()
        self.board_obstructions = ()
        self.corner_overlay_until = 0.0
        self.game_suggested_move = None
        self.game_motion_plan = None
        self.game_waiting_visual_confirmation = False
        self.game_finished = False
        self.game_pending_move = None
        self._cancel_game_timers()
        DiagnosticApp._set_match_result_banner(self, "对局进行中：等待稳定棋局", finished=False)
        self.game_phase_var.set("干运行已启动：等待三帧一致的摄像头棋局。")
        self.game_message_var.set("等待摄像头连续三帧确认当前棋局。")
        self.game_plan_var.set("等待稳定棋局后由 AI 生成黑棋建议。")
        self._draw_game_board(GomokuBoard())
        self.status_var.set("进阶赛对局已启动：默认只读；受监督执行开关关闭时不会驱动硬件。")
        DiagnosticApp._record_session_event(self, "advanced_match_started", automatic_authorized=False)

    def _set_match_result_banner(self, text: str, *, finished: bool) -> None:
        """Keep the operator-facing result adjacent to the virtual board."""
        result_var = getattr(self, "match_result_var", None)
        if result_var is not None:
            result_var.set(text)
        label = getattr(self, "match_result_label", None)
        if label is not None:
            label.configure(style="MatchResult.TLabel" if finished else "MatchPending.TLabel")

    def _announce_match_result(self, text: str) -> None:
        voice_enabled = getattr(self, "match_voice_enabled_var", None)
        if voice_enabled is None or not voice_enabled.get():
            return
        threading.Thread(target=announce_voice, args=(text,), name="match-result-voice", daemon=True).start()

    def _stop_dry_run(self) -> None:
        if self.game_action_in_progress:
            self._emergency_stop_game_action()
        self.dry_run_enabled = False
        self.game_hardware_enabled_var.set(False)
        self.game_auto_execution_var.set(False)
        self.game_waiting_visual_confirmation = False
        self._cancel_game_timers()
        DiagnosticApp._set_match_result_banner(self, "对局已停止", finished=False)
        self.game_phase_var.set("干运行已停止")
        self.game_message_var.set("视觉预览可继续使用；不会再生成新的 AI 动作计划。")
        self.status_var.set("进阶赛干运行已停止；未发送任何硬件命令。")
        DiagnosticApp._record_session_event(self, "advanced_match_stopped")

    def _prepare_next_match(self) -> None:
        """Disarm the completed/current game before the operator clears real pieces.

        This intentionally clears only the on-screen state.  A physical board
        cannot be cleared by software, so the old empty-board reference is
        invalidated and the next game remains blocked until the operator has
        removed every physical piece and saved a fresh empty-board baseline.
        """
        if self.game_action_in_progress or self.hardware_busy:
            self.status_var.set("机械臂动作尚未结束，不能准备下一局。请等待回待命位和动作回执。")
            return
        self.dry_run_enabled = False
        self.game_hardware_enabled_var.set(False)
        self.game_auto_execution_var.set(False)
        self.game_controller.reset()
        self.board_tracker.reset()
        self.game_suggested_move = None
        self.game_motion_plan = None
        self.game_waiting_visual_confirmation = False
        self.game_finished = False
        self.game_pending_move = None
        self._cancel_game_timers()
        self.reference_invalidated = True
        self.recognizer.reset_reference_cache()
        self._draw_game_board(GomokuBoard())
        DiagnosticApp._set_match_result_banner(self, "请清空实体棋盘后开始下一局", finished=False)
        self.game_phase_var.set("准备下一局：请先清空实体棋盘")
        self.game_message_var.set("UI 已清空且自动执行已撤销；请手动移走所有棋子。")
        self.game_plan_var.set("清空实体棋盘后，点击“保存空盘基准”，再开始对局。")
        self.status_var.set("已准备下一局：未发送硬件命令。请清空实体棋盘并重新保存空盘基准。")
        DiagnosticApp._record_session_event(self, "advanced_match_reset_for_next_game")

    def _apply_gameflow_update(self, update: GameFlowUpdate) -> None:
        """Render a stable camera state and its guarded action plan."""
        if update.phase == GamePhase.FINISHED:
            DiagnosticApp._finish_completed_game(self, update)
            return
        self.game_phase_var.set(f"当前阶段：{update.phase.value}")
        self.game_message_var.set(update.message)
        self.game_suggested_move = update.plan.move if update.plan is not None else None
        self.game_motion_plan = update.plan
        if update.plan is None:
            self.game_plan_var.set("当前没有新的黑棋动作计划。")
        else:
            coordinate = board_to_competition(update.plan.move)
            details = "\n".join(update.plan.display_lines())
            self.game_plan_var.set(
                f"AI 建议黑棋：竞赛坐标 {coordinate}（内部 {update.plan.move}）。\n{details}"
            )
        self._draw_game_board(update.board)
        DiagnosticApp._record_session_event(
            self,
            "board_state_updated",
            phase=update.phase.value,
            stone_count=len(update.board.moves),
            suggested_move=board_to_competition(update.plan.move) if update.plan is not None else None,
        )
        if self._game_auto_execution_enabled():
            suffix = "比赛自动执行已授权；本局将仅在满足全部安全门禁后自动执行黑棋。"
        elif self.game_hardware_enabled_var.get():
            suffix = "受监督执行已启用，仍需点击“执行本次黑棋”。"
        else:
            suffix = "未发送硬件命令。"
        self.status_var.set(f"对局更新：{update.message}（{suffix}）")
        if update.phase == GamePhase.ROBOT_BLACK_TURN and update.plan is not None and self._game_auto_execution_enabled():
            self._execute_game_plan(automatic=True)

    def _finish_completed_game(self, update: GameFlowUpdate) -> None:
        """End a legally recognised game without treating it as a fault."""
        self._cancel_game_timers()
        self.game_action_in_progress = False
        self.game_waiting_visual_confirmation = False
        self.game_pending_move = None
        self.game_finished = True
        self.game_hardware_enabled_var.set(False)
        self.game_auto_execution_var.set(False)
        self.game_suggested_move = None
        self.game_motion_plan = None
        self.game_phase_var.set("当前阶段：对局结束")
        self.game_message_var.set(f"{update.message} 本局已正常结束，不会再执行机械臂动作。")
        self.game_plan_var.set("对局结果已确认。清空棋盘、重新保存空棋盘基准后，才可开始下一局。")
        if "机器人获胜" in update.message:
            result = "机器人获胜！"
        elif "人类获胜" in update.message:
            result = "人类获胜！"
        else:
            result = "本局平局"
        DiagnosticApp._set_match_result_banner(self, result, finished=True)
        DiagnosticApp._announce_match_result(self, result)
        self._draw_game_board(update.board)
        self.status_var.set(f"进阶赛对局正常结束：{update.message}")
        DiagnosticApp._record_session_event(
            self,
            "advanced_match_finished",
            message=update.message,
            stone_count=len(update.board.moves),
        )

    def _on_game_execution_toggle(self) -> None:
        if not self.game_hardware_enabled_var.get():
            if self.game_action_in_progress:
                self._emergency_stop_game_action()
            self.status_var.set("受监督执行已关闭；不会再启动新的机械臂动作。")
            return
        self.status_var.set("受监督执行已启用：仍需视觉生成黑棋计划，并手动点击“执行本次黑棋”。")

    def _game_auto_execution_enabled(self) -> bool:
        """Return the explicit match-level automation authorization, if present."""
        variable = getattr(self, "game_auto_execution_var", None)
        return bool(variable is not None and variable.get())

    def _on_game_auto_execution_toggle(self) -> None:
        """Arm or revoke only the match-level automatic execution permission."""
        if not self._game_auto_execution_enabled():
            if self.game_action_in_progress:
                self._emergency_stop_game_action()
            self.status_var.set("比赛自动执行已撤销；不会启动新的黑棋动作。")
            DiagnosticApp._record_session_event(self, "automatic_execution_revoked")
            return
        self.status_var.set(
            "比赛自动执行已一次授权：后续仅对三帧稳定、标定有效且已连接 DEBUG 的机器人黑棋回合自动执行。"
        )
        DiagnosticApp._record_session_event(self, "automatic_execution_authorized")
        if self.dry_run_enabled and self.game_motion_plan is not None:
            self._execute_game_plan(automatic=True)

    def _execute_game_plan(self, *, automatic: bool = False) -> None:
        """Send one camera-confirmed plan in supervised or explicitly armed match mode."""
        if automatic:
            if not DiagnosticApp._game_auto_execution_enabled(self):
                return
        elif not self.game_hardware_enabled_var.get():
            self.status_var.set("未执行：请先勾选“受监督执行（默认关闭）”。")
            return
        if not self.dry_run_enabled:
            self.status_var.set("未执行：请先开始进阶赛对局并等待稳定视觉棋局。")
            return
        if getattr(self, "game_finished", False):
            self.status_var.set("未执行：本局已结束。请清空棋盘、重新保存空棋盘基准后开始下一局。")
            return
        if self.game_action_in_progress or self.hardware_busy:
            self.status_var.set("未执行：已有硬件动作正在执行。")
            return
        plan = self.game_motion_plan
        calibration = self.advanced_arm_calibration
        if plan is None or self.game_suggested_move is None:
            self.status_var.set("未执行：当前没有经视觉确认的黑棋动作计划。")
            return
        if not plan.calibrated or calibration is None:
            self.status_var.set("未执行：缺少已验证的四角机械臂标定。")
            return
        if self.calibration_invalidated or self.reference_invalidated or self.last_vision_result is None:
            self.status_var.set("未执行：视觉标定或空棋盘基准无效，请重新确认后再开始对局。")
            return
        if self.board_tracker.consecutive_frames < self.board_tracker.required_frames:
            self.status_var.set("未执行：当前棋局尚未达到三帧一致。")
            return
        if not self.hardware.connected:
            self.status_var.set("未执行：请先连接机械臂调试端口并确认处于 DEBUG 模式。")
            return
        target = calibration.board_to_arm(plan.move)
        foundation = self.foundation_calibration
        self.game_action_in_progress = True
        self.game_waiting_visual_confirmation = False
        self.game_pending_move = plan.move
        self._schedule_game_action_timeout(plan.move)
        coordinate = board_to_competition(plan.move)
        self.game_phase_var.set("当前阶段：执行机器人黑棋")
        self.game_message_var.set(
            f"正在执行黑棋 {coordinate}。请保持棋盘周围无人、支架固定；可随时点击“紧急停止本手”。"
        )
        DiagnosticApp._record_session_event(
            self,
            "robot_move_started",
            mode="automatic" if automatic else "supervised",
            move=coordinate,
            target=[round(value, 3) for value in target],
        )
        self._hardware_reply(
            f"进阶赛{'自动' if automatic else '受监督'}黑棋 {coordinate} 取放",
            lambda: self.hardware.pick_and_place(
                calibration.pickup_point,
                target,
                calibration.standby_point,
                safe_z=calibration.safe_z,
                turntable_receive_duty=foundation.turntable_receive_duty,
                turntable_pickup_duty=foundation.turntable_pickup_duty,
                cancel_requested=self._hardware_cancel_requested,
            ),
            on_success=lambda _reply: self._begin_game_visual_confirmation(plan.move),
            on_failure=self._stop_game_after_motion_failure,
        )

    def _begin_game_visual_confirmation(self, move: tuple[int, int]) -> None:
        self._cancel_game_action_timeout()
        self.game_action_in_progress = False
        # pick_and_place ends by returning the turntable to its receive/storage
        # position, so later empty-board capture does not rotate it again.
        self.turntable_duty = self.foundation_calibration.turntable_receive_duty
        self.game_waiting_visual_confirmation = True
        self.game_pending_move = move
        self.board_tracker.reset()
        self.game_phase_var.set("当前阶段：等待视觉确认机器人黑棋")
        self.game_message_var.set("机械臂已回待命位。请勿触碰棋盘，等待摄像头连续三帧确认黑棋在目标点。")
        self.game_plan_var.set("本手已执行，等待视觉确认；未确认前不会进入人类白棋回合。")
        self._schedule_game_confirmation_timeout(move)
        self.status_var.set("机器人动作回执已收到：正在进行落子视觉复核。")
        DiagnosticApp._record_session_event(self, "robot_move_waiting_visual_confirmation", move=board_to_competition(move))

    def _check_game_visual_confirmation(self, update: GameFlowUpdate) -> None:
        if not self.game_waiting_visual_confirmation or self.game_pending_move is None:
            return
        pending = self.game_pending_move
        confirmed_stone = dict(update.board.moves).get(pending)
        if confirmed_stone != Stone.BLACK:
            self.game_phase_var.set("当前阶段：等待视觉确认机器人黑棋")
            self.game_message_var.set("尚未在目标点确认黑棋，继续等待稳定视觉结果。")
            return
        self._cancel_game_confirmation_timeout()
        self.game_waiting_visual_confirmation = False
        self.game_pending_move = None
        self.status_var.set("机器人黑棋已由视觉确认：现在可等待人类白棋，或继续受监督执行下一手。")
        DiagnosticApp._record_session_event(self, "robot_move_visually_confirmed", move=board_to_competition(pending))

    def _schedule_game_action_timeout(self, move: tuple[int, int]) -> None:
        self._cancel_game_action_timeout()
        self.game_action_timeout_after_id = self.root.after(
            GAME_ACTION_TIMEOUT_MS,
            lambda: self._timeout_game_action(move),
        )

    def _timeout_game_action(self, move: tuple[int, int]) -> None:
        self.game_action_timeout_after_id = None
        if not self.game_action_in_progress or self.game_pending_move != move:
            return
        self._request_hardware_stop()
        self.status_var.set("本手动作超时：已请求停止，等待当前底层指令结束后不会继续下一步。")

    def _schedule_game_confirmation_timeout(self, move: tuple[int, int]) -> None:
        self._cancel_game_confirmation_timeout()
        self.game_confirmation_timeout_after_id = self.root.after(
            GAME_VISUAL_CONFIRMATION_TIMEOUT_MS,
            lambda: self._timeout_game_confirmation(move),
        )

    def _timeout_game_confirmation(self, move: tuple[int, int]) -> None:
        self.game_confirmation_timeout_after_id = None
        if self.game_waiting_visual_confirmation and self.game_pending_move == move:
            self._stop_game_after_motion_failure("落子视觉确认超时")

    def _emergency_stop_game_action(self) -> None:
        if not self.game_action_in_progress:
            self.status_var.set("当前没有正在执行的进阶赛黑棋动作。")
            return
        self._request_hardware_stop()
        self.game_hardware_enabled_var.set(False)
        self.game_auto_execution_var.set(False)
        self.status_var.set("已请求紧急停止：当前底层指令结束后将取消后续动作并关闭气泵。")
        DiagnosticApp._record_session_event(self, "advanced_match_emergency_stop_requested")

    def _stop_game_after_motion_failure(self, error) -> None:
        self._cancel_game_timers()
        self.game_action_in_progress = False
        self.game_waiting_visual_confirmation = False
        self.game_pending_move = None
        self.game_finished = False
        self.game_hardware_enabled_var.set(False)
        self.game_auto_execution_var.set(False)
        self.dry_run_enabled = False
        self.game_phase_var.set("当前阶段：安全停止")
        self.game_message_var.set(f"本局已停止：{error}。请检查现场和棋盘，再重新开始。")
        self.game_plan_var.set("安全停止后不再生成或执行动作计划。")
        self.status_var.set(f"进阶赛动作失败/未确认，已安全停止：{error}")
        DiagnosticApp._record_session_event(self, "advanced_match_safety_stop", reason=str(error))

    def _cancel_game_action_timeout(self) -> None:
        if self.game_action_timeout_after_id is not None:
            self.root.after_cancel(self.game_action_timeout_after_id)
            self.game_action_timeout_after_id = None

    def _cancel_game_confirmation_timeout(self) -> None:
        if self.game_confirmation_timeout_after_id is not None:
            self.root.after_cancel(self.game_confirmation_timeout_after_id)
            self.game_confirmation_timeout_after_id = None

    def _cancel_game_timers(self) -> None:
        self._cancel_game_action_timeout()
        self._cancel_game_confirmation_timeout()

    def _draw_game_board(self, board: GomokuBoard) -> None:
        """Draw the camera-confirmed board and the unexecuted AI suggestion."""
        canvas = getattr(self, "game_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        size = board.size
        margin = 28
        span = 274
        spacing = span / (size - 1)
        for index in range(size):
            point = margin + index * spacing
            canvas.create_line(margin, point, margin + span, point, fill="#49341d")
            canvas.create_line(point, margin, point, margin + span, fill="#49341d")
            if index in (0, 6, 12):
                canvas.create_text(point, margin - 14, text=str(size - index), fill="#49341d", font=("Arial", 8))
                canvas.create_text(margin - 14, point, text=str(index + 1), fill="#49341d", font=("Arial", 8))
        for move, stone in board.moves:
            column, row = move
            x = margin + column * spacing
            y = margin + row * spacing
            fill = "#181818" if stone == Stone.BLACK else "#f7f4e8"
            canvas.create_oval(x - 8, y - 8, x + 8, y + 8, fill=fill, outline="#333333")
        if self.game_suggested_move is not None:
            column, row = self.game_suggested_move
            x = margin + column * spacing
            y = margin + row * spacing
            canvas.create_oval(x - 12, y - 12, x + 12, y + 12, outline="#d34522", width=3)

    def _connect_hardware(self) -> None:
        selected_port = self.hardware_port_var.get().strip()
        if not selected_port:
            self.status_var.set("请先点击“刷新串口”并选择机械臂调试端口。")
            return
        self.hardware.port = selected_port
        try:
            reply = self.hardware.connect()
        except HardwareDebugError as error:
            self._set_arm_status(f"连接失败（{error}）")
            self.status_var.set(str(error))
            DiagnosticApp._record_session_event(self, "hardware_debug_connection_failed", reason=str(error))
            return
        self._set_arm_status(f"调试模式已连接 {self.hardware.port}")
        self.status_var.set(f"已收到 {reply.response}。现在只允许调试移动/转盘命令。")
        self.hardware_connect_button.configure(text="已连接")
        DiagnosticApp._record_session_event(self, "hardware_debug_connected", port=self.hardware.port)

    def _hardware_connection_text(self) -> str:
        return "硬件调试串口已连接。" if self.hardware.connected else "硬件调试串口未连接。"

    def _disconnect_hardware(self) -> None:
        if self.hardware_busy:
            self.status_var.set("当前硬件动作仍在执行，请等待完成回执后再退出调试。")
            return
        self.hardware.disconnect()
        self._set_arm_status(f"已断开 {self.hardware.port}")
        self.status_var.set("硬件调试串口已断开；自动对局仍不会发送动作。")
        self.hardware_connect_button.configure(text="连接并进入调试")
        DiagnosticApp._record_session_event(self, "hardware_debug_disconnected", port=self.hardware.port)

    def _hardware_reply(self, action: str, callback, on_success=None, on_failure=None) -> None:
        if self.hardware_busy:
            self.status_var.set("已有硬件动作正在执行，请等待完成回执。")
            return
        if not self.hardware.connected:
            message = f"{action}未执行：请先从刷新到的串口列表中选择端口并进入调试模式。"
            self.status_var.set(message)
            if action.startswith("取子机构舵机调试台"):
                self.servo_var.set("取子机构舵机调试台：未执行（未连接调试端口）")
            return
        self._clear_hardware_cancel_request()
        self.hardware_busy = True
        self.status_var.set(f"{action}执行中，请等待机械臂回执……")
        self.root.update_idletasks()

        def worker() -> None:
            reply = None
            error = None
            try:
                reply = callback()
            except Exception as caught:
                error = caught
            if not self.closing:
                self.root.after(0, lambda: self._finish_hardware_reply(action, reply, error, on_success, on_failure))

        threading.Thread(target=worker, name="robot-hardware-action", daemon=True).start()

    def _finish_hardware_reply(self, action: str, reply, error, on_success, on_failure=None) -> None:
        self.hardware_busy = False
        if error is not None:
            if isinstance(error, HardwareCancelled):
                self.status_var.set(f"{action}已停止：已取消后续动作并关闭气泵。")
            else:
                self.status_var.set(f"{action}失败：{error}")
                if action.startswith("取子机构舵机调试台"):
                    self.servo_var.set(f"取子机构舵机调试台：命令失败（{error}）")
            if on_failure is not None:
                on_failure(error)
            return
        self.status_var.set(f"{action}完成：{reply.response}")
        if on_success is not None:
            on_success(reply)

    def _turntable_receive(self) -> None:
        self._turntable(self.foundation_calibration.turntable_receive_duty, "0°接子")

    def _turntable_pickup(self) -> None:
        self._turntable(self.foundation_calibration.turntable_pickup_duty, "90°取子")

    def _turntable(self, duty: int, position: str) -> None:
        self.servo_var.set(f"取子机构舵机调试台：正在发送 {position}，占空比 {duty}……")
        self._hardware_reply(
            f"取子机构舵机调试台{position}（占空比 {duty}）",
            lambda: self.hardware.turntable(duty),
            on_success=lambda _reply: self._set_turntable_state(position, duty),
        )

    def _set_turntable_state(self, position: str, duty: int) -> None:
        self.turntable_duty = duty
        self.servo_var.set(f"取子机构舵机调试台：已发送 {position}，占空比 {duty}")

    def _pump(self, state: int, action: str) -> None:
        if state == 2 and self.hardware_busy:
            self._request_hardware_stop()
            self.status_var.set("已请求停止：当前机械臂指令完成后将取消后续动作并关闭气泵。")
            return
        self._hardware_reply(f"末端{action}", lambda: self.hardware.pump(state))

    def _hardware_cancel_requested(self) -> bool:
        event = getattr(self, "hardware_cancel_event", None)
        return bool(event is not None and event.is_set())

    def _clear_hardware_cancel_request(self) -> None:
        event = getattr(self, "hardware_cancel_event", None)
        if event is not None:
            event.clear()

    def _request_hardware_stop(self) -> None:
        event = getattr(self, "hardware_cancel_event", None)
        if event is not None:
            event.set()

    def _reset_hardware(self) -> None:
        self._hardware_reply("机械臂回零", self.hardware.reset)

    def _initialize_hardware(self) -> None:
        self._hardware_reply(
            "初始化机械臂",
            self.hardware.reset,
            on_success=lambda _reply: self.status_var.set("机械臂初始化完成：已执行复位，可继续进行调试。"),
        )

    def _move_hardware(self) -> None:
        try:
            values = [float(variable.get()) for variable in self.hardware_coordinate_vars]
        except ValueError:
            self.status_var.set("坐标必须是数字。")
            return
        if values[0] < 0:
            self.status_var.set("X 坐标不能为负数。")
            return
        point = tuple(values)
        if values[2] < DIRECT_MOVE_MIN_Z:
            self._hardware_reply(
                "低位坐标安全移动",
                lambda: self.hardware.move_via_approach(point),
            )
            return
        self._hardware_reply("空载坐标移动", lambda: self.hardware.move(*values))

    def _pickup_hardware(self) -> None:
        calibration = self.foundation_calibration
        self._hardware_reply(
            "机械臂取子流程",
            lambda: self.hardware.pickup(
                calibration.pickup_point,
                approach_z=calibration.safe_z,
                turntable_receive_duty=calibration.turntable_receive_duty,
                turntable_pickup_duty=calibration.turntable_pickup_duty,
                cancel_requested=self._hardware_cancel_requested,
            ),
            on_success=lambda _reply: self.status_var.set(
                "取子完成：已在保存的取子点 "
                f"X={calibration.pickup_point[0]:g}、Y={calibration.pickup_point[1]:g}、Z={calibration.pickup_point[2]:g} "
                "下降吸取，气泵保持开启；移动到落子点后再点击“放子”。"
            ),
        )

    def _drop_hardware(self) -> None:
        calibration = self.foundation_calibration
        self._hardware_reply(
            "机械臂放子流程",
            lambda: self.hardware.return_piece_to_pickup(
                calibration.pickup_point,
                calibration.standby_point,
                safe_z=calibration.safe_z,
                turntable_receive_duty=calibration.turntable_receive_duty,
                release_point=calibration.release_point,
                cancel_requested=self._hardware_cancel_requested,
            ),
            on_success=lambda _reply: self.status_var.set(
                "放子完成：已在放子点 "
                f"X={calibration.release_point[0]:g}、Y={calibration.release_point[1]:g}、Z={calibration.release_point[2]:g} "
                "正常泄压，机械臂已恢复待命。"
            ),
        )

    def _hardware_locked(self) -> None:
        self.status_var.set("五黑点取放测试尚未启用：先逐点完成空载标定和吸放测试。")

    def _close(self) -> None:
        if self.hardware_busy:
            self.status_var.set("硬件动作仍在执行，请等待回执后再关闭程序。")
            return
        self.closing = True
        if self.annotation_workbench is not None:
            try:
                self.annotation_workbench.close()
            except tk.TclError:
                pass
        self.camera.stop()
        self.hardware.disconnect()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
