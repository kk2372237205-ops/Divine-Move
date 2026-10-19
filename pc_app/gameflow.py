"""Autonomous game-turn logic independent from camera and arm I/O."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .ai import AlphaBetaGomokuAI, SearchResult
from .board import GomokuBoard, Move, Stone
from .motion import MotionPlan, MotionPlanner


class GamePhase(str, Enum):
    ROBOT_BLACK_TURN = "robot_black_turn"
    WAITING_FOR_HUMAN_WHITE = "waiting_for_human_white"
    FINISHED = "finished"
    INVALID_BOARD = "invalid_board"


@dataclass(frozen=True)
class GameFlowUpdate:
    phase: GamePhase
    message: str
    board: GomokuBoard
    plan: MotionPlan | None = None
    search: SearchResult | None = None


class AutonomousGomokuController:
    """Determine turns from camera-confirmed stones and prepare robot plans.

    It never changes the physical board and never assumes a robot action has
    succeeded.  After a plan is shown, the next transition comes only from a
    new stable camera state containing the placed black stone.
    """

    def __init__(self, ai: AlphaBetaGomokuAI | None = None, planner: MotionPlanner | None = None) -> None:
        self.ai = ai or AlphaBetaGomokuAI()
        self.planner = planner or MotionPlanner()
        self._last_planned_state: tuple[tuple[Move, Stone], ...] | None = None

    def reset(self) -> None:
        self._last_planned_state = None

    def observe(self, stones: Mapping[Move, Stone]) -> GameFlowUpdate:
        board = GomokuBoard()
        try:
            board.replace_stones(stones)
        except ValueError as error:
            return GameFlowUpdate(GamePhase.INVALID_BOARD, f"视觉棋局无效：{error}", board)

        black_count = sum(stone == Stone.BLACK for _, stone in board.moves)
        white_count = sum(stone == Stone.WHITE for _, stone in board.moves)
        if black_count not in (white_count, white_count + 1):
            return GameFlowUpdate(
                GamePhase.INVALID_BOARD,
                f"视觉棋局无效：黑棋 {black_count} 颗，白棋 {white_count} 颗，不符合黑棋先行规则。",
                board,
            )

        # A visible five-in-a-row is a result only when the state also has a
        # legal black-first turn count.  This prevents a missed robot stone
        # plus an early human stone from being presented as a valid victory.
        winner = board.winner()
        if winner == Stone.BLACK:
            return GameFlowUpdate(GamePhase.FINISHED, "黑棋五连，机器人获胜。", board)
        if winner == Stone.WHITE:
            return GameFlowUpdate(GamePhase.FINISHED, "白棋五连，人类获胜。", board)
        if board.is_full():
            return GameFlowUpdate(GamePhase.FINISHED, "棋盘已满，平局。", board)

        if black_count == white_count + 1:
            return GameFlowUpdate(
                GamePhase.WAITING_FOR_HUMAN_WHITE,
                "轮到人类白棋落子，等待摄像头确认一颗新的白棋。",
                board,
            )

        state = tuple(sorted(stones.items()))
        if state == self._last_planned_state:
            return GameFlowUpdate(
                GamePhase.ROBOT_BLACK_TURN,
                "机器人黑棋计划已生成，等待机械臂完成后由摄像头确认结果。",
                board,
            )
        search = self.ai.choose(board, Stone.BLACK)
        if search.move is None:
            return GameFlowUpdate(GamePhase.FINISHED, "没有可用落子，棋局结束。", board, search=search)
        plan = self.planner.plan_black_move(search.move)
        self._last_planned_state = state
        return GameFlowUpdate(
            GamePhase.ROBOT_BLACK_TURN,
            "轮到机器人黑棋：已根据稳定视觉棋局生成取放计划。",
            board,
            plan=plan,
            search=search,
        )
