"""Coordinate conventions kept separate from camera and arm calibration."""

from __future__ import annotations

from .board import BOARD_SIZE, Move


def board_to_competition(move: Move, size: int = BOARD_SIZE) -> tuple[int, int]:
    """Convert internal upper-left 0-based cells to the required match system.

    Internal coordinates use `(column, row)`, 0 through 12, because that is
    natural for image processing.  The competition convention uses positive
    integers and takes the board's upper-right corner, when facing the robot,
    as `(1, 1)`.  The column therefore reverses during conversion.
    """
    column, row = move
    if not 0 <= column < size or not 0 <= row < size:
        raise ValueError(f"Move out of board: {move}")
    return size - column, row + 1


def competition_to_board(coordinate: tuple[int, int], size: int = BOARD_SIZE) -> Move:
    """Inverse of :func:`board_to_competition`."""
    x, y = coordinate
    if not 1 <= x <= size or not 1 <= y <= size:
        raise ValueError(f"Competition coordinate out of board: {coordinate}")
    return size - x, y - 1
