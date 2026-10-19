"""Pure Gomoku rules and a small deterministic starter AI.

All functions in this module are hardware-independent.  This allows us to
test the game flow before a move is ever translated into a robot command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterator, Mapping


BOARD_SIZE = 13


class Stone(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2


Move = tuple[int, int]


@dataclass
class GomokuBoard:
    size: int = BOARD_SIZE
    cells: list[list[Stone]] = field(init=False)
    moves: list[tuple[Move, Stone]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.size < 5:
            raise ValueError("The board must be at least 5 x 5")
        self.cells = [[Stone.EMPTY for _ in range(self.size)] for _ in range(self.size)]

    def reset(self) -> None:
        self.cells = [[Stone.EMPTY for _ in range(self.size)] for _ in range(self.size)]
        self.moves.clear()

    def replace_stones(self, stones: Mapping[Move, Stone]) -> None:
        """Load a full board state confirmed by the camera."""
        self.reset()
        for move, stone in sorted(stones.items(), key=lambda item: (item[0][1], item[0][0])):
            self.play(move, stone)

    def undo_last(self) -> tuple[Move, Stone]:
        """Undo the newest move, used internally by the bounded AI search."""
        if not self.moves:
            raise ValueError("没有可撤销的落子")
        move, stone = self.moves.pop()
        column, row = move
        self.cells[row][column] = Stone.EMPTY
        return move, stone

    def in_bounds(self, move: Move) -> bool:
        column, row = move
        return 0 <= column < self.size and 0 <= row < self.size

    def stone_at(self, move: Move) -> Stone:
        if not self.in_bounds(move):
            raise ValueError(f"Move out of board: {move}")
        column, row = move
        return self.cells[row][column]

    def play(self, move: Move, stone: Stone) -> None:
        if stone == Stone.EMPTY:
            raise ValueError("Cannot play an empty stone")
        if not self.in_bounds(move):
            raise ValueError(f"Move out of board: {move}")
        if self.stone_at(move) != Stone.EMPTY:
            raise ValueError(f"Cell is already occupied: {move}")
        column, row = move
        self.cells[row][column] = stone
        self.moves.append((move, stone))

    def is_full(self) -> bool:
        return all(cell != Stone.EMPTY for row in self.cells for cell in row)

    def winner(self) -> Stone | None:
        for row in range(self.size):
            for column in range(self.size):
                stone = self.cells[row][column]
                if stone == Stone.EMPTY:
                    continue
                if any(self._line_length(column, row, dx, dy, stone) >= 5 for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1))):
                    return stone
        return None

    def _line_length(self, column: int, row: int, dx: int, dy: int, stone: Stone) -> int:
        length = 1
        for sign in (-1, 1):
            next_column, next_row = column + sign * dx, row + sign * dy
            while 0 <= next_column < self.size and 0 <= next_row < self.size:
                if self.cells[next_row][next_column] != stone:
                    break
                length += 1
                next_column += sign * dx
                next_row += sign * dy
        return length

    def empty_moves(self) -> Iterator[Move]:
        for row in range(self.size):
            for column in range(self.size):
                if self.cells[row][column] == Stone.EMPTY:
                    yield (column, row)

    def would_win(self, move: Move, stone: Stone) -> bool:
        if self.stone_at(move) != Stone.EMPTY:
            return False
        column, row = move
        self.cells[row][column] = stone
        try:
            return self.winner() == stone
        finally:
            self.cells[row][column] = Stone.EMPTY


def choose_move(board: GomokuBoard, stone: Stone = Stone.WHITE) -> Move | None:
    """Choose with minimax search and alpha-beta pruning."""
    from .ai import choose_move_alpha_beta

    return choose_move_alpha_beta(board, stone)
