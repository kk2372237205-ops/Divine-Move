"""Deterministic Gomoku search using minimax and alpha-beta pruning."""

from __future__ import annotations

from dataclasses import dataclass

from .board import GomokuBoard, Move, Stone


MATE_SCORE = 1_000_000
DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (1, 1), (1, -1))


@dataclass(frozen=True)
class SearchResult:
    move: Move | None
    score: int
    searched_nodes: int


class AlphaBetaGomokuAI:
    """A bounded, explainable minimax player for a 13 x 13 board."""

    def __init__(self, depth: int = 4, max_candidates: int = 14) -> None:
        if depth < 1:
            raise ValueError("搜索深度至少为 1")
        if max_candidates < 1:
            raise ValueError("候选点数量至少为 1")
        self.depth = depth
        self.max_candidates = max_candidates
        self.searched_nodes = 0

    def choose(self, board: GomokuBoard, stone: Stone) -> SearchResult:
        if stone == Stone.EMPTY:
            raise ValueError("AI 不能执空棋")
        if board.winner() is not None or board.is_full():
            return SearchResult(None, 0, 0)

        self.searched_nodes = 0
        opponent = self._opponent(stone)
        candidates = self._candidate_moves(board, stone)
        for move in candidates:
            if board.would_win(move, stone):
                return SearchResult(move, MATE_SCORE, self.searched_nodes)
        for move in self._candidate_moves(board, opponent):
            if board.would_win(move, opponent):
                return SearchResult(move, MATE_SCORE - 1, self.searched_nodes)

        best_move: Move | None = None
        best_score = -MATE_SCORE * 2
        best_tactical_priority = -1
        alpha, beta = -MATE_SCORE * 2, MATE_SCORE * 2
        for move in candidates:
            tactical_priority = self._tactical_priority(board, move, stone, opponent)
            board.play(move, stone)
            score = self._search(board, opponent, stone, self.depth - 1, alpha, beta)
            board.undo_last()
            # The regulation explicitly evaluates forming a connected three and
            # blocking a human connected three.  Preserve alpha-beta's score
            # within a tactic class, but never let a quieter, higher-scoring
            # future shape skip a currently available required tactic.
            if tactical_priority > best_tactical_priority or (
                tactical_priority == best_tactical_priority and score > best_score
            ):
                best_move, best_score = move, score
                best_tactical_priority = tactical_priority
            alpha = max(alpha, best_score)
        return SearchResult(best_move, best_score, self.searched_nodes)

    def _search(
        self,
        board: GomokuBoard,
        current: Stone,
        root: Stone,
        depth: int,
        alpha: int,
        beta: int,
    ) -> int:
        self.searched_nodes += 1
        winner = board.winner()
        if winner == root:
            return MATE_SCORE + depth
        if winner == self._opponent(root):
            return -MATE_SCORE - depth
        if depth == 0 or board.is_full():
            return self._evaluate(board, root)

        moves = self._candidate_moves(board, current)
        if current == root:
            value = -MATE_SCORE * 2
            for move in moves:
                board.play(move, current)
                value = max(value, self._search(board, self._opponent(current), root, depth - 1, alpha, beta))
                board.undo_last()
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        value = MATE_SCORE * 2
        for move in moves:
            board.play(move, current)
            value = min(value, self._search(board, self._opponent(current), root, depth - 1, alpha, beta))
            board.undo_last()
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    def _candidate_moves(self, board: GomokuBoard, stone: Stone) -> list[Move]:
        if not board.moves:
            return [(board.size // 2, board.size // 2)]
        candidates: set[Move] = set()
        for (column, row), _ in board.moves:
            for delta_y in range(-2, 3):
                for delta_x in range(-2, 3):
                    if delta_x == 0 and delta_y == 0:
                        continue
                    move = (column + delta_x, row + delta_y)
                    if board.in_bounds(move) and board.stone_at(move) == Stone.EMPTY:
                        candidates.add(move)
        opponent = self._opponent(stone)
        ordered = sorted(
            candidates,
            key=lambda move: (-self._move_priority(board, move, stone, opponent), move[1], move[0]),
        )
        return ordered[: self.max_candidates]

    def _move_priority(self, board: GomokuBoard, move: Move, stone: Stone, opponent: Stone) -> int:
        return self._point_strength(board, move, stone) * 2 + self._point_strength(board, move, opponent)

    @classmethod
    def _tactical_priority(cls, board: GomokuBoard, move: Move, stone: Stone, opponent: Stone) -> int:
        """Rank regulation-visible tactics before long-term evaluation.

        Immediate wins and blocks of an opponent's immediate win are handled
        earlier by ``choose``.  Here, preventing an opponent's existing three
        from becoming four outranks extending our own line to three.
        """
        if cls._longest_line_through(board, move, opponent) >= 4:
            return 2
        if cls._longest_line_through(board, move, stone) >= 3:
            return 1
        return 0

    @staticmethod
    def _longest_line_through(board: GomokuBoard, move: Move, stone: Stone) -> int:
        """Return the longest line that ``stone`` would make by playing ``move``."""
        column, row = move
        longest = 1
        for delta_x, delta_y in DIRECTIONS:
            length = 1
            for sign in (-1, 1):
                next_column = column + sign * delta_x
                next_row = row + sign * delta_y
                while board.in_bounds((next_column, next_row)) and board.stone_at((next_column, next_row)) == stone:
                    length += 1
                    next_column += sign * delta_x
                    next_row += sign * delta_y
            longest = max(longest, length)
        return longest

    @staticmethod
    def _point_strength(board: GomokuBoard, move: Move, stone: Stone) -> int:
        column, row = move
        strongest = 0
        for delta_x, delta_y in DIRECTIONS:
            length = 1
            open_ends = 0
            for sign in (-1, 1):
                next_column, next_row = column + sign * delta_x, row + sign * delta_y
                while board.in_bounds((next_column, next_row)) and board.stone_at((next_column, next_row)) == stone:
                    length += 1
                    next_column += sign * delta_x
                    next_row += sign * delta_y
                if board.in_bounds((next_column, next_row)) and board.stone_at((next_column, next_row)) == Stone.EMPTY:
                    open_ends += 1
            strongest = max(strongest, AlphaBetaGomokuAI._line_value(length, open_ends))
        return strongest

    def _evaluate(self, board: GomokuBoard, root: Stone) -> int:
        return self._side_score(board, root) - self._side_score(board, self._opponent(root))

    @staticmethod
    def _side_score(board: GomokuBoard, stone: Stone) -> int:
        score = 0
        for row in range(board.size):
            for column in range(board.size):
                if board.stone_at((column, row)) != stone:
                    continue
                for delta_x, delta_y in DIRECTIONS:
                    previous = (column - delta_x, row - delta_y)
                    if board.in_bounds(previous) and board.stone_at(previous) == stone:
                        continue
                    length = 0
                    next_column, next_row = column, row
                    while board.in_bounds((next_column, next_row)) and board.stone_at((next_column, next_row)) == stone:
                        length += 1
                        next_column += delta_x
                        next_row += delta_y
                    open_ends = int(board.in_bounds(previous) and board.stone_at(previous) == Stone.EMPTY)
                    open_ends += int(
                        board.in_bounds((next_column, next_row))
                        and board.stone_at((next_column, next_row)) == Stone.EMPTY
                    )
                    score += AlphaBetaGomokuAI._line_value(length, open_ends)
        return score

    @staticmethod
    def _line_value(length: int, open_ends: int) -> int:
        if length >= 5:
            return MATE_SCORE
        if length == 4:
            return 120_000 if open_ends == 2 else 25_000 if open_ends == 1 else 0
        if length == 3:
            return 8_000 if open_ends == 2 else 1_200 if open_ends == 1 else 0
        if length == 2:
            return 450 if open_ends == 2 else 80 if open_ends == 1 else 0
        return 12 if open_ends == 2 else 3 if open_ends == 1 else 0

    @staticmethod
    def _opponent(stone: Stone) -> Stone:
        return Stone.WHITE if stone == Stone.BLACK else Stone.BLACK


def choose_move_alpha_beta(board: GomokuBoard, stone: Stone = Stone.WHITE) -> Move | None:
    """Convenience API retained for the UI and simple callers."""
    return AlphaBetaGomokuAI().choose(board, stone).move
