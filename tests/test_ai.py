from pc_app.ai import AlphaBetaGomokuAI
from pc_app.board import GomokuBoard, Stone


def test_alpha_beta_defaults_to_four_ply() -> None:
    assert AlphaBetaGomokuAI().depth == 4


def test_alpha_beta_opens_in_the_center() -> None:
    result = AlphaBetaGomokuAI().choose(GomokuBoard(), Stone.BLACK)

    assert result.move == (6, 6)


def test_alpha_beta_makes_a_three_in_a_row_when_available() -> None:
    board = GomokuBoard()
    board.play((5, 6), Stone.BLACK)
    board.play((6, 6), Stone.BLACK)
    board.play((3, 3), Stone.WHITE)

    result = AlphaBetaGomokuAI().choose(board, Stone.BLACK)

    assert result.move in {(4, 6), (7, 6)}
    assert result.searched_nodes > 0


def test_alpha_beta_blocks_an_open_human_three() -> None:
    board = GomokuBoard()
    board.play((4, 6), Stone.WHITE)
    board.play((5, 6), Stone.WHITE)
    board.play((6, 6), Stone.WHITE)
    board.play((1, 1), Stone.BLACK)

    result = AlphaBetaGomokuAI().choose(board, Stone.BLACK)

    assert result.move in {(3, 6), (7, 6)}


def test_alpha_beta_wins_instead_of_blocking_a_human_three() -> None:
    board = GomokuBoard()
    for move in ((4, 6), (5, 6), (6, 6), (7, 6)):
        board.play(move, Stone.BLACK)
    for move in ((0, 0), (1, 0), (2, 0)):
        board.play(move, Stone.WHITE)

    result = AlphaBetaGomokuAI().choose(board, Stone.BLACK)

    assert result.move in {(3, 6), (8, 6)}
