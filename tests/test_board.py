from pc_app.board import GomokuBoard, Stone, choose_move


def test_horizontal_five_is_a_win() -> None:
    board = GomokuBoard()
    for column in range(5):
        board.play((column, 4), Stone.BLACK)
    assert board.winner() == Stone.BLACK


def test_ai_takes_immediate_win() -> None:
    board = GomokuBoard()
    for column in range(4):
        board.play((column, 6), Stone.WHITE)
    assert choose_move(board, Stone.WHITE) == (4, 6)


def test_ai_blocks_immediate_loss() -> None:
    board = GomokuBoard()
    for column in range(4):
        board.play((column, 6), Stone.BLACK)
    assert choose_move(board, Stone.WHITE) == (4, 6)


def test_first_ai_move_uses_center() -> None:
    board = GomokuBoard()
    assert choose_move(board) == (6, 6)


def test_replace_stones_loads_a_camera_confirmed_state() -> None:
    board = GomokuBoard()
    board.play((0, 0), Stone.BLACK)

    board.replace_stones({(6, 6): Stone.BLACK, (7, 6): Stone.WHITE})

    assert board.stone_at((0, 0)) == Stone.EMPTY
    assert board.stone_at((6, 6)) == Stone.BLACK
    assert board.stone_at((7, 6)) == Stone.WHITE


def test_undo_last_restores_the_previous_state() -> None:
    board = GomokuBoard()
    board.play((6, 6), Stone.BLACK)

    assert board.undo_last() == ((6, 6), Stone.BLACK)
    assert board.stone_at((6, 6)) == Stone.EMPTY
