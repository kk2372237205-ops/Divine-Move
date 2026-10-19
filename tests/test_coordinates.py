import pytest

from pc_app.coordinates import board_to_competition, competition_to_board


def test_competition_origin_is_upper_right() -> None:
    assert board_to_competition((12, 0)) == (1, 1)
    assert board_to_competition((0, 0)) == (13, 1)
    assert board_to_competition((12, 12)) == (1, 13)


def test_coordinate_conversion_round_trip() -> None:
    for move in ((0, 0), (6, 6), (12, 12)):
        assert competition_to_board(board_to_competition(move)) == move


def test_invalid_coordinates_are_rejected() -> None:
    with pytest.raises(ValueError):
        board_to_competition((13, 0))
    with pytest.raises(ValueError):
        competition_to_board((0, 1))
