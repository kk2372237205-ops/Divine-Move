from pc_app.protocol import ArmMoveCommand


def test_protocol_is_encoded_but_not_transmitted() -> None:
    command = ArmMoveCommand(300, -20.5, 62, retract=False)
    assert command.encode() == b"300.00/-20.50/62.00/False\n"


def test_retract_protocol_value() -> None:
    command = ArmMoveCommand(1, 2, 3, retract=True)
    assert command.encode().endswith(b"/True\n")
