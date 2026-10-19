"""Read-only serial port discovery.  This module never opens a port."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SerialPortInfo:
    device: str
    description: str
    hardware_id: str


def list_serial_ports() -> tuple[list[SerialPortInfo], str | None]:
    try:
        from serial.tools import list_ports
    except ImportError:
        return [], "未安装 pyserial；请先运行 .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt。"

    ports = [
        SerialPortInfo(port.device, port.description or "无描述", port.hwid or "")
        for port in list_ports.comports()
    ]
    return sorted(ports, key=lambda port: port.device), None
