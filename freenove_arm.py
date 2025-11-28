"""
Lightweight Freenove robot arm client used by the planar hand-eye calibration script.

This module does not implement the full Freenove protocol; instead it offers a
simple TCP-based interface that mirrors the commands provided by the official
Python server.  If you run the Freenove "Server.py" on the Raspberry Pi, the
server accepts a single-line JSON command such as:

    {"cmd": "move", "x": 100, "y": 150, "z": 80, "speed": 50}

The :class:`FreenoveArmClient` class sends these commands to the configured
``host``/``port``.  When ``dry_run=True`` (the default), commands are only
printed so the script can be exercised on a development machine without the
hardware.
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Optional


@dataclass
class FreenoveArmClient:
    """Minimal client for the Freenove Robot Arm server.

    Parameters
    ----------
    host: str
        IP address of the Raspberry Pi running the Freenove server.
    port: int
        TCP port exposed by the Freenove server (default 20001 in the examples).
    dry_run: bool
        When ``True``, print commands instead of sending them to hardware.
    timeout: float
        Socket timeout in seconds when ``dry_run`` is ``False``.
    """

    host: str = "127.0.0.1"
    port: int = 20001
    dry_run: bool = True
    timeout: float = 2.0

    _socket: Optional[socket.socket] = None

    def __enter__(self) -> "FreenoveArmClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def connect(self) -> None:
        """Open the TCP connection unless running in dry-run mode."""

        if self.dry_run:
            return
        if self._socket is None:
            self._socket = socket.create_connection((self.host, self.port), timeout=self.timeout)

    def close(self) -> None:
        """Close the underlying socket if it was opened."""

        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None

    def move_to(self, x: float, y: float, z: float, speed: int = 50) -> None:
        """Send a cartesian move command to the robot arm.

        The Freenove server understands JSON objects terminated by a newline.  If
        you use a different command schema, adjust the payload in this method.
        """

        payload = {"cmd": "move", "x": x, "y": y, "z": z, "speed": speed}
        encoded = json.dumps(payload)
        if self.dry_run:
            print(f"[dry-run] send -> {encoded}")
            return

        if self._socket is None:
            self.connect()
        if self._socket is None:
            raise RuntimeError("Socket could not be opened")

        self._socket.sendall(encoded.encode("utf-8") + b"\n")

    def wait(self, seconds: float) -> None:
        """Placeholder for compatibility with the calibration loop."""

        # The Freenove reference server does not expose an explicit wait command.
        # The caller can `time.sleep` directly, but we keep this method to match
        # the calibration code structure and to leave room for future status
        # polling.
        return
