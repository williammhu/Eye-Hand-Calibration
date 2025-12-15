"""
Freenove robot arm client that speaks the same TCP text protocol as the official app.

The Freenove desktop UI (see ``main.py``/``client.py``) sends simple ASCII
commands such as ``G0 X0 Y200 Z45`` to port 5000.  This wrapper reuses the
existing `Client` class so calibration scripts can drive the arm directly
without duplicating socket handling code.  When ``dry_run=True`` commands are
printed instead of sent, which is handy for debugging on a machine without the
arm attached.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from client import Client
from command import Command


@dataclass
class FreenoveArmClient:
    host: str = "10.149.65.232"
    port: int = 5000
    dry_run: bool = False
    auto_enable: bool = True
    verbose: bool = False

    _client: Optional[Client] = field(default=None, init=False, repr=False)
    _cmd: Command = field(default_factory=Command, init=False, repr=False)

    def __enter__(self) -> "FreenoveArmClient":
        self.connect()
        if self.auto_enable:
            self.enable_motors(True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # Connection helpers -------------------------------------------------
    def connect(self) -> None:
        """Open the TCP connection unless running in dry-run mode."""

        if self.dry_run:
            return
        if self._client is None:
            self._client = Client()
            self._client.port = self.port
            if not self._client.connect(self.host):
                self._client = None
                raise RuntimeError(f"Could not connect to Freenove arm at {self.host}:{self.port}")

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.disconnect()
            finally:
                self._client = None

    # Command primitives --------------------------------------------------
    def _send(self, text: str) -> None:
        """Send a raw command line, adding CRLF to match the app behaviour."""

        if self.verbose or self.dry_run:
            prefix = "[dry-run]" if self.dry_run else "[send]"
            print(f"{prefix} {text}")
        if self.dry_run:
            return

        if self._client is None or not self._client.connect_flag:
            self.connect()
        if self._client is None or not self._client.connect_flag:
            raise RuntimeError("Socket is not connected")
        self._client.send_messages(text + "\r\n")

    def enable_motors(self, enable: bool = True) -> None:
        """
        Mirror the UI \"Load Motor\" toggle.
        From the stock UI (see freenove_source_code/main.py): S8 E0 = load/enable, S8 E1 = relax.
        """

        state = "0" if enable else "1"
        cmd = f"{self._cmd.CUSTOM_ACTION}8 {self._cmd.ARM_ENABLE}{state}"
        self._send(cmd)

    def move_to(self, x: float, y: float, z: float, speed: int | None = None) -> None:
        """
        Absolute move in millimetres.

        The native protocol encodes position as ``G0 X.. Y.. Z..``.  A feed
        rate/speed parameter is not currently supported by the firmware; the
        optional ``speed`` argument is accepted for API compatibility and
        ignored if provided.
        """

        cmd = (
            f"{self._cmd.MOVE_ACTION}0 "
            f"{self._cmd.AXIS_X_ACTION}{x:.1f} "
            f"{self._cmd.AXIS_Y_ACTION}{y:.1f} "
            f"{self._cmd.AXIS_Z_ACTION}{z:.1f}"
        )
        self._send(cmd)

    def wait(self, seconds: float) -> None:
        time.sleep(seconds)
