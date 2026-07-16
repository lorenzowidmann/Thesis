"""UDP listener for the Livox SDK2 point-cloud stream.

The device must already be streaming to this host (configure the host IP once
with LivoxViewer2, then close it -- see README). This just binds the data port
and yields decoded point blocks. No control handshake is sent.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator

from .packets import PointBlock, parse_packet

DEFAULT_DATA_PORT = 57000  # Livox SDK2 default point-cloud destination port
_MAX_UDP = 1500


class LivoxReceiver:
    def __init__(self, host_ip: str = "0.0.0.0", data_port: int = DEFAULT_DATA_PORT,
                 timeout: float = 3.0) -> None:
        self.host_ip = host_ip
        self.data_port = data_port
        self.timeout = timeout
        self._sock: socket.socket | None = None

    def __enter__(self) -> "LivoxReceiver":
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host_ip, self.data_port))
        s.settimeout(self.timeout)
        self._sock = s
        return self

    def __exit__(self, *exc) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def blocks(self) -> Iterator[PointBlock]:
        """Yield decoded point blocks until a socket timeout (no data)."""
        assert self._sock is not None, "use LivoxReceiver as a context manager"
        while True:
            try:
                payload, _addr = self._sock.recvfrom(_MAX_UDP)
            except socket.timeout:
                return
            block = parse_packet(payload)
            if block is not None:
                yield block
