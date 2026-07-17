"""Livox SDK2 control channel -- arm the Mid-360 so it streams point cloud.

The point-cloud receiver (receiver.py) is passive: it only works while the
sensor is already in the SAMPLING state pushing to this host. After a power
cycle the Mid-360 comes up in IDLE and stays there until something sends it a
Parameter Configuration command (LivoxViewer2 does this; when you close it the
sensor drops back). This module sends that command ourselves so the tool is
self-sufficient.

Protocol: Livox LiDAR Communication Protocol -- Mid-360 (control command on
LiDAR UDP port 56100, host replies received on 56101). Frame is a 24-byte
header + data, header guarded by CRC-16/CCITT-FALSE and data by CRC-32.
Everything little-endian.
"""

from __future__ import annotations

import binascii
import socket
import struct

CMD_PORT_LIDAR = 56100       # sensor listens for control commands here
CMD_PORT_HOST = 56101        # recommended host port for the ACK
CMD_ID_PARAM_CONFIG = 0x0100

# key_value_list keys (see protocol "0x0100 Parameter Configuration")
KEY_PCL_DATA_TYPE = 0x0000       # 1 = Cartesian 32-bit (matches packets.py)
KEY_POINTCLOUD_HOST_IPCFG = 0x0006
KEY_WORK_TGT_MODE = 0x001A       # target work mode

WORK_MODE_SAMPLING = 0x01
PCL_DATA_TYPE_CARTESIAN32 = 0x01
LIDAR_POINT_SRC_PORT = 56300     # sensor's default point-cloud source port


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, xorout 0."""
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def build_frame(cmd_id: int, data: bytes, seq_num: int) -> bytes:
    """Wrap a control-command data field in the Mid-360 frame envelope."""
    length = 24 + len(data)  # sof..end of data
    header = struct.pack(
        "<BBHIHBB6x",
        0xAA,       # sof
        0x00,       # version
        length,
        seq_num,
        cmd_id,
        0x00,       # cmd_type REQ
        0x00,       # sender_type host
    )  # 18 bytes, resv zero-filled
    c16 = crc16(header)
    c32 = binascii.crc32(data) & 0xFFFFFFFF  # 0 for empty data, as spec requires
    return header + struct.pack("<HI", c16, c32) + data


def _ip_bytes(ip: str) -> bytes:
    parts = [int(p) for p in ip.split(".")]
    if len(parts) != 4 or any(not 0 <= p <= 255 for p in parts):
        raise ValueError(f"bad IPv4 address: {ip!r}")
    return bytes(parts)


def _kv(key: int, value: bytes) -> bytes:
    return struct.pack("<HH", key, len(value)) + value


def build_arm_data(push_ip: str, point_port: int) -> bytes:
    """key_value_list that points the point cloud at push_ip:point_port,
    forces Cartesian-32 data, and requests the SAMPLING work mode."""
    ipcfg = _ip_bytes(push_ip) + struct.pack("<HH", point_port, LIDAR_POINT_SRC_PORT)
    kvs = (
        _kv(KEY_POINTCLOUD_HOST_IPCFG, ipcfg)
        + _kv(KEY_PCL_DATA_TYPE, bytes([PCL_DATA_TYPE_CARTESIAN32]))
        + _kv(KEY_WORK_TGT_MODE, bytes([WORK_MODE_SAMPLING]))
    )
    key_num = 3
    return struct.pack("<HH", key_num, 0) + kvs


class LivoxController:
    """Send the arm (Parameter Configuration) command to a Mid-360."""

    def __init__(self, sensor_ip: str, host_ip: str = "0.0.0.0",
                 host_cmd_port: int = CMD_PORT_HOST, timeout: float = 2.0) -> None:
        self.sensor_ip = sensor_ip
        self.host_ip = host_ip
        self.host_cmd_port = host_cmd_port
        self.timeout = timeout
        self._seq = 0

    def arm(self, push_ip: str, point_port: int) -> None:
        """Program the point-cloud destination and put the sensor into SAMPLING.

        Raises RuntimeError on a non-zero ACK, TimeoutError if the sensor never
        replies (wrong IP / not reachable / cmd port held by another app).
        """
        self._seq += 1
        data = build_arm_data(push_ip, point_port)
        frame = build_frame(CMD_ID_PARAM_CONFIG, data, self._seq)

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((self.host_ip, self.host_cmd_port))
            s.settimeout(self.timeout)
            s.sendto(frame, (self.sensor_ip, CMD_PORT_LIDAR))
            ret = self._await_ack(s)
        finally:
            s.close()
        if ret != 0x00:
            raise RuntimeError(
                f"sensor rejected arm command (ret_code=0x{ret:02X}); see "
                "protocol 'Return Code Description'"
            )

    def _await_ack(self, s: socket.socket) -> int:
        """Return ret_code of the matching 0x0100 ACK. TimeoutError if none."""
        while True:
            payload, _addr = s.recvfrom(1500)  # raises socket.timeout -> TimeoutError
            if len(payload) < 25:
                continue
            (sof, _ver, _len, _seq, cmd_id, cmd_type, _sender) = struct.unpack_from(
                "<BBHIHBB", payload, 0
            )
            if sof != 0xAA or cmd_id != CMD_ID_PARAM_CONFIG or cmd_type != 0x01:
                continue
            return payload[24]  # data[0] = ret_code
