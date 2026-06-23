"""
varzz_bz2_v1 — per-channel delta (int16 wraparound) → zigzag encode →
variable-length byte pack → bz2-9.

99.57% of ECG deltas have |d| < 64 → zzag < 128 → stream ≈ 1.3 MB
(half of the 2.6 MB int16 stream). bz2 BWT then exploits the ≈385-byte
heartbeat period on this half-size stream.

Variable-length encoding (by first byte b0):
  b0 < 0x80          → 1 byte:  zzag = b0                           (0..127)
  b0 in [0x80, 0xFE] → 2 bytes: zzag = 128 + ((b0 & 0x7F) << 8 | b1)  (128..32639)
  b0 == 0xFF         → 3 bytes: zzag = b1 << 8 | b2                 (0..65535 catch-all)
"""
import bz2
import struct

import numpy as np

MAGIC = b'ECG1'
HEAD = 10


def _zigzag(d: np.ndarray) -> np.ndarray:
    """int16 ndarray -> uint16 zigzag: 0->0, -1->1, 1->2, ..."""
    d32 = d.astype(np.int32)
    return np.where(d32 >= 0, d32 * 2, -d32 * 2 - 1).astype(np.uint16)


def _unzigzag(z: np.ndarray) -> np.ndarray:
    """uint16 zigzag -> int16."""
    z32 = z.astype(np.int32)
    return np.where(z32 & 1, -(z32 >> 1) - 1, z32 >> 1).astype(np.int16)


def _pack(zzag: np.ndarray) -> bytes:
    """Vectorized variable-length pack of a uint16 array."""
    t0 = zzag < 128
    t1 = (zzag >= 128) & (zzag < 32640)
    t2 = zzag >= 32640  # practically empty for ECG data

    sizes = np.ones(len(zzag), dtype=np.intp)
    sizes[t1] = 2
    sizes[t2] = 3

    offsets = np.empty(len(zzag) + 1, dtype=np.intp)
    offsets[0] = 0
    np.cumsum(sizes, out=offsets[1:])
    out = np.zeros(int(offsets[-1]), dtype=np.uint8)

    if t0.any():
        out[offsets[:-1][t0]] = zzag[t0].astype(np.uint8)

    if t1.any():
        p = offsets[:-1][t1]
        v = zzag[t1].astype(np.uint32) - 128
        out[p] = (0x80 | (v >> 8)).astype(np.uint8)
        out[p + 1] = (v & 0xFF).astype(np.uint8)

    if t2.any():
        p = offsets[:-1][t2]
        v = zzag[t2]
        out[p] = np.uint8(0xFF)
        out[p + 1] = (v >> 8).astype(np.uint8)
        out[p + 2] = (v & 0xFF).astype(np.uint8)

    return bytes(out)


def _unpack(data: bytes, n: int) -> np.ndarray:
    """Sequential variable-length decode into uint16 array of length n."""
    mv = memoryview(data)
    zzag = np.empty(n, dtype=np.uint16)
    pos = 0
    for i in range(n):
        b0 = mv[pos]
        if b0 < 0x80:
            zzag[i] = b0
            pos += 1
        elif b0 < 0xFF:
            zzag[i] = 128 + (((b0 & 0x7F) << 8) | mv[pos + 1])
            pos += 2
        else:
            zzag[i] = (mv[pos + 1] << 8) | mv[pos + 2]
            pos += 3
    return zzag


def encode(blob: bytes) -> bytes:
    assert blob[:4] == MAGIC
    n_chan, n_samp = struct.unpack_from('<HI', blob, 4)
    sig = np.frombuffer(blob[HEAD:], dtype='<i2').reshape(n_chan, n_samp)

    d = sig.astype(np.int64)
    diff = np.empty_like(d)
    diff[:, 0] = d[:, 0]
    diff[:, 1:] = np.diff(d, axis=1)
    diff = ((diff + 32768) % 65536 - 32768).astype('<i2')

    zzag = _zigzag(diff.ravel())
    packed = _pack(zzag)

    payload = struct.pack('<I', len(packed)) + packed
    return bz2.compress(blob[:HEAD] + payload, compresslevel=9)


def decode(blob: bytes) -> bytes:
    raw = bz2.decompress(blob)
    assert raw[:4] == MAGIC
    n_chan, n_samp = struct.unpack_from('<HI', raw, 4)

    plen = struct.unpack_from('<I', raw, HEAD)[0]
    packed = raw[HEAD + 4:HEAD + 4 + plen]

    zzag = _unpack(packed, n_chan * n_samp)
    delta = _unzigzag(zzag).reshape(n_chan, n_samp)

    sig = np.cumsum(delta.astype(np.int64), axis=1)
    sig = ((sig + 32768) % 65536 - 32768).astype('<i2')

    return raw[:HEAD] + sig.tobytes()
