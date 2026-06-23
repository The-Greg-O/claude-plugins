"""
perchan_varzz_bz2_v1 — per-channel delta zigzag variable-length pack + per-channel bz2-9.

Identical to varzz_bz2_v1 except each channel's varzz stream is compressed
independently with bz2. In varzz_bz2_v1, the 1.3 MB concatenated stream spans
~1.45 bz2 blocks (900 KB each): ch0 fully in block 1, ch1 split across blocks 1
and 2. The split prevents bz2's BWT from finding all 1688 heartbeat repetitions
in ch1 (649 periods in block 1, 1039 in block 2 — different BWT transforms).
Per-channel gives each 650 KB stream its own block → 1688 repetitions fully
visible to BWT for both channels.
"""
import bz2
import struct

import numpy as np

MAGIC = b'ECG1'
HEAD = 10


def _zigzag(d: np.ndarray) -> np.ndarray:
    d32 = d.astype(np.int32)
    return np.where(d32 >= 0, d32 * 2, -d32 * 2 - 1).astype(np.uint16)


def _unzigzag(z: np.ndarray) -> np.ndarray:
    z32 = z.astype(np.int32)
    return np.where(z32 & 1, -(z32 >> 1) - 1, z32 >> 1).astype(np.int16)


def _pack(zzag: np.ndarray) -> bytes:
    t0 = zzag < 128
    t1 = (zzag >= 128) & (zzag < 32640)
    t2 = zzag >= 32640
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

    # Compress each channel independently so each 650 KB stream fits in one
    # bz2 block (900 KB), giving BWT access to all ~1688 heartbeat periods.
    chan_parts = []
    for c in range(n_chan):
        zzag = _zigzag(diff[c])
        packed = _pack(zzag)
        compressed = bz2.compress(packed, compresslevel=9)
        chan_parts.append(struct.pack('<I', len(compressed)) + compressed)

    payload = b''.join(chan_parts)
    # Store header uncompressed; channel compressed blobs follow.
    return blob[:HEAD] + struct.pack('<I', n_chan) + payload


def decode(blob: bytes) -> bytes:
    header = blob[:HEAD]
    assert header[:4] == MAGIC
    n_chan, n_samp = struct.unpack_from('<HI', header, 4)

    pos = HEAD + 4  # skip n_chan uint32
    channels = []
    for _ in range(n_chan):
        clen = struct.unpack_from('<I', blob, pos)[0]
        pos += 4
        packed = bz2.decompress(blob[pos:pos + clen])
        pos += clen
        zzag = _unpack(packed, n_samp)
        delta = _unzigzag(zzag)
        channels.append(delta)

    diff = np.stack(channels, axis=0).astype(np.int64)
    sig = np.cumsum(diff, axis=1)
    sig = ((sig + 32768) % 65536 - 32768).astype('<i2')
    return header + sig.tobytes()
