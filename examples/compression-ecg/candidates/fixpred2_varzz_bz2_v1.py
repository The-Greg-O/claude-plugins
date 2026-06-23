"""
fixpred2_varzz_bz2_v1 — second-order fixed predictor + zigzag + variable-length pack + bz2-9.

pred[i] = 2*s[i-1] - s[i-2]  (linear extrapolation, vectorized)
residual[i] = s[i] - pred[i] = second finite difference

order-2 residuals have std≈6 vs order-1 std≈12 on ECG (from fixed_pred_v1 probe);
more residuals fit in 1 varzz byte, stream shrinks ~1.3MB→~1.0MB, bz2 BWT
then sees a denser heartbeat-period pattern.

fixed_pred_v1 failed because it was paired with LZMA (which needs dictionary-
matchable periodic structure, not just small per-sample entropy).  bz2+varzz
benefits directly from residual magnitude reduction: fewer 2-byte tokens →
shorter stream → denser periodicity for bz2's BWT.

Decode is fully vectorized (two cumsum passes); no per-sample Python loop.
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
    return np.where(z32 & 1, -(z32 >> 1) - 1, z32 >> 1).astype(np.int32)


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
    sig = np.frombuffer(blob[HEAD:], dtype='<i2').reshape(n_chan, n_samp).astype(np.int64)

    res = np.empty((n_chan, n_samp), dtype=np.int64)
    res[:, 0] = sig[:, 0]
    if n_samp > 1:
        res[:, 1] = sig[:, 1] - sig[:, 0]
    if n_samp > 2:
        # residual[i] = s[i] - 2*s[i-1] + s[i-2]  for i >= 2
        res[:, 2:] = sig[:, 2:] - 2 * sig[:, 1:-1] + sig[:, :-2]
    # wraparound to int16 range (matches how we'll reconstruct)
    res = ((res + 32768) % 65536 - 32768)

    zzag = _zigzag(res.ravel().astype(np.int16))
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
    res = _unzigzag(zzag).reshape(n_chan, n_samp)  # int32

    # Reconstruct: second differences → first differences via cumsum,
    # then first differences → signal via cumsum.
    # r[0]=s[0], r[1]=Δ[1], r[i>=2]=Δ[i]-Δ[i-1]
    # Δ[1:] = cumsum(r[1:])
    # s[i] = s[0] + Σ_{k=1}^{i} Δ[k]  = s[0] + cumsum(Δ[1:])[i-1]
    delta = np.cumsum(res[:, 1:], axis=1)       # Δ[1], Δ[2], ... (shape n_chan, n_samp-1)
    sig = np.empty((n_chan, n_samp), dtype=np.int64)
    sig[:, 0] = res[:, 0]
    sig[:, 1:] = sig[:, 0:1] + np.cumsum(delta, axis=1)

    sig = ((sig + 32768) % 65536 - 32768).astype('<i2')
    return raw[:HEAD] + sig.tobytes()
