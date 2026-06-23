"""Seed codec: per-channel first-difference (int16 wraparound) + LZMA.

A deliberately simple, lossless starting point. It removes the obvious
sample-to-sample correlation with a delta, then lets LZMA do the entropy coding.
On this data it roughly TIES bzip2-9 — the loop's job is to clearly beat it by
discovering a real predictor + residual entropy coder.
"""
import lzma
import struct

import numpy as np

HEAD = 10  # canonical header: b'ECG1' + uint16 n_chan + uint32 n_samp


def _shape(blob):
    n_chan, n_samp = struct.unpack("<HI", blob[4:HEAD])
    sig = np.frombuffer(blob[HEAD:HEAD + n_chan * n_samp * 2], dtype="<i2")
    return n_chan, n_samp, sig.reshape(n_chan, n_samp)


def encode(blob):
    n_chan, n_samp, sig = _shape(blob)
    d = sig.astype(np.int64)
    diff = np.empty_like(d)
    diff[:, 0] = d[:, 0]
    diff[:, 1:] = np.diff(d, axis=1)
    diff = ((diff + 32768) % 65536 - 32768).astype("<i2")
    return blob[:HEAD] + lzma.compress(diff.tobytes(), preset=6)


def decode(comp):
    head = comp[:HEAD]
    n_chan, n_samp = struct.unpack("<HI", head[4:HEAD])
    diff = np.frombuffer(lzma.decompress(comp[HEAD:]), dtype="<i2").reshape(n_chan, n_samp)
    sig = np.cumsum(diff.astype(np.int64), axis=1)
    sig = ((sig + 32768) % 65536 - 32768).astype("<i2")
    return head + sig.tobytes()
