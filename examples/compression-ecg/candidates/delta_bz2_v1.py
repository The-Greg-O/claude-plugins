"""delta_bz2_v1: per-channel first-difference (int16 wraparound) + bz2-9.

Identical to the seed codec except the backend is bz2 instead of LZMA.
bz2's BWT+RLE handles ECG periodicity better than LZMA's LZ77 sliding window
(raw bz2=3.85x vs raw xz=3.23x), and train/215 probe showed delta+bz2 > delta+lzma.
"""
import bz2
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
    return blob[:HEAD] + bz2.compress(diff.tobytes(), compresslevel=9)


def decode(comp):
    head = comp[:HEAD]
    n_chan, n_samp = struct.unpack("<HI", head[4:HEAD])
    diff = np.frombuffer(bz2.decompress(comp[HEAD:]), dtype="<i2").reshape(n_chan, n_samp)
    sig = np.cumsum(diff.astype(np.int64), axis=1)
    sig = ((sig + 32768) % 65536 - 32768).astype("<i2")
    return head + sig.tobytes()
