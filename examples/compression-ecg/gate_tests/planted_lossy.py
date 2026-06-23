"""Planted LOSSY candidate — used only to prove the gate rejects it.

Zeros the least-significant bit of every sample before compressing, so the
roundtrip cannot be byte-exact. evaluate.py must report this as a gate failure.
Not a real candidate; delete after verifying the gate.
"""
import lzma
import struct

import numpy as np

HEAD = 10


def encode(blob):
    n_chan, n_samp = struct.unpack("<HI", blob[4:HEAD])
    sig = np.frombuffer(blob[HEAD:HEAD + n_chan * n_samp * 2], dtype="<i2")
    lossy = (sig & ~np.int16(1))          # drop LSB — irreversible
    return blob[:HEAD] + lzma.compress(lossy.tobytes(), preset=1)


def decode(comp):
    return comp[:HEAD] + lzma.decompress(comp[HEAD:])
