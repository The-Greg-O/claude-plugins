"""byte_split_v1: delta (channel-major) + byte-plane deinterlacing + LZMA.

After first-differencing all channels, the int16 LE bytes are deinterlaced
into two planes: lo (even bytes — actual ECG variation) and hi (odd bytes —
mostly 0x00/0xFF sign extension). The hi plane compresses to near-zero; the
lo plane keeps the ~385-sample heartbeat periodicity but at half the LZMA
match distance vs seed's interleaved layout.
"""
import struct
import lzma

import numpy as np

HEAD = 10  # b'ECG1' + uint16 n_chan + uint32 n_samp


def encode(blob: bytes) -> bytes:
    n_chan, n_samp = struct.unpack("<HI", blob[4:HEAD])
    sig = np.frombuffer(blob[HEAD:], dtype="<i2").reshape(n_chan, n_samp)

    # Channel-major first difference with int16 wraparound
    diff = np.empty((n_chan, n_samp), dtype=np.int64)
    diff[:, 0] = sig[:, 0]
    diff[:, 1:] = np.diff(sig.astype(np.int64), axis=1)
    diff = ((diff + 32768) % 65536 - 32768).astype("<i2")

    # Deinterlace: low bytes (variation) then high bytes (sign extension)
    raw = diff.tobytes()          # [lo0, hi0, lo1, hi1, ...] channel-major LE
    n_total = n_chan * n_samp
    plane = bytes(raw[0::2]) + bytes(raw[1::2])   # all lows, then all highs

    return blob[:HEAD] + lzma.compress(plane, preset=9)


def decode(comp: bytes) -> bytes:
    head = comp[:HEAD]
    n_chan, n_samp = struct.unpack("<HI", head[4:HEAD])
    n_total = n_chan * n_samp

    plane = lzma.decompress(comp[HEAD:])
    lo = plane[:n_total]
    hi = plane[n_total:]

    # Re-interlace bytes
    raw = bytearray(n_total * 2)
    raw[0::2] = lo
    raw[1::2] = hi

    diff = np.frombuffer(bytes(raw), dtype="<i2").reshape(n_chan, n_samp)
    sig = np.cumsum(diff.astype(np.int64), axis=1)
    sig = ((sig + 32768) % 65536 - 32768).astype("<i2")
    return head + sig.tobytes()
