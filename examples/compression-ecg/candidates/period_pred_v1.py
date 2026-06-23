"""
Period-lag predictor: predict s[i] from s[i-T] where T = heartbeat period.
Beat-to-beat residuals are tiny (~std 1-5) vs first-diff (~std 12), AND their
variation is itself periodic at the beat rate -> LZMA wins on both axes.
"""
import lzma
import struct

import numpy as np

MAGIC = b'ECG1'
HEAD = 10          # b'ECG1' + uint16 n_chan + uint32 n_samp
_LO = 150          # min period samples (~144 BPM at 360 Hz)
_HI = 720          # max period samples (~30 BPM at 360 Hz)


def _beat_period(ch: np.ndarray) -> int:
    """Estimate dominant heartbeat period via autocorrelation peak search."""
    n = len(ch)
    x = ch.astype(np.float32)
    x -= x.mean()
    fft = np.fft.rfft(x, n=n)
    ac = np.fft.irfft((fft * fft.conj()).real, n=n).real
    lo = min(_LO, n // 4)
    hi = min(_HI, n // 2)
    if lo >= hi:
        return lo
    return lo + int(np.argmax(ac[lo:hi]))


def encode(blob: bytes) -> bytes:
    assert blob[:4] == MAGIC
    n_chan, n_samp = struct.unpack_from('<HI', blob, 4)
    sig = np.frombuffer(blob[HEAD:], dtype='<i2').reshape(n_chan, n_samp)

    periods = []
    parts = []
    for ch_data in sig:
        p = _beat_period(ch_data)
        periods.append(p)

        res = np.empty(n_samp, dtype=np.int16)
        # First p samples: first-difference fallback (same as seed)
        res[0] = ch_data[0]
        if p > 1 and n_samp > 1:
            end = min(p, n_samp)
            res[1:end] = (ch_data[1:end].astype(np.int32) -
                          ch_data[:end - 1].astype(np.int32)).astype(np.int16)
        # Remainder: period-lag residuals (beat-to-beat differences)
        if n_samp > p:
            res[p:] = (ch_data[p:].astype(np.int32) -
                       ch_data[:n_samp - p].astype(np.int32)).astype(np.int16)
        parts.append(res.tobytes())

    meta = struct.pack(f'<{n_chan}H', *periods)
    payload = blob[:HEAD] + meta + b''.join(parts)
    return lzma.compress(payload, preset=9)


def decode(blob: bytes) -> bytes:
    raw = lzma.decompress(blob)
    n_chan, n_samp = struct.unpack_from('<HI', raw, 4)
    periods = struct.unpack_from(f'<{n_chan}H', raw, HEAD)
    ch_bytes = n_samp * 2
    offset = HEAD + n_chan * 2

    channels = []
    for ch in range(n_chan):
        res = np.frombuffer(raw[offset:offset + ch_bytes], dtype='<i2').astype(np.int32)
        offset += ch_bytes
        p = int(periods[ch])
        s = np.empty(n_samp, dtype=np.int16)

        # Decode first min(p, n_samp) samples via cumsum (= sequential int16 add)
        lim = min(p, n_samp)
        s[:lim] = np.cumsum(res[:lim]).astype(np.int16)

        # Decode remainder: vectorized period-by-period (O(N/p) Python iterations)
        for start in range(p, n_samp, p):
            end = min(start + p, n_samp)
            s[start:end] = (s[start - p:end - p].astype(np.int32) +
                            res[start:end]).astype(np.int16)

        channels.append(s)

    result = np.stack(channels, axis=0)
    return raw[:HEAD] + result.astype('<i2').tobytes()
