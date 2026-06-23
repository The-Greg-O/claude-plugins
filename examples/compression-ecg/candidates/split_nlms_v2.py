"""
split_nlms_v2 — Wider long-tap window for split-tap NLMS ECG compression.

Parent: split_nlms_v1 (best-of-45, 4.3133×).

Hypothesis: The champion's 3 split configs use 16 long taps centered at the
estimated beat period T (window T-8 to T+7). FFT-based period estimation has
±few-sample uncertainty, and beat-to-beat variability in slightly irregular
records means the 16-tap window misses QRS samples near the window edge. Widening
to 32 long taps (T-16 to T+15) provides more robust cross-beat QRS prediction:
covers period jitter of ±16 samples without extra per-sample compute (dot product
grows from 32→48 elements, a ~50% inner-loop increase per config). Adds 3 new
split configs: (16,32,0.03), (16,32,0.01), (8,32,0.03).

Best-of-60 = champion 30 + 30 split-NLMS (6 configs × 5 pack types).
Flags 0-29: champion (pred_idx=flag//5, pack_type=flag%5).
Flags 30-59: split-NLMS (split_idx=(flag-30)//5, pack_type=(flag-30)%5).
Channel data for flags 30-59: [uint16 beat_period][compressed_residuals].
"""
import bz2
import struct

import numpy as np

MAGIC = b'ECG1'
HEAD = 10
_EPS = 1e-6

_NLMS_CONFIGS = [(2, 0.20), (4, 0.10), (8, 0.05), (16, 0.05), (32, 0.03)]
_RICE_KS = [1, 2, 3]

# (short_order, long_order, mu)
_SPLIT_CONFIGS = [
    # Champion configs — 16-tap long window at T±8
    (16, 16, 0.03),
    (16, 16, 0.01),
    (8, 16, 0.03),
    # New — 32-tap long window at T±16 (wider QRS coverage, handles period jitter)
    (16, 32, 0.03),
    (16, 32, 0.01),
    (8, 32, 0.03),
]


def _zigzag(d):
    d32 = d.astype(np.int32)
    return np.where(d32 >= 0, d32 * 2, -d32 * 2 - 1).astype(np.uint32)


def _unzigzag(z):
    z32 = z.astype(np.int32)
    return np.where(z32 & 1, -(z32 >> 1) - 1, z32 >> 1).astype(np.int16)


def _varzz_pack(zzag):
    t0 = zzag < 128
    t1 = (zzag >= 128) & (zzag < 32640)
    t2 = zzag >= 32640
    n = len(zzag)
    sizes = np.ones(n, dtype=np.intp)
    sizes[t1] = 2
    sizes[t2] = 3
    off = np.empty(n + 1, dtype=np.intp)
    off[0] = 0
    np.cumsum(sizes, out=off[1:])
    out = np.zeros(int(off[-1]), dtype=np.uint8)
    if t0.any():
        out[off[:-1][t0]] = zzag[t0].astype(np.uint8)
    if t1.any():
        p = off[:-1][t1]
        v = zzag[t1].astype(np.uint32) - 128
        out[p] = (0x80 + (v >> 8)).astype(np.uint8)
        out[p + 1] = (v & 0xFF).astype(np.uint8)
    if t2.any():
        p = off[:-1][t2]
        v = zzag[t2]
        out[p] = np.uint8(0xFF)
        out[p + 1] = (v >> 8).astype(np.uint8)
        out[p + 2] = (v & 0xFF).astype(np.uint8)
    return bytes(out)


def _varzz_unpack(data, n):
    mv = memoryview(data)
    zzag = np.empty(n, dtype=np.uint32)
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


def _nibble_pack(zzag_arr):
    out = bytearray()
    n = len(zzag_arr)
    i = 0
    while i < n:
        z = int(zzag_arr[i])
        if i + 1 < n:
            z2 = int(zzag_arr[i + 1])
            if z < 8 and z2 < 8:
                out.append((z << 3) | z2)
                i += 2
                continue
        if z < 128:
            out.append(0x40 + z)
        elif z < 32640:
            v = z - 128
            out.append(0xC0 + (v >> 8))
            out.append(v & 0xFF)
        else:
            out.append(0xFF)
            out.append(z >> 8)
            out.append(z & 0xFF)
        i += 1
    return bytes(out)


def _nibble_unpack(data, n):
    mv = memoryview(data)
    zzag = np.empty(n, dtype=np.uint32)
    pos = 0
    i = 0
    while i < n:
        b = mv[pos]
        if b < 0x40:
            zzag[i] = (b >> 3) & 0x7
            i += 1
            if i < n:
                zzag[i] = b & 0x7
                i += 1
            pos += 1
        elif b < 0xC0:
            zzag[i] = b - 0x40
            i += 1
            pos += 1
        elif b < 0xFF:
            zzag[i] = 128 + (((b - 0xC0) << 8) | mv[pos + 1])
            i += 1
            pos += 2
        else:
            zzag[i] = (mv[pos + 1] << 8) | mv[pos + 2]
            i += 1
            pos += 3
    return zzag


def _rice_encode(zzag, k):
    n = len(zzag)
    zzag64 = zzag.astype(np.uint64)
    q = zzag64 >> k
    r = zzag64 & ((1 << k) - 1)
    total_bits = int(q.sum()) + n * (1 + k)
    total_bytes = (total_bits + 7) >> 3
    bits = np.zeros(total_bytes * 8, dtype=np.uint8)
    code_lens = q + (1 + k)
    start_offs = np.empty(n + 1, dtype=np.int64)
    start_offs[0] = 0
    np.cumsum(code_lens, out=start_offs[1:])
    stop_pos = (start_offs[:n] + q).astype(np.int64)
    bits[stop_pos] = 1
    for j in range(k):
        rem_pos = stop_pos + 1 + j
        bits[rem_pos] = ((r >> (k - 1 - j)) & 1).astype(np.uint8)
    return struct.pack('<I', total_bits) + np.packbits(bits[:total_bytes * 8]).tobytes()


def _rice_decode(data, n, k):
    bits_arr = np.unpackbits(np.frombuffer(data, dtype=np.uint8, offset=4))
    zzag = np.empty(n, dtype=np.uint32)
    pos = 0
    for i in range(n):
        q = 0
        while bits_arr[pos] == 0:
            q += 1
            pos += 1
        pos += 1
        r = 0
        for _ in range(k):
            r = (r << 1) | int(bits_arr[pos])
            pos += 1
        zzag[i] = (q << k) | r
    return zzag


def _nlms_fwd(diff_ch, order, mu):
    n = len(diff_ch)
    res = np.empty(n, dtype=np.int16)
    w = np.zeros(order, dtype=np.float64)
    x = np.zeros(order, dtype=np.float64)
    d_int = diff_ch.astype(np.int32)
    for i in range(n):
        pred = int(round(float(np.dot(w, x))))
        r = (int(d_int[i]) - pred + 32768) % 65536 - 32768
        res[i] = r
        norm = float(np.dot(x, x)) + _EPS
        w += (mu / norm) * float(r) * x
        x[1:] = x[:-1]
        x[0] = float(d_int[i])
    return res


def _nlms_inv(res_ch, order, mu):
    n = len(res_ch)
    diff = np.empty(n, dtype=np.int16)
    w = np.zeros(order, dtype=np.float64)
    x = np.zeros(order, dtype=np.float64)
    r_int = res_ch.astype(np.int32)
    for i in range(n):
        pred = int(round(float(np.dot(w, x))))
        d = (int(r_int[i]) + pred + 32768) % 65536 - 32768
        diff[i] = d
        norm = float(np.dot(x, x)) + _EPS
        w += (mu / norm) * float(r_int[i]) * x
        x[1:] = x[:-1]
        x[0] = float(d)
    return diff


def _estimate_period(diff_ch, lo=150, hi=650):
    """Estimate heartbeat period from autocorrelation of diff^2 via FFT."""
    n = len(diff_ch)
    hi_eff = min(hi, n // 3)
    if hi_eff <= lo:
        return 308  # fallback: 70 bpm at 360 Hz
    d2 = diff_ch.astype(np.float64) ** 2
    nfft = 1 << int(np.ceil(np.log2(n + hi_eff)))
    D = np.fft.rfft(d2, nfft)
    acorr = np.fft.irfft(D * np.conj(D))
    return lo + int(np.argmax(acorr[lo:hi_eff]))


def _split_fwd(diff_ch, short_order, long_start, long_order, mu):
    """Split-tap NLMS forward pass: short taps 1..short_order, long taps at beat period."""
    n = len(diff_ch)
    buf_size = long_start + long_order + short_order + 1
    total_order = short_order + long_order
    res = np.empty(n, dtype=np.int16)
    w = np.zeros(total_order, dtype=np.float64)
    hist = np.zeros(buf_size, dtype=np.float64)
    d_int = diff_ch.astype(np.int32)

    short_offsets = np.arange(1, short_order + 1, dtype=np.int64)
    long_offsets = np.arange(long_start, long_start + long_order, dtype=np.int64)
    all_offsets = np.concatenate([short_offsets, long_offsets])

    idxs = (-all_offsets) % buf_size
    pos = 0

    for i in range(n):
        x = hist[idxs]
        pred = int(round(float(np.dot(w, x))))
        r = (int(d_int[i]) - pred + 32768) % 65536 - 32768
        res[i] = r
        norm = float(np.dot(x, x)) + _EPS
        w += (mu / norm) * float(r) * x
        hist[pos] = float(d_int[i])
        pos += 1
        if pos == buf_size:
            pos = 0
        idxs += 1
        idxs[idxs == buf_size] = 0
    return res


def _split_inv(res_ch, short_order, long_start, long_order, mu):
    """Split-tap NLMS inverse: reconstructs diff from residuals."""
    n = len(res_ch)
    buf_size = long_start + long_order + short_order + 1
    total_order = short_order + long_order
    diff = np.empty(n, dtype=np.int16)
    w = np.zeros(total_order, dtype=np.float64)
    hist = np.zeros(buf_size, dtype=np.float64)
    r_int = res_ch.astype(np.int32)

    short_offsets = np.arange(1, short_order + 1, dtype=np.int64)
    long_offsets = np.arange(long_start, long_start + long_order, dtype=np.int64)
    all_offsets = np.concatenate([short_offsets, long_offsets])

    idxs = (-all_offsets) % buf_size
    pos = 0

    for i in range(n):
        x = hist[idxs]
        pred = int(round(float(np.dot(w, x))))
        d = (int(r_int[i]) + pred + 32768) % 65536 - 32768
        diff[i] = d
        norm = float(np.dot(x, x)) + _EPS
        w += (mu / norm) * float(r_int[i]) * x
        hist[pos] = float(d)
        pos += 1
        if pos == buf_size:
            pos = 0
        idxs += 1
        idxs[idxs == buf_size] = 0
    return diff


def _pack_compress(zzag, pack_type):
    if pack_type == 0:
        return bz2.compress(_varzz_pack(zzag), 9)
    elif pack_type == 1:
        return bz2.compress(_nibble_pack(zzag), 9)
    else:
        return _rice_encode(zzag, _RICE_KS[pack_type - 2])


def _pack_decompress(data, n, pack_type):
    if pack_type == 0:
        return _varzz_unpack(bz2.decompress(data), n)
    elif pack_type == 1:
        return _nibble_unpack(bz2.decompress(data), n)
    else:
        return _rice_decode(data, n, _RICE_KS[pack_type - 2])


def encode(blob):
    assert blob[:4] == MAGIC
    n_chan, n_samp = struct.unpack_from('<HI', blob, 4)
    sig = np.frombuffer(blob[HEAD:], dtype='<i2').reshape(n_chan, n_samp)

    d = sig.astype(np.int64)
    diff = np.empty_like(d)
    diff[:, 0] = d[:, 0]
    diff[:, 1:] = np.diff(d, axis=1)
    diff = ((diff + 32768) % 65536 - 32768).astype('<i2')

    chan_parts = []
    for c in range(n_chan):
        dc = diff[c]
        options = []

        # Champion options: flags 0-29
        zzag = _zigzag(dc)
        for pt in range(5):
            packed = _pack_compress(zzag, pt)
            options.append((len(packed), pt, packed))

        for cfg_idx, (order, mu) in enumerate(_NLMS_CONFIGS):
            res = _nlms_fwd(dc, order, mu)
            zz_r = _zigzag(res)
            flag_base = (cfg_idx + 1) * 5
            for pt in range(5):
                packed = _pack_compress(zz_r, pt)
                options.append((len(packed), flag_base + pt, packed))

        # Split-NLMS options: flags 30-59 (6 configs × 5 pack types)
        beat_period = _estimate_period(dc)
        period_hdr = struct.pack('<H', beat_period)

        for split_idx, (short_o, long_o, mu) in enumerate(_SPLIT_CONFIGS):
            long_start = max(short_o + 1, beat_period - long_o // 2)
            res = _split_fwd(dc, short_o, long_start, long_o, mu)
            zz_r = _zigzag(res)
            flag_base = 30 + split_idx * 5
            for pt in range(5):
                packed = period_hdr + _pack_compress(zz_r, pt)
                options.append((len(packed), flag_base + pt, packed))

        _, flag, data = min(options)
        chan_parts.append(struct.pack('<BI', flag, len(data)) + data)

    return blob[:HEAD] + b''.join(chan_parts)


def decode(blob):
    header = blob[:HEAD]
    assert header[:4] == MAGIC
    n_chan, n_samp = struct.unpack_from('<HI', header, 4)

    pos = HEAD
    channels = []
    for _ in range(n_chan):
        flag = blob[pos]
        clen = struct.unpack_from('<I', blob, pos + 1)[0]
        pos += 5
        data = blob[pos:pos + clen]
        pos += clen

        if flag >= 30:
            split_idx = (flag - 30) // 5
            pack_type = (flag - 30) % 5
            beat_period = struct.unpack_from('<H', data, 0)[0]
            short_o, long_o, mu = _SPLIT_CONFIGS[split_idx]
            long_start = max(short_o + 1, beat_period - long_o // 2)
            zzag = _pack_decompress(data[2:], n_samp, pack_type)
            vals = _unzigzag(zzag)
            diff_ch = _split_inv(vals, short_o, long_start, long_o, mu)
        else:
            pred_idx = flag // 5
            pack_type = flag % 5
            zzag = _pack_decompress(data, n_samp, pack_type)
            vals = _unzigzag(zzag)
            if pred_idx == 0:
                diff_ch = vals
            else:
                order, mu = _NLMS_CONFIGS[pred_idx - 1]
                diff_ch = _nlms_inv(vals, order, mu)

        channels.append(diff_ch)

    diff = np.stack(channels, axis=0).astype(np.int64)
    sig = np.cumsum(diff, axis=1)
    sig = ((sig + 32768) % 65536 - 32768).astype('<i2')
    return header + sig.tobytes()
