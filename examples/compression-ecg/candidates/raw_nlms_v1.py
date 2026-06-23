"""
raw_nlms_v1 — NLMS prediction on the RAW SIGNAL (not first-differences).

Parent: rice_nlms_v1 (champion: best-of-30 diff-domain NLMS + bz2/Rice).

Hypothesis: diff-domain NLMS residuals still carry lag-1 autocorrelation=0.807
because smooth P/T-wave segments produce mildly correlated diffs. Raw-signal
AR(N) NLMS can predict pure sinusoidal segments exactly (s[i] = 2cos(w)s[i-1]
- s[i-2] for a sine), concentrating residuals only at QRS spikes (~15% of signal).
Smooth baseline + P/T waves (~85%) get near-zero residuals, creating an
impulsive distribution that may beat bz2's BWT with Rice coding.

Best-of-50 = champion 30 (diff-domain) + 20 new (4 raw-domain configs x 5 packs).
Regression-proof by construction.

Flag encoding:
  pred_idx = flag // 5
    0     = raw diff (no NLMS)
    1-5   = diff-domain NLMS _NLMS_CONFIGS[0-4]
    6-9   = raw-signal NLMS _RAW_NLMS_CONFIGS[0-3]
  pack_type = flag % 5
    0=varzz+bz2, 1=nibble+bz2, 2=Rice-k1, 3=Rice-k2, 4=Rice-k3

For diff-domain channels: stored values are diffs; decode applies cumsum.
For raw-domain channels: stored values are raw residuals; decode inverts NLMS,
no cumsum needed.
"""
import bz2
import struct

import numpy as np

MAGIC = b'ECG1'
HEAD = 10
_EPS = 1e-6

_NLMS_CONFIGS = [(2, 0.20), (4, 0.10), (8, 0.05), (16, 0.05), (32, 0.03)]
_RAW_NLMS_CONFIGS = [(2, 0.20), (4, 0.10), (8, 0.05), (16, 0.05)]
_RICE_KS = [1, 2, 3]

# pred_idx 0-5 = diff-domain; pred_idx 6+ = raw-domain (flag bit 5 set via >= 6)
_RAW_PRED_OFFSET = 6


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
    """Vectorized Rice-k encoder. Stores 4-byte total_bits prefix + packed bits."""
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
    """Decode Rice-k stream (data includes 4-byte total_bits prefix)."""
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


def _nlms_diff_fwd(diff_ch, order, mu):
    """Champion NLMS on first-differences."""
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


def _nlms_diff_inv(res_ch, order, mu):
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


def _nlms_raw_fwd(sig_ch, order, mu):
    """NLMS on raw signal values. For smooth segments s[i]≈2cos(w)s[i-1]-s[i-2]
    is satisfied by AR(2), giving near-zero residuals; QRS residuals are similar
    to diff-domain. Weight update uses quantized residual for encoder/decoder parity."""
    n = len(sig_ch)
    res = np.empty(n, dtype=np.int16)
    w = np.zeros(order, dtype=np.float64)
    x = np.zeros(order, dtype=np.float64)
    s_int = sig_ch.astype(np.int32)
    for i in range(n):
        pred = int(round(float(np.dot(w, x))))
        r = (int(s_int[i]) - pred + 32768) % 65536 - 32768
        res[i] = r
        norm = float(np.dot(x, x)) + _EPS
        w += (mu / norm) * float(r) * x
        x[1:] = x[:-1]
        x[0] = float(s_int[i])
    return res


def _nlms_raw_inv(res_ch, order, mu):
    """Invert raw-signal NLMS to recover original signal samples."""
    n = len(res_ch)
    sig = np.empty(n, dtype=np.int16)
    w = np.zeros(order, dtype=np.float64)
    x = np.zeros(order, dtype=np.float64)
    r_int = res_ch.astype(np.int32)
    for i in range(n):
        pred = int(round(float(np.dot(w, x))))
        s = (int(r_int[i]) + pred + 32768) % 65536 - 32768
        sig[i] = s
        norm = float(np.dot(x, x)) + _EPS
        w += (mu / norm) * float(r_int[i]) * x
        x[1:] = x[:-1]
        x[0] = float(s)
    return sig


def _get_options(zzag, pred_idx):
    flag_base = pred_idx * 5
    c_vz = bz2.compress(_varzz_pack(zzag), 9)
    c_nb = bz2.compress(_nibble_pack(zzag), 9)
    opts = [
        (len(c_vz), flag_base + 0, c_vz),
        (len(c_nb), flag_base + 1, c_nb),
    ]
    for ri, k in enumerate(_RICE_KS):
        c_rice = _rice_encode(zzag, k)
        opts.append((len(c_rice), flag_base + 2 + ri, c_rice))
    return opts


def encode(blob):
    assert blob[:4] == MAGIC
    n_chan, n_samp = struct.unpack_from('<HI', blob, 4)
    sig = np.frombuffer(blob[HEAD:], dtype='<i2').reshape(n_chan, n_samp)

    # First-differences for diff-domain options
    d = sig.astype(np.int64)
    diff = np.empty_like(d)
    diff[:, 0] = d[:, 0]
    diff[:, 1:] = np.diff(d, axis=1)
    diff = ((diff + 32768) % 65536 - 32768).astype('<i2')

    chan_parts = []
    for c in range(n_chan):
        dc = diff[c]
        sc = sig[c]

        # Champion diff-domain options (pred_idx 0-5, flags 0-29)
        zzag = _zigzag(dc)
        options = _get_options(zzag, pred_idx=0)
        for cfg_idx, (order, mu) in enumerate(_NLMS_CONFIGS):
            res = _nlms_diff_fwd(dc, order, mu)
            zz_r = _zigzag(res)
            options.extend(_get_options(zz_r, pred_idx=cfg_idx + 1))

        # New raw-signal NLMS options (pred_idx 6-9, flags 30-49)
        for cfg_idx, (order, mu) in enumerate(_RAW_NLMS_CONFIGS):
            res_raw = _nlms_raw_fwd(sc, order, mu)
            zz_raw = _zigzag(res_raw)
            options.extend(_get_options(zz_raw, pred_idx=_RAW_PRED_OFFSET + cfg_idx))

        _, flag, data = min(options)
        chan_parts.append(struct.pack('<BI', flag, len(data)) + data)

    return blob[:HEAD] + struct.pack('<I', n_chan) + b''.join(chan_parts)


def decode(blob):
    header = blob[:HEAD]
    assert header[:4] == MAGIC
    n_chan, n_samp = struct.unpack_from('<HI', header, 4)

    pos = HEAD + 4
    # Collect (channel_data, is_diff) tuples
    chan_info = []
    for _ in range(n_chan):
        flag = blob[pos]
        clen = struct.unpack_from('<I', blob, pos + 1)[0]
        pos += 5
        data = blob[pos:pos + clen]
        pos += clen

        pred_idx = flag // 5
        pack_type = flag % 5

        if pack_type == 0:
            zzag = _varzz_unpack(bz2.decompress(data), n_samp)
        elif pack_type == 1:
            zzag = _nibble_unpack(bz2.decompress(data), n_samp)
        else:
            zzag = _rice_decode(data, n_samp, _RICE_KS[pack_type - 2])

        vals = _unzigzag(zzag)

        if pred_idx == 0:
            chan_info.append((vals, True))  # diff, needs cumsum
        elif pred_idx <= 5:
            order, mu = _NLMS_CONFIGS[pred_idx - 1]
            diff_ch = _nlms_diff_inv(vals, order, mu)
            chan_info.append((diff_ch, True))  # diff, needs cumsum
        else:
            # Raw-signal NLMS: invert to get raw signal directly
            order, mu = _RAW_NLMS_CONFIGS[pred_idx - _RAW_PRED_OFFSET]
            raw_ch = _nlms_raw_inv(vals, order, mu)
            chan_info.append((raw_ch, False))  # raw, no cumsum

    result = []
    for vals, is_diff in chan_info:
        if is_diff:
            v64 = vals.astype(np.int64)
            raw = np.cumsum(v64)
            raw = ((raw + 32768) % 65536 - 32768).astype('<i2')
        else:
            raw = vals
        result.append(raw)

    sig = np.stack(result, axis=0)
    return header + sig.tobytes()
