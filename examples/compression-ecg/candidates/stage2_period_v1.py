"""
stage2_period_v1 -- two-stage codec: NLMS then period predictor on residuals.

Parent: rice_nlms_v1 (champion 4.3113x, best-of-30).
Lineage: stage2-period.

Hypothesis: after champion NLMS, residuals still have inter-beat periodic
structure (QRS prediction-error bursts at heartbeat period T). A one-tap
period predictor r2[n] = r1[n] - w*r1[n-T] eliminates this via OLS weight
w estimated from residual autocorrelation. Unlike the failed period_pred_v1
(raw signal, 2.89x), stage-2 operates on ALREADY-SMALL NLMS residuals:
when T is slightly wrong (irregular rhythms), the prediction error is still
small ("~0 predicts ~0"). Encode and decode are fully vectorized in stride-T
chunks -- no Python per-sample loop overhead for stage 2.

Best-of-60 = 6 pred configs x 2 (with/without period stage) x 5 pack types.
Champion's 30 options are a subset, so strictly >= champion by construction.

Per-channel flag encoding:
  pred_idx = flag // 5   (0=raw, 1-5=NLMS configs, 6-11=same with period stage)
  pack_type = flag % 5   (0=varzz+bz2, 1=nibble+bz2, 2-4=rice k=1/2/3)

For period-stage options (pred_idx >= 6), the data starts with:
  [uint16 T][int16 w_int][uint8 shift=12]   (5 bytes header)
  then the packed residuals (same format as non-period options)
"""
import bz2
import struct

import numpy as np

MAGIC = b'ECG1'
HEAD = 10
_EPS = 1e-6
_NLMS_CONFIGS = [(2, 0.20), (4, 0.10), (8, 0.05), (16, 0.05), (32, 0.03)]
_RICE_KS = [1, 2, 3]
_PERIOD_SHIFT = 12
_T_MIN = 300
_T_MAX = 500


# ---------------------------------------------------------------------------
# Entropy coding helpers (identical to champion)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# NLMS predictor (identical to champion)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Period predictor (stage 2) -- fully vectorized
# ---------------------------------------------------------------------------

def _estimate_period(r1, t_min=_T_MIN, t_max=_T_MAX):
    """Estimate dominant heartbeat period from residual autocorrelation."""
    r1f = r1.astype(np.float64)
    n = len(r1f)
    r0 = float(np.dot(r1f, r1f))
    if r0 < 1e-10:
        return (t_min + t_max) // 2, 0.0
    best_t = t_min
    best_corr = -1e18
    for t in range(t_min, t_max + 1):
        c = float(np.dot(r1f[:n - t], r1f[t:])) / r0
        if c > best_corr:
            best_corr = c
            best_t = t
    # OLS weight (biased toward zero via regularisation)
    n2 = n - best_t
    num = float(np.dot(r1f[:n2], r1f[best_t:]))
    den = float(np.dot(r1f[:n2], r1f[:n2])) + 1e-10
    w_ols = num / den
    # Clip weight to avoid amplification
    w_ols = float(np.clip(w_ols, -0.8, 0.8))
    return best_t, w_ols


def _period_fwd(r1, T, w_int):
    """Vectorized period prediction: r2[n] = r1[n] - (w_int * r1[n-T]) >> shift."""
    n = len(r1)
    r1_i32 = r1.astype(np.int32)
    r2 = np.empty(n, dtype=np.int16)
    r2[:T] = r1[:T]
    preds_raw = (w_int * r1_i32[:n - T].astype(np.int64)) >> _PERIOD_SHIFT
    tail = (r1_i32[T:] - preds_raw.astype(np.int32) + 32768) % 65536 - 32768
    r2[T:] = tail.astype(np.int16)
    return r2


def _period_inv(r2, T, w_int):
    """Vectorized period inverse in stride-T chunks.

    r1[n] = r2[n] + (w_int * r1[n-T]) >> shift  for n >= T
    Each stride-T chunk depends only on the previous chunk, enabling
    vectorized processing without a per-sample Python loop.
    """
    n = len(r2)
    r1 = np.empty(n, dtype=np.int16)
    r2_i32 = r2.astype(np.int32)

    # First T samples: no prediction
    r1[:T] = r2[:T]

    # Remaining in chunks of T
    for start in range(T, n, T):
        end = min(start + T, n)
        B = end - start
        ctx = r1[start - T:start - T + B].astype(np.int64)
        preds = (w_int * ctx) >> _PERIOD_SHIFT
        chunk = (r2_i32[start:end] + preds.astype(np.int32) + 32768) % 65536 - 32768
        r1[start:end] = chunk.astype(np.int16)

    return r1


# ---------------------------------------------------------------------------
# Option builder helpers
# ---------------------------------------------------------------------------

def _pack_options(zzag, flag_base):
    """Generate (size, flag, data) tuples for all 5 pack types."""
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


def _period_pack_options(zzag, T, w_int, flag_base):
    """Pack options for period-stage residuals; data includes 5-byte period header."""
    hdr = struct.pack('<HhB', T, w_int, _PERIOD_SHIFT)  # T(u16) + w(i16) + shift(u8)
    c_vz = bz2.compress(_varzz_pack(zzag), 9)
    c_nb = bz2.compress(_nibble_pack(zzag), 9)
    opts = [
        (5 + len(c_vz), flag_base + 0, hdr + c_vz),
        (5 + len(c_nb), flag_base + 1, hdr + c_nb),
    ]
    for ri, k in enumerate(_RICE_KS):
        c_rice = _rice_encode(zzag, k)
        opts.append((5 + len(c_rice), flag_base + 2 + ri, hdr + c_rice))
    return opts


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------

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
        zzag_raw = _zigzag(dc)

        # Champion options: raw (pred_idx=0) + 5 NLMS configs (pred_idx=1..5)
        # Cache NLMS residuals to avoid double computation for period-stage options.
        options = _pack_options(zzag_raw, flag_base=0)
        nlms_residuals = []
        for cfg_idx, (order, mu) in enumerate(_NLMS_CONFIGS):
            r1 = _nlms_fwd(dc, order, mu)
            nlms_residuals.append(r1)
            options.extend(_pack_options(_zigzag(r1), flag_base=(cfg_idx + 1) * 5))

        # Period-stage options (pred_idx=6..11 = raw+period, nlms0+period, ..., nlms4+period)
        # raw + period
        T, w_ols = _estimate_period(dc)
        w_int = int(round(w_ols * (1 << _PERIOD_SHIFT)))
        r2_raw = _period_fwd(dc, T, w_int)
        options.extend(_period_pack_options(_zigzag(r2_raw), T, w_int, flag_base=6 * 5))

        # NLMS + period (reuse cached NLMS residuals)
        for cfg_idx in range(len(_NLMS_CONFIGS)):
            r1 = nlms_residuals[cfg_idx]
            T, w_ols = _estimate_period(r1)
            w_int = int(round(w_ols * (1 << _PERIOD_SHIFT)))
            r2 = _period_fwd(r1, T, w_int)
            options.extend(_period_pack_options(
                _zigzag(r2), T, w_int, flag_base=(7 + cfg_idx) * 5))

        _, flag, data = min(options)
        chan_parts.append(struct.pack('<BI', flag, len(data)) + data)

    return blob[:HEAD] + struct.pack('<I', n_chan) + b''.join(chan_parts)


def decode(blob):
    header = blob[:HEAD]
    assert header[:4] == MAGIC
    n_chan, n_samp = struct.unpack_from('<HI', header, 4)

    pos = HEAD + 4
    channels = []
    for _ in range(n_chan):
        flag = blob[pos]
        clen = struct.unpack_from('<I', blob, pos + 1)[0]
        pos += 5
        data = blob[pos:pos + clen]
        pos += clen

        pred_idx = flag // 5
        pack_type = flag % 5

        has_period = pred_idx >= 6

        if has_period:
            # Parse period header
            T, w_int, _shift = struct.unpack_from('<HhB', data, 0)
            payload = data[5:]
            nlms_pred_idx = pred_idx - 6  # 0=raw, 1..5=NLMS configs
        else:
            payload = data
            nlms_pred_idx = pred_idx

        # Decode residuals
        if pack_type == 0:
            zzag = _varzz_unpack(bz2.decompress(payload), n_samp)
        elif pack_type == 1:
            zzag = _nibble_unpack(bz2.decompress(payload), n_samp)
        else:
            zzag = _rice_decode(payload, n_samp, _RICE_KS[pack_type - 2])

        vals = _unzigzag(zzag)

        # Invert period stage
        if has_period:
            vals = _period_inv(vals, T, w_int)

        # Invert NLMS stage
        if nlms_pred_idx > 0:
            order, mu = _NLMS_CONFIGS[nlms_pred_idx - 1]
            vals = _nlms_inv(vals, order, mu)

        channels.append(vals)

    diff = np.stack(channels, axis=0).astype(np.int64)
    sig = np.cumsum(diff, axis=1)
    sig = ((sig + 32768) % 65536 - 32768).astype('<i2')
    return header + sig.tobytes()
