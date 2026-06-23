"""
lpc_bz2_v1 -- per-segment Levinson-Durbin LPC predictor, new lineage.

Parent: rice_nlms_v1 (champion 4.3113x, best-of-30).
Lineage: lpc-bz2.

Hypothesis: block-wise LPC (Levinson-Durbin, MMSE-optimal per 512-sample
segment) applied to the diff stream. Unlike NLMS, coefficients are fitted
exactly to each block's autocorrelation -- no warm-up lag, no stochastic
approximation. Forward prediction is fully vectorized (fixed weights per
block => sliding-window matmul). Residuals may be more uniform, reducing
the periodic structure that keeps bz2 winning over Rice.

Best-of-45 = champion 30 options + 3 LPC orders x 5 pack types.
Strictly >= champion by construction.

Per-channel data layout for LPC options (flag // 5 in {6,7,8}):
  [uint16 n_segs][uint8 lpc_order][uint8 shift=12]
  [int16 x n_segs x lpc_order]   <- quantised AR coefficient table
  [packed residuals, optionally bz2-compressed]

Flag encoding:
  pred_idx = flag // 5   (0=raw, 1-5=NLMS configs, 6=LPC-4, 7=LPC-8, 8=LPC-16)
  pack_type = flag % 5   (0=varzz+bz2, 1=nibble+bz2, 2=rice-k1, 3=rice-k2, 4=rice-k3)
"""
import bz2
import struct

import numpy as np
from numpy.lib.stride_tricks import as_strided

MAGIC = b'ECG1'
HEAD = 10
SEG_SIZE = 512
LPC_SHIFT = 12
_EPS = 1e-6
_NLMS_CONFIGS = [(2, 0.20), (4, 0.10), (8, 0.05), (16, 0.05), (32, 0.03)]
_RICE_KS = [1, 2, 3]
_LPC_ORDERS = [4, 8, 16]


# ---------------------------------------------------------------------------
# Entropy coding helpers (identical to rice_nlms_v1)
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
# NLMS predictor (identical to rice_nlms_v1)
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
# LPC helpers
# ---------------------------------------------------------------------------

def _levinson(acf, order):
    """Levinson-Durbin: autocorrelation -> AR coefficients.

    acf: float64 array of length order+1, r[0..order].
    Returns: float64 array [a1, ..., ap] where x_hat[n] = sum(a[k]*x[n-k-1]).
    """
    if acf[0] < 1e-15:
        return np.zeros(order)
    a = np.zeros(order)
    e = acf[0]
    for m in range(order):
        k = -acf[m + 1]
        for j in range(m):
            k -= a[j] * acf[m - j]
        k /= e
        k = float(np.clip(k, -0.99999, 0.99999))
        tmp = a[:m].copy()
        a[m] = k
        for j in range(m):
            a[j] += k * tmp[m - 1 - j]
        e *= 1.0 - k * k
    return a


def _lpc_fwd(diff_ch, lpc_order):
    """Per-segment LPC forward pass on diff channel.

    Vectorized within each segment (fixed coefficients = sliding-window matmul).
    Returns: residuals (int16), coeff_table (int32, shape [n_segs, lpc_order]).
    """
    n = len(diff_ch)
    n_segs = (n + SEG_SIZE - 1) // SEG_SIZE
    coeff_table = np.zeros((n_segs, lpc_order), dtype=np.int32)
    residuals = np.empty(n, dtype=np.int16)

    dc_int = diff_ch.astype(np.int32)
    dc_f64 = diff_ch.astype(np.float64)

    # History: last lpc_order decoded diff values, oldest-first
    hist_old_first = np.zeros(lpc_order, dtype=np.int32)

    scale = 1 << LPC_SHIFT

    for seg_idx in range(n_segs):
        start = seg_idx * SEG_SIZE
        end = min(start + SEG_SIZE, n)
        B = end - start

        # Compute LPC from biased autocorrelation of this block
        blk_f = dc_f64[start:end]
        acf = np.empty(lpc_order + 1)
        acf[0] = float(np.dot(blk_f, blk_f)) / B
        for lag in range(1, lpc_order + 1):
            acf[lag] = float(np.dot(blk_f[:B - lag], blk_f[lag:])) / B

        a_float = _levinson(acf, lpc_order)
        a_int = np.round(a_float * scale).astype(np.int32)
        coeff_table[seg_idx] = a_int

        # Build extended context: [hist (oldest-first) | block]
        # extended[j] for j=0..lpc_order-1: history (oldest first)
        # extended[lpc_order+i]: dc_int[start+i]
        extended = np.empty(lpc_order + B, dtype=np.int64)
        extended[:lpc_order] = hist_old_first
        extended[lpc_order:] = dc_int[start:end]

        # Sliding window: windows[i] = extended[i:i+lpc_order], shape (B, lpc_order)
        # pred[i] = sum(a_int[k] * extended[i+lpc_order-1-k] for k=0..p-1)
        #         = a_int_flip . windows[i]  where a_int_flip = a_int[::-1]
        elem_stride = extended.strides[0]
        windows = as_strided(extended,
                             shape=(B, lpc_order),
                             strides=(elem_stride, elem_stride))

        a_flip = a_int[::-1].astype(np.int64)
        preds_raw = windows.astype(np.int64) @ a_flip  # (B,) int64
        preds = (preds_raw >> LPC_SHIFT).astype(np.int32)

        blk_int = dc_int[start:end]
        res_i32 = (blk_int - preds + 32768) % 65536 - 32768
        residuals[start:end] = res_i32.astype(np.int16)

        # Update history (oldest-first)
        if B >= lpc_order:
            hist_old_first = blk_int[B - lpc_order:].astype(np.int32)
        else:
            new_hist = np.empty(lpc_order, dtype=np.int32)
            new_hist[:lpc_order - B] = hist_old_first[B:]
            new_hist[lpc_order - B:] = blk_int
            hist_old_first = new_hist

    return residuals, coeff_table


def _lpc_inv(residuals, coeff_table, n_segs, lpc_order):
    """Per-segment LPC inverse pass. Sequential (each sample uses previous outputs)."""
    n = len(residuals)
    diff_ch = np.empty(n, dtype=np.int16)

    res_int = residuals.astype(np.int32)
    hist_old_first = np.zeros(lpc_order, dtype=np.int32)

    for seg_idx in range(n_segs):
        start = seg_idx * SEG_SIZE
        end = min(start + SEG_SIZE, n)
        B = end - start

        a_int = coeff_table[seg_idx].astype(np.int64)

        # Sequential synthesis (cannot vectorize: each output feeds next prediction)
        ctx = hist_old_first.astype(np.int64)  # oldest-first

        for i in range(B):
            pred_raw = int(np.dot(a_int, ctx[::-1]))
            pred = pred_raw >> LPC_SHIFT
            d = (int(res_int[start + i]) + pred + 32768) % 65536 - 32768
            diff_ch[start + i] = d

            # Shift context: push d as the new most-recent value
            ctx[:-1] = ctx[1:]
            ctx[-1] = d

        hist_old_first = ctx.astype(np.int32)

    return diff_ch


def _lpc_pack_options(residuals, coeff_table, lpc_order, n_segs, pred_idx):
    """Produce (size, flag, data) tuples for all 5 pack types for one LPC config."""
    n = len(residuals)
    # LPC header: n_segs(u16) + lpc_order(u8) + shift(u8) + coeffs(int16)
    hdr = struct.pack('<HBB', n_segs, lpc_order, LPC_SHIFT)
    hdr += coeff_table.astype(np.int16).tobytes()

    zzag = _zigzag(residuals)
    flag_base = pred_idx * 5
    opts = []

    c_vz = bz2.compress(_varzz_pack(zzag), 9)
    opts.append((len(hdr) + len(c_vz), flag_base + 0, hdr + c_vz))

    c_nb = bz2.compress(_nibble_pack(zzag), 9)
    opts.append((len(hdr) + len(c_nb), flag_base + 1, hdr + c_nb))

    for ri, k in enumerate(_RICE_KS):
        c_rice = _rice_encode(zzag, k)
        opts.append((len(hdr) + len(c_rice), flag_base + 2 + ri, hdr + c_rice))

    return opts


# ---------------------------------------------------------------------------
# Champion NLMS options helper (identical to rice_nlms_v1)
# ---------------------------------------------------------------------------

def _get_nlms_options(zzag, pred_idx):
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
        options = _get_nlms_options(zzag_raw, pred_idx=0)
        for cfg_idx, (order, mu) in enumerate(_NLMS_CONFIGS):
            res = _nlms_fwd(dc, order, mu)
            zz_r = _zigzag(res)
            options.extend(_get_nlms_options(zz_r, pred_idx=cfg_idx + 1))

        # New LPC options (pred_idx=6..8)
        for lpc_cfg_idx, lpc_order in enumerate(_LPC_ORDERS):
            lpc_res, coeff_table = _lpc_fwd(dc, lpc_order)
            n_segs = coeff_table.shape[0]
            options.extend(_lpc_pack_options(
                lpc_res, coeff_table, lpc_order, n_segs,
                pred_idx=6 + lpc_cfg_idx))

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

        if pred_idx >= 6:
            # LPC option: parse header then residuals
            lpc_order = data[2]  # byte at offset 2
            n_segs = struct.unpack_from('<H', data, 0)[0]
            # shift = data[3]  (always LPC_SHIFT=12)
            hdr_len = 4 + n_segs * lpc_order * 2
            coeff_raw = np.frombuffer(data[4:hdr_len], dtype=np.int16).reshape(n_segs, lpc_order)
            coeff_table = coeff_raw.astype(np.int32)
            payload = data[hdr_len:]

            if pack_type == 0:
                zzag = _varzz_unpack(bz2.decompress(payload), n_samp)
            elif pack_type == 1:
                zzag = _nibble_unpack(bz2.decompress(payload), n_samp)
            else:
                zzag = _rice_decode(payload, n_samp, _RICE_KS[pack_type - 2])

            residuals = _unzigzag(zzag)
            vals = _lpc_inv(residuals, coeff_table, n_segs, lpc_order)
        else:
            # NLMS or raw option
            if pack_type == 0:
                zzag = _varzz_unpack(bz2.decompress(data), n_samp)
            elif pack_type == 1:
                zzag = _nibble_unpack(bz2.decompress(data), n_samp)
            else:
                zzag = _rice_decode(data, n_samp, _RICE_KS[pack_type - 2])

            vals = _unzigzag(zzag)
            if pred_idx > 0:
                order, mu = _NLMS_CONFIGS[pred_idx - 1]
                vals = _nlms_inv(vals, order, mu)

        channels.append(vals)

    diff = np.stack(channels, axis=0).astype(np.int64)
    sig = np.cumsum(diff, axis=1)
    sig = ((sig + 32768) % 65536 - 32768).astype('<i2')
    return header + sig.tobytes()
