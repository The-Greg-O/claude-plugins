"""
beat_tmpl_v1 — Onset-aligned beat template predictor.

Parent: rice_nlms_v1 (champion best-of-30).

Hypothesis: after NLMS, the remaining compressed structure is the REPEATED QRS SHAPE
across beats — bz2's BWT exploits this by grouping similar byte contexts. To remove it
at the signal level: detect QRS onsets (|diff|>threshold) and predict each QRS window
from the PREVIOUS QRS diff pattern aligned to onset. Template residuals = diff[onset+j]
- template[j] ≈ 0 for consistent-morphology records (RBBB: records 119,232; LBBB: 207).
Best-of-31 = champion 30 + 1 template option (5 pack sub-types tried internally).

Key difference from graveyard:
  - period_pred_v1: predicted s[i] from s[i-T] — wrong beat alignment, irregular timing
  - stage2_period_v1: applied OLS lag-T predictor to NLMS RESIDUALS
  This: uses raw first-differences (not residuals), onset-ALIGNED (not lag-T), onset
  positions stored explicitly (no causal detection issue at decode).

Losslessness: onset list stored explicitly → decoder applies same template logic →
template updated from decoded diff (= true diff since encoding is invertible mod 65536).

Channel data for flag 30 (template option):
  [uint16 n_onsets]
  [n_onsets × uint16 onset_deltas]  (delta from 0 for first, delta between successive)
  [uint8 sub_pk]  (0=varzz+bz2, 1=nibble+bz2, 2=Rice-k1, 3=Rice-k2, 4=Rice-k3)
  [compressed residual stream]
"""
import bz2
import struct

import numpy as np

MAGIC = b'ECG1'
HEAD = 10
_EPS = 1e-6

_NLMS_CONFIGS = [(2, 0.20), (4, 0.10), (8, 0.05), (16, 0.05), (32, 0.03)]
_RICE_KS = [1, 2, 3]

_TMPL_THRESHOLD = 20   # |diff| > this triggers QRS onset detection
_TMPL_W = 90           # QRS window width in samples (250ms at 360Hz)
_TMPL_REFRAC = 100     # min samples between consecutive onsets (280ms)


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


def _detect_onsets(diff_i32, threshold, refrac):
    """Detect QRS onset positions: first sample where |diff|>threshold after refractory."""
    n = len(diff_i32)
    onsets = []
    last = -refrac - 1
    for i in range(n):
        if abs(diff_i32[i]) > threshold and (i - last) >= refrac:
            onsets.append(i)
            last = i
    return onsets


def _tmpl_fwd(diff_ch, onsets, W):
    """Apply onset-aligned beat template predictor (encoder side)."""
    n = len(diff_ch)
    diff_i32 = diff_ch.astype(np.int32)
    res = diff_i32.copy()
    template = np.zeros(W, dtype=np.int32)
    template_valid = False

    for onset in onsets:
        end = min(onset + W, n)
        w = end - onset
        if template_valid:
            for j in range(w):
                pred = template[j]
                r = (diff_i32[onset + j] - pred + 32768) % 65536 - 32768
                res[onset + j] = r
        # update template with true diff values (decoder will recover same values)
        template[:w] = diff_i32[onset:end]
        if w < W:
            template[w:] = 0
        template_valid = True

    return res.astype(np.int16)


def _tmpl_inv(res_ch, onsets, n, W):
    """Invert onset-aligned template predictor (decoder side)."""
    res_i32 = res_ch.astype(np.int32)
    diff = res_i32.copy()
    template = np.zeros(W, dtype=np.int32)
    template_valid = False

    for onset in onsets:
        end = min(onset + W, n)
        w = end - onset
        if template_valid:
            for j in range(w):
                pred = template[j]
                d = (res_i32[onset + j] + pred + 32768) % 65536 - 32768
                diff[onset + j] = d
        # update template with decoded diff values
        template[:w] = diff[onset:end]
        if w < W:
            template[w:] = 0
        template_valid = True

    return diff.astype(np.int16)


def _pack_residuals(zzag):
    """Try all 5 pack types; return (best_size, sub_pk, best_data)."""
    opts = []
    c_vz = bz2.compress(_varzz_pack(zzag), 9)
    opts.append((len(c_vz), 0, c_vz))
    c_nb = bz2.compress(_nibble_pack(zzag), 9)
    opts.append((len(c_nb), 1, c_nb))
    for ri, k in enumerate(_RICE_KS):
        c_r = _rice_encode(zzag, k)
        opts.append((len(c_r), 2 + ri, c_r))
    return min(opts)


def _tmpl_encode_channel(diff_ch):
    """Encode with template predictor; return (size, flag=30, blob)."""
    onsets = _detect_onsets(diff_ch.astype(np.int32), _TMPL_THRESHOLD, _TMPL_REFRAC)
    res = _tmpl_fwd(diff_ch, onsets, _TMPL_W)
    zzag = _zigzag(res)

    # Pack onset list as uint16 deltas
    n_onsets = len(onsets)
    if n_onsets > 0:
        deltas = np.empty(n_onsets, dtype=np.uint16)
        deltas[0] = onsets[0]
        for i in range(1, n_onsets):
            deltas[i] = onsets[i] - onsets[i - 1]
    else:
        deltas = np.array([], dtype=np.uint16)

    onset_blob = struct.pack('<H', n_onsets) + deltas.tobytes()
    _, sub_pk, res_data = _pack_residuals(zzag)
    blob = onset_blob + struct.pack('<B', sub_pk) + res_data
    return len(blob), 30, blob


def _tmpl_decode_channel(data, n_samp):
    """Decode template predictor channel."""
    n_onsets = struct.unpack_from('<H', data, 0)[0]
    offset = 2
    if n_onsets > 0:
        deltas = np.frombuffer(data[offset:offset + n_onsets * 2], dtype=np.uint16)
        offset += n_onsets * 2
        onsets = np.empty(n_onsets, dtype=np.int32)
        onsets[0] = int(deltas[0])
        for i in range(1, n_onsets):
            onsets[i] = int(onsets[i - 1]) + int(deltas[i])
        onsets = list(onsets)
    else:
        offset += 0
        onsets = []

    sub_pk = data[offset]
    res_data = data[offset + 1:]

    if sub_pk == 0:
        zzag = _varzz_unpack(bz2.decompress(res_data), n_samp)
    elif sub_pk == 1:
        zzag = _nibble_unpack(bz2.decompress(res_data), n_samp)
    else:
        zzag = _rice_decode(res_data, n_samp, _RICE_KS[sub_pk - 2])

    res_ch = _unzigzag(zzag)
    return _tmpl_inv(res_ch, onsets, n_samp, _TMPL_W)


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

    d = sig.astype(np.int64)
    diff = np.empty_like(d)
    diff[:, 0] = d[:, 0]
    diff[:, 1:] = np.diff(d, axis=1)
    diff = ((diff + 32768) % 65536 - 32768).astype('<i2')

    chan_parts = []
    for c in range(n_chan):
        dc = diff[c]
        zzag = _zigzag(dc)
        options = _get_options(zzag, pred_idx=0)
        for cfg_idx, (order, mu) in enumerate(_NLMS_CONFIGS):
            res = _nlms_fwd(dc, order, mu)
            zz_r = _zigzag(res)
            options.extend(_get_options(zz_r, pred_idx=cfg_idx + 1))
        # template option
        options.append(_tmpl_encode_channel(dc))
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

        if flag == 30:
            diff_ch = _tmpl_decode_channel(data, n_samp)
        else:
            pred_idx = flag // 5
            pack_type = flag % 5

            if pack_type == 0:
                zzag = _varzz_unpack(bz2.decompress(data), n_samp)
            elif pack_type == 1:
                zzag = _nibble_unpack(bz2.decompress(data), n_samp)
            else:
                zzag = _rice_decode(data, n_samp, _RICE_KS[pack_type - 2])

            vals = _unzigzag(zzag)
            if pred_idx > 0:
                order, mu = _NLMS_CONFIGS[pred_idx - 1]
                diff_ch = _nlms_inv(vals, order, mu)
            else:
                diff_ch = vals
        channels.append(diff_ch)

    diff = np.stack(channels, axis=0).astype(np.int64)
    sig = np.cumsum(diff, axis=1)
    sig = ((sig + 32768) % 65536 - 32768).astype('<i2')
    return header + sig.tobytes()
