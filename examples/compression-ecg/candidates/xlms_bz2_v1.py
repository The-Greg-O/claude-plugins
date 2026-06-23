"""
xlms_bz2_v1 — cross-lead NLMS + bz2: revive dormant cross-lead lineage.

Parent nlms_bz2_v3 runs best-of-12 within-channel NLMS per channel.
This adds 4 cross-lead options for ch1 only: NLMS filters that incorporate
ch0's co-temporal delta as an extra tap alongside k within-channel taps.

Prior cross-lead attempt (simple linear model) yielded residual std=72 vs
within-channel std=4 — catastrophic. The key difference here: ch0_delta is
ONE EXTRA TAP in a joint (order+1)-tap NLMS filter that also has k
within-channel taps. The NLMS weight on the cross-lead tap converges to zero
if ch0 provides no information, and to the MMSE linear coefficient otherwise.
Regression-proof via best-of-N: cross-lead options are rejected if any
within-channel option wins.

Mechanism: during QRS onset in ch1 the within-channel NLMS context (recent
small baseline deltas) gives poor prediction. ch0_delta[i] is large at the
same sample (co-temporal QRS), providing the NLMS with a leading indicator
of the impending deflection. MLII and precordial leads often have opposite
QRS polarity; NLMS will learn a negative cross-lead weight naturally.

Flag scheme (per channel):
  0: raw varzz       1: raw nibble
  2k+2: NLMS[k] vz  2k+3: NLMS[k] nb   k in 0..4 (5 configs = _NLMS_CONFIGS)
  12: xlms[0] vz    13: xlms[0] nb      xlms(order=2, mu=0.20)
  14: xlms[1] vz    15: xlms[1] nb      xlms(order=4, mu=0.10)
  Flags 12-15 appear in ch1 only; ch0 always uses 0-11.

Losslessness: weight updates use the quantised residual (int), so encoder and
decoder NLMS states advance identically. ch0 is decoded before ch1, making
ch0's delta sequence available for ch1's cross-lead decode.
"""
import bz2
import struct

import numpy as np

MAGIC = b'ECG1'
HEAD = 10
_EPS = 1e-6

_NLMS_CONFIGS = [(2, 0.20), (4, 0.10), (8, 0.05), (16, 0.05), (32, 0.03)]
_XL_CONFIGS = [(2, 0.20), (4, 0.10)]


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


def _nlms_xl_fwd(ch0_diff, ch1_diff, order, mu):
    """Forward cross-lead NLMS: k within-ch1 taps + ch0_current delta as extra tap."""
    n = len(ch1_diff)
    res = np.empty(n, dtype=np.int16)
    w = np.zeros(order + 1, dtype=np.float64)
    x = np.zeros(order + 1, dtype=np.float64)
    d0 = ch0_diff.astype(np.int32)
    d1 = ch1_diff.astype(np.int32)
    for i in range(n):
        x[order] = float(d0[i])
        pred = int(round(float(np.dot(w, x))))
        r = (int(d1[i]) - pred + 32768) % 65536 - 32768
        res[i] = r
        norm = float(np.dot(x, x)) + _EPS
        w += (mu / norm) * float(r) * x
        x[1:order] = x[:order - 1]
        x[0] = float(d1[i])
    return res


def _nlms_xl_inv(ch0_diff, res_ch, order, mu):
    """Inverse cross-lead NLMS: decode ch1 deltas using ch0 deltas as context."""
    n = len(res_ch)
    diff = np.empty(n, dtype=np.int16)
    w = np.zeros(order + 1, dtype=np.float64)
    x = np.zeros(order + 1, dtype=np.float64)
    d0 = ch0_diff.astype(np.int32)
    r = res_ch.astype(np.int32)
    for i in range(n):
        x[order] = float(d0[i])
        pred = int(round(float(np.dot(w, x))))
        d = (int(r[i]) + pred + 32768) % 65536 - 32768
        diff[i] = d
        norm = float(np.dot(x, x)) + _EPS
        w += (mu / norm) * float(r[i]) * x
        x[1:order] = x[:order - 1]
        x[0] = float(d)
    return diff


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
    ch0_diff_arr = None
    for c in range(n_chan):
        dc = diff[c]
        zzag = _zigzag(dc)
        c_vz = bz2.compress(_varzz_pack(zzag), 9)
        c_nb = bz2.compress(_nibble_pack(zzag), 9)
        options = [
            (len(c_vz), 0, c_vz),
            (len(c_nb), 1, c_nb),
        ]
        for cfg_idx, (order, mu) in enumerate(_NLMS_CONFIGS):
            res = _nlms_fwd(dc, order, mu)
            zz_r = _zigzag(res)
            c_vz2 = bz2.compress(_varzz_pack(zz_r), 9)
            c_nb2 = bz2.compress(_nibble_pack(zz_r), 9)
            options.append((len(c_vz2), 2 + cfg_idx * 2, c_vz2))
            options.append((len(c_nb2), 3 + cfg_idx * 2, c_nb2))

        if c == 1 and ch0_diff_arr is not None:
            for xl_idx, (order, mu) in enumerate(_XL_CONFIGS):
                res = _nlms_xl_fwd(ch0_diff_arr, dc, order, mu)
                zz_r = _zigzag(res)
                c_vz2 = bz2.compress(_varzz_pack(zz_r), 9)
                c_nb2 = bz2.compress(_nibble_pack(zz_r), 9)
                options.append((len(c_vz2), 12 + xl_idx * 2, c_vz2))
                options.append((len(c_nb2), 13 + xl_idx * 2, c_nb2))

        _, flag, data = min(options)
        chan_parts.append(struct.pack('<BI', flag, len(data)) + data)
        if c == 0:
            ch0_diff_arr = dc

    return blob[:HEAD] + struct.pack('<I', n_chan) + b''.join(chan_parts)


def decode(blob):
    header = blob[:HEAD]
    assert header[:4] == MAGIC
    n_chan, n_samp = struct.unpack_from('<HI', header, 4)

    pos = HEAD + 4
    channels = []
    ch0_diff = None
    for ch_idx in range(n_chan):
        flag = blob[pos]
        clen = struct.unpack_from('<I', blob, pos + 1)[0]
        pos += 5
        data = bz2.decompress(blob[pos:pos + clen])
        pos += clen

        if flag < 12:
            if flag < 2:
                zzag = _varzz_unpack(data, n_samp) if flag == 0 else _nibble_unpack(data, n_samp)
                vals = _unzigzag(zzag)
            else:
                cfg_idx = (flag - 2) // 2
                is_nibble = (flag - 2) % 2 == 1
                order, mu = _NLMS_CONFIGS[cfg_idx]
                zzag = _nibble_unpack(data, n_samp) if is_nibble else _varzz_unpack(data, n_samp)
                vals = _unzigzag(zzag)
                vals = _nlms_inv(vals, order, mu)
        else:
            xl_idx = (flag - 12) // 2
            is_nibble = (flag - 12) % 2 == 1
            order, mu = _XL_CONFIGS[xl_idx]
            zzag = _nibble_unpack(data, n_samp) if is_nibble else _varzz_unpack(data, n_samp)
            vals = _unzigzag(zzag)
            vals = _nlms_xl_inv(ch0_diff, vals, order, mu)

        channels.append(vals)
        if ch_idx == 0:
            ch0_diff = vals

    diff = np.stack(channels, axis=0).astype(np.int64)
    sig = np.cumsum(diff, axis=1)
    sig = ((sig + 32768) % 65536 - 32768).astype('<i2')
    return header + sig.tobytes()
