"""
psb_nlms_v1 — phase-stable byte packing added to champion rice_nlms_v1.

Parent: rice_nlms_v1 (best-of-30: {raw, NLMS-2/4/8/16/32} x {varzz+bz2, nibble+bz2,
Rice-k1/2/3}).

Hypothesis (entropy-layer lever, not predictor): bz2's BWT wins on the residual
stream because it matches the repeating QRS shape across beats. But varzz/nibble use
2-3 byte escapes for the *large* values that occur AT the QRS peaks, so the number of
bytes consumed before each QRS varies beat-to-beat -> the same QRS morphology lands at
different byte phases -> BWT sees it as several distinct patterns (the documented
nibble-alignment lesson). A "phase-stable" packing fixes this: emit EXACTLY ONE byte
per sample (zzag clipped to 0..254), and route the rare zzag>=255 values to a side
stream via a single escape marker (255). The main stream is exactly n bytes with no
phase drift, so every QRS shape appears at a consistent byte phase and bz2's BWT can
cluster all beats together.

Added as pack_type 5 alongside the champion's 5 pack types. Best-of-36 = 6 predictor
configs x 6 pack options. Monotonically >= champion by construction (regression-proof).

Flag byte encoding:
  pred_idx  = flag // 6   (0=raw, 1..5 = NLMS configs 0..4)
  pack_type = flag % 6    (0=varzz+bz2, 1=nibble+bz2, 2=Rice-k1, 3=Rice-k2,
                           4=Rice-k3, 5=phase-stable+bz2)

Phase-stable payload: [uint32 n_exc][uint16 exc * n_exc][bz2(main_bytes)]
  main_bytes[i] = min(zzag[i], 255); positions == 255 are escapes whose true zzag
  (0..65535, fits uint16) is read in order from the exc array.
Header per channel: [uint8 flag][uint32_le clen][...data...]
"""
import bz2
import struct

import numpy as np

MAGIC = b'ECG1'
HEAD = 10
_EPS = 1e-6

_NLMS_CONFIGS = [(2, 0.20), (4, 0.10), (8, 0.05), (16, 0.05), (32, 0.03)]
_RICE_KS = [1, 2, 3]
_ESC = 255


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


def _psb_pack(zzag):
    """Phase-stable: 1 byte/sample (clipped to 0..254), rare zzag>=255 escaped to a
    uint16 side stream via marker 255. main bz2'd; exceptions stored raw."""
    main = np.minimum(zzag, _ESC).astype(np.uint8)
    exc_mask = zzag >= _ESC
    exc = zzag[exc_mask].astype('<u2')
    main_c = bz2.compress(main.tobytes(), 9)
    return struct.pack('<I', int(exc.shape[0])) + exc.tobytes() + main_c


def _psb_unpack(data, n):
    n_exc = struct.unpack_from('<I', data, 0)[0]
    off = 4
    exc = np.frombuffer(data, dtype='<u2', count=n_exc, offset=off)
    off += 2 * n_exc
    main = np.frombuffer(bz2.decompress(data[off:]), dtype=np.uint8)
    zzag = main.astype(np.uint32)
    if n_exc:
        idx = np.nonzero(main == _ESC)[0]
        zzag[idx] = exc.astype(np.uint32)
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


def _get_options(zzag, pred_idx):
    flag_base = pred_idx * 6
    c_vz = bz2.compress(_varzz_pack(zzag), 9)
    c_nb = bz2.compress(_nibble_pack(zzag), 9)
    c_psb = _psb_pack(zzag)
    opts = [
        (len(c_vz), flag_base + 0, c_vz),
        (len(c_nb), flag_base + 1, c_nb),
    ]
    for ri, k in enumerate(_RICE_KS):
        c_rice = _rice_encode(zzag, k)
        opts.append((len(c_rice), flag_base + 2 + ri, c_rice))
    opts.append((len(c_psb), flag_base + 5, c_psb))
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

        pred_idx = flag // 6
        pack_type = flag % 6

        if pack_type == 0:
            zzag = _varzz_unpack(bz2.decompress(data), n_samp)
        elif pack_type == 1:
            zzag = _nibble_unpack(bz2.decompress(data), n_samp)
        elif pack_type == 5:
            zzag = _psb_unpack(data, n_samp)
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
