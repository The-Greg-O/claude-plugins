"""
nibble4_nlms_v1 — 4-bit fixed-width nibble packing as additional encoding option.

Parent: rice_nlms_v1 (best-of-30: 6 pred configs x 5 pack types).

Hypothesis: champion nibble uses variable-length tokens — when zzag >= 8, a pair breaks
and the sample takes a full byte. This shifts the byte boundary at QRS peaks, so each beat's
QRS lands at a slightly different byte offset. bz2 BWT sees QRS as different contexts
beat-to-beat and cannot fully exploit the periodicity.

4-bit fixed: every sample occupies exactly one nibble (4 bits), two per byte, regardless of
residual magnitude. For zzag in 0-14: encode directly. For zzag >= 15: encode 0xF (escape)
in main stream; actual value stored in a side overflow stream.

After NLMS, ~94-97% of residuals have zzag < 15:
  Main stream:  N/2 bytes EXACTLY, heartbeat period = 192.5 bytes (constant, no phase drift)
  Overflow:     ~3-6% x N values ~ 20-40KB (bz2'd separately)
  Main stream is shorter than nibble (~325KB vs ~341KB) AND phase-consistent.

Best-of-36 = champion 30 (flags 0-29, regression-proof) + 6 nibble4 options (flags 30-35).
Flag encoding:
  flags 0-29: pred_idx = flag // 5, pack_type = flag % 5 (champion layout, unchanged)
  flags 30-35: nibble4 + pred_idx = flag - 30
Channel data for nibble4 (flags 30-35):
  [uint32_le main_bz2_len][bz2(main4)][bz2(overflow_varzz)]
  where main4 = ceil(N/2) bytes, each byte = (zzag_lo[2i]<<4)|zzag_lo[2i+1],
  zzag_lo = min(zzag, 15); overflow_varzz = varzz of actual zzag for escaped positions.
"""
import bz2
import struct

import numpy as np

MAGIC = b'ECG1'
HEAD = 10
_EPS = 1e-6

_NLMS_CONFIGS = [(2, 0.20), (4, 0.10), (8, 0.05), (16, 0.05), (32, 0.03)]
_RICE_KS = [1, 2, 3]


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


def _nibble4_pack(zzag):
    """Pack zzag into 4-bit nibbles (2 per byte). zzag>=15 becomes 0xF (escape)."""
    n = len(zzag)
    clamped = np.minimum(zzag, np.uint32(15)).astype(np.uint8)
    n_bytes = (n + 1) // 2
    main = np.zeros(n_bytes, dtype=np.uint8)
    # even samples in high nibble, odd samples in low nibble
    main[:] = clamped[0::2][:n_bytes] << 4
    odd_n = n // 2
    if odd_n > 0:
        main[:odd_n] |= clamped[1::2][:odd_n]
    # overflow: actual zzag for escaped positions (zzag >= 15)
    overflow = zzag[zzag >= 15].astype(np.uint32)
    return bytes(main), overflow


def _nibble4_unpack(main_bytes, overflow, n):
    """Unpack 4-bit nibble stream. overflow provides actual values for 0xF escapes."""
    main = np.frombuffer(main_bytes, dtype=np.uint8)
    zzag = np.empty(n, dtype=np.uint32)
    n_bytes = (n + 1) // 2
    hi = main[:n_bytes].astype(np.uint32) >> 4   # even samples
    lo = main[:n_bytes].astype(np.uint32) & 0xF  # odd samples
    zzag[0::2] = hi
    odd_n = n // 2
    if odd_n > 0:
        zzag[1::2] = lo[:odd_n]
    # fill escapes from overflow in order
    if len(overflow) > 0:
        esc_pos = np.where(zzag[:n] == 15)[0]
        zzag[esc_pos] = overflow[:len(esc_pos)]
    return zzag


def _nibble4_encode(zzag):
    """Encode zzag as nibble4 channel blob: [uint32 main_bz2_len][bz2(main)][bz2(overflow)]."""
    main_bytes, overflow = _nibble4_pack(zzag)
    c_main = bz2.compress(main_bytes, 9)
    if len(overflow) > 0:
        c_ov = bz2.compress(_varzz_pack(overflow), 9)
    else:
        c_ov = b''
    return struct.pack('<I', len(c_main)) + c_main + c_ov


def _nibble4_decode(data, n):
    """Decode nibble4 channel blob."""
    main_bz2_len = struct.unpack_from('<I', data, 0)[0]
    c_main = data[4:4 + main_bz2_len]
    c_ov = data[4 + main_bz2_len:]
    main_bytes = bz2.decompress(c_main)
    if len(c_ov) > 0:
        # count escapes to know overflow length
        main_arr = np.frombuffer(main_bytes, dtype=np.uint8)
        n_bytes = (n + 1) // 2
        hi = (main_arr[:n_bytes] >> 4).astype(np.uint8)
        lo = (main_arr[:n_bytes] & 0xF).astype(np.uint8)
        nibbles = np.empty(n, dtype=np.uint8)
        nibbles[0::2] = hi
        odd_n = n // 2
        if odd_n > 0:
            nibbles[1::2] = lo[:odd_n]
        n_esc = int((nibbles == 15).sum())
        ov_bytes = bz2.decompress(c_ov)
        overflow = _varzz_unpack(ov_bytes, n_esc)
    else:
        overflow = np.array([], dtype=np.uint32)
    return _nibble4_unpack(main_bytes, overflow, n)


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
        # champion options: flags 0-29
        options = _get_options(zzag, pred_idx=0)
        # nibble4 for raw diff: flag 30
        nb4 = _nibble4_encode(zzag)
        options.append((len(nb4), 30, nb4))

        for cfg_idx, (order, mu) in enumerate(_NLMS_CONFIGS):
            res = _nlms_fwd(dc, order, mu)
            zz_r = _zigzag(res)
            options.extend(_get_options(zz_r, pred_idx=cfg_idx + 1))
            # nibble4 for NLMS residuals: flags 31-35
            nb4 = _nibble4_encode(zz_r)
            options.append((len(nb4), 31 + cfg_idx, nb4))

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

        if flag >= 30:
            pred_idx = flag - 30
            pack_type = 5  # nibble4
        else:
            pred_idx = flag // 5
            pack_type = flag % 5

        if pack_type == 0:
            zzag = _varzz_unpack(bz2.decompress(data), n_samp)
        elif pack_type == 1:
            zzag = _nibble_unpack(bz2.decompress(data), n_samp)
        elif pack_type == 5:
            zzag = _nibble4_decode(data, n_samp)
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
