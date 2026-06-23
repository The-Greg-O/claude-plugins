"""
cascade_nlms_v1 -- two-stage cascade NLMS predictor.

Parent: rice_nlms_v1 (champion 4.3113x, best-of-30).
Lineage: cascade-nlms.

Key insight (probed on train/101.ecg NLMS-8 residuals):
  lag-1 autocorrelation = 0.807, lag-5 = -0.288, lag-385 ~ -0.011.
The diff-domain NLMS leaves substantial short-range autocorrelation in its
own residuals. Stage-2 NLMS applied to r1 targets exactly this structure and
can reduce variance by ~65% (1 - 0.807^2 ~ 0.35). Heartbeat-period
autocorrelation is negligible, confirming why the period predictor gave 0x.

Targeted cascade configs (to stay within eval timeout):
  s1=raw_diff + s2=NLMS-2/mu=0.30
  s1=NLMS-4   + s2=NLMS-2/mu=0.30
  s1=NLMS-8   + s2=NLMS-2/mu=0.30
  s1=NLMS-8   + s2=NLMS-4/mu=0.20

Best-of-50 = champion 30 + 4 cascade configs x 5 pack types.
Champion options are a strict subset => no regression by construction.

Flag encoding:
  pred_idx = flag // 5   (0-5: champion options; 6-9: cascade configs)
  pack_type = flag % 5   (0=varzz+bz2, 1=nibble+bz2, 2-4=Rice k=1/2/3)

For cascade flags (pred_idx >= 6), data starts with:
  [uint8 s1_idx][uint8 s2_idx]   (2-byte header identifying the cascade config)
  then packed residuals (same format as champion options)

s1_idx: 0=raw_diff, 1-5=NLMS_CONFIGS index + 1
s2_idx: index into _CASCADE_S2_CONFIGS
"""
import bz2
import struct

import numpy as np

MAGIC = b'ECG1'
HEAD = 10
_EPS = 1e-6

_NLMS_CONFIGS = [(2, 0.20), (4, 0.10), (8, 0.05), (16, 0.05), (32, 0.03)]
_RICE_KS = [1, 2, 3]

# Stage-2 configs: short-order, high-mu NLMS applied to stage-1 residuals.
_CASCADE_S2_CONFIGS = [(2, 0.30), (4, 0.20)]

# Targeted cascade pairs: (s1_idx, s2_idx)
#   s1_idx 0=raw diff, 3=NLMS_CONFIGS[2]=(8,0.05), 2=NLMS_CONFIGS[1]=(4,0.10)
_CASCADE_PAIRS = [
    (0, 0),  # raw_diff + NLMS-2/mu=0.30
    (2, 0),  # NLMS-4   + NLMS-2/mu=0.30
    (3, 0),  # NLMS-8   + NLMS-2/mu=0.30
    (3, 1),  # NLMS-8   + NLMS-4/mu=0.20
]


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

def _nlms_fwd(signal, order, mu):
    n = len(signal)
    res = np.empty(n, dtype=np.int16)
    w = np.zeros(order, dtype=np.float64)
    x = np.zeros(order, dtype=np.float64)
    s_int = signal.astype(np.int32)
    for i in range(n):
        pred = int(round(float(np.dot(w, x))))
        r = (int(s_int[i]) - pred + 32768) % 65536 - 32768
        res[i] = r
        norm = float(np.dot(x, x)) + _EPS
        w += (mu / norm) * float(r) * x
        x[1:] = x[:-1]
        x[0] = float(s_int[i])
    return res


def _nlms_inv(signal, order, mu):
    n = len(signal)
    out = np.empty(n, dtype=np.int16)
    w = np.zeros(order, dtype=np.float64)
    x = np.zeros(order, dtype=np.float64)
    r_int = signal.astype(np.int32)
    for i in range(n):
        pred = int(round(float(np.dot(w, x))))
        d = (int(r_int[i]) + pred + 32768) % 65536 - 32768
        out[i] = d
        norm = float(np.dot(x, x)) + _EPS
        w += (mu / norm) * float(r_int[i]) * x
        x[1:] = x[:-1]
        x[0] = float(d)
    return out


# ---------------------------------------------------------------------------
# Option builders
# ---------------------------------------------------------------------------

def _pack_options(zzag, flag_base):
    c_vz = bz2.compress(_varzz_pack(zzag), 9)
    c_nb = bz2.compress(_nibble_pack(zzag), 9)
    opts = [(len(c_vz), flag_base, c_vz), (len(c_nb), flag_base + 1, c_nb)]
    for ri, k in enumerate(_RICE_KS):
        c = _rice_encode(zzag, k)
        opts.append((len(c), flag_base + 2 + ri, c))
    return opts


def _cascade_pack_options(zzag, pair_idx, flag_base):
    s1_idx, s2_idx = _CASCADE_PAIRS[pair_idx]
    hdr = struct.pack('<BB', s1_idx, s2_idx)
    c_vz = bz2.compress(_varzz_pack(zzag), 9)
    c_nb = bz2.compress(_nibble_pack(zzag), 9)
    opts = [
        (2 + len(c_vz), flag_base, hdr + c_vz),
        (2 + len(c_nb), flag_base + 1, hdr + c_nb),
    ]
    for ri, k in enumerate(_RICE_KS):
        c = _rice_encode(zzag, k)
        opts.append((2 + len(c), flag_base + 2 + ri, hdr + c))
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

        # Champion options: raw (pred_idx=0) + 5 NLMS configs (pred_idx=1..5)
        # Cache stage-1 residuals indexed 0=raw, 1..5=NLMS configs
        options = _pack_options(_zigzag(dc), flag_base=0)
        stage1 = [dc]  # index 0 = raw diff (s1_idx=0)
        for cfg_idx, (order, mu) in enumerate(_NLMS_CONFIGS):
            r1 = _nlms_fwd(dc, order, mu)
            stage1.append(r1)  # index cfg_idx+1 = s1_idx cfg_idx+1
            options.extend(_pack_options(_zigzag(r1), flag_base=(cfg_idx + 1) * 5))

        # Cascade options: stage-2 NLMS on specific stage-1 outputs
        # Compute stage-2 only for the needed s1 outputs
        stage2_needed = {}  # s1_idx -> set of s2_idx
        for pair_idx, (s1_idx, s2_idx) in enumerate(_CASCADE_PAIRS):
            stage2_needed.setdefault(s1_idx, set()).add(s2_idx)

        # Run stage-2 passes and collect options
        r2_cache = {}  # (s1_idx, s2_idx) -> r2
        for s1_idx, s2_set in stage2_needed.items():
            r1 = stage1[s1_idx]
            for s2_idx in s2_set:
                order2, mu2 = _CASCADE_S2_CONFIGS[s2_idx]
                r2_cache[(s1_idx, s2_idx)] = _nlms_fwd(r1, order2, mu2)

        for pair_idx, (s1_idx, s2_idx) in enumerate(_CASCADE_PAIRS):
            r2 = r2_cache[(s1_idx, s2_idx)]
            flag_base = (6 + pair_idx) * 5
            options.extend(_cascade_pack_options(_zigzag(r2), pair_idx, flag_base))

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
        is_cascade = pred_idx >= 6

        if is_cascade:
            s1_idx = data[0]
            s2_idx = data[1]
            payload = data[2:]
        else:
            payload = data

        if pack_type == 0:
            zzag = _varzz_unpack(bz2.decompress(payload), n_samp)
        elif pack_type == 1:
            zzag = _nibble_unpack(bz2.decompress(payload), n_samp)
        else:
            zzag = _rice_decode(payload, n_samp, _RICE_KS[pack_type - 2])

        vals = _unzigzag(zzag)

        if is_cascade:
            order2, mu2 = _CASCADE_S2_CONFIGS[s2_idx]
            vals = _nlms_inv(vals, order2, mu2)  # invert stage 2
            if s1_idx > 0:
                order1, mu1 = _NLMS_CONFIGS[s1_idx - 1]
                vals = _nlms_inv(vals, order1, mu1)  # invert stage 1
        else:
            if pred_idx > 0:
                order, mu = _NLMS_CONFIGS[pred_idx - 1]
                vals = _nlms_inv(vals, order, mu)

        channels.append(vals)

    diff = np.stack(channels, axis=0).astype(np.int64)
    sig = np.cumsum(diff, axis=1)
    sig = ((sig + 32768) % 65536 - 32768).astype('<i2')
    return header + sig.tobytes()
