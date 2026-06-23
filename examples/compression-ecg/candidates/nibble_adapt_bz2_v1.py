"""
nibble_adapt_bz2_v1 — per-channel best-of-two: nibble-pair encoding vs varzz.

Nibble-pair packs consecutive sample pairs where both zzag<8 into one byte
(3 bits + 3 bits), shrinking the stream by 35-44% for records with ≥75% small
deltas. bz2 then sees more heartbeat-period repetitions per block => better BWT.
For records with fewer small deltas (irregular rhythms), the pair-boundary shifts
hurt BWT alignment more than the shorter stream helps; plain varzz wins there.

Per channel: encode both, compress both with bz2-9, store whichever is smaller.
A single flag byte (0=varzz, 1=nibble) per channel signals the decoder.

Byte layout:
  ECG1 header (10B) | n_chan uint32 | per-channel: [uint8 flag][uint32 clen][data]

Nibble encoding byte space:
  0x00-0x3F  pair  (z<<3)|z2  both zzag<8
  0x40-0xBF  single small  0x40+z  zzag 0-127
  0xC0-0xFE  2-byte large  0xC0+(v>>8), v&0xFF  zzag 128-32639 (v=zzag-128)
  0xFF       3-byte escape  0xFF, z>>8, z&0xFF  zzag 32640-65535
"""
import bz2
import struct

import numpy as np

MAGIC = b'ECG1'
HEAD = 10


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
    offsets = np.empty(n + 1, dtype=np.intp)
    offsets[0] = 0
    np.cumsum(sizes, out=offsets[1:])
    out = np.zeros(int(offsets[-1]), dtype=np.uint8)
    if t0.any():
        out[offsets[:-1][t0]] = zzag[t0].astype(np.uint8)
    if t1.any():
        p = offsets[:-1][t1]
        v = zzag[t1].astype(np.uint32) - 128
        out[p] = (0x80 + (v >> 8)).astype(np.uint8)
        out[p + 1] = (v & 0xFF).astype(np.uint8)
    if t2.any():
        p = offsets[:-1][t2]
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
        zzag = _zigzag(diff[c])
        vz_c = bz2.compress(_varzz_pack(zzag), compresslevel=9)
        nb_c = bz2.compress(_nibble_pack(zzag), compresslevel=9)
        if len(vz_c) <= len(nb_c):
            chan_parts.append(struct.pack('<BI', 0, len(vz_c)) + vz_c)
        else:
            chan_parts.append(struct.pack('<BI', 1, len(nb_c)) + nb_c)

    return blob[:HEAD] + struct.pack('<I', n_chan) + b''.join(chan_parts)


def decode(blob):
    header = blob[:HEAD]
    assert header[:4] == MAGIC
    n_chan, n_samp = struct.unpack_from('<HI', header, 4)

    pos = HEAD + 4  # skip n_chan uint32
    channels = []
    for _ in range(n_chan):
        flag = blob[pos]
        clen = struct.unpack_from('<I', blob, pos + 1)[0]
        pos += 5
        data = bz2.decompress(blob[pos:pos + clen])
        pos += clen
        if flag == 0:
            zzag = _varzz_unpack(data, n_samp)
        else:
            zzag = _nibble_unpack(data, n_samp)
        channels.append(_unzigzag(zzag))

    diff = np.stack(channels, axis=0).astype(np.int64)
    sig = np.cumsum(diff, axis=1)
    sig = ((sig + 32768) % 65536 - 32768).astype('<i2')
    return header + sig.tobytes()
