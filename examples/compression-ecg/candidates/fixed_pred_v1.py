"""
Fixed-order finite-difference predictors + LZMA (FLAC-style).

For each channel, try orders 1-4 and pick the one minimising sum|residuals|.
Higher orders remove ECG's smooth local trends better than first-difference alone,
shrinking residual entropy before handing off to LZMA.

Format (after the canonical ECG header):
  [10:10+n_chan]      uint8 order per channel (1-4)
  [10+n_chan:...]     warmup int16 LE: each channel contributes `order` values
                      (initial values of derivatives 0 through order-1)
  [rest]              LZMA-compressed int16 LE residuals, channels concatenated
"""
import lzma
import struct

import numpy as np

HEAD = 10


def _parse(blob):
    n_chan, n_samp = struct.unpack("<HI", blob[4:HEAD])
    data = np.frombuffer(blob[HEAD:HEAD + n_chan * n_samp * 2], dtype="<i2")
    return n_chan, n_samp, data.reshape(n_chan, n_samp).astype(np.int64)


def _best_residuals(x):
    """Try orders 1-4; return (order, warmup_list, residuals_int16) for the winner."""
    best_score, best_order, best_warmup, best_r16 = None, None, None, None
    for order in range(1, 5):
        cur = x.copy()
        warmup = []
        for _ in range(order):
            warmup.append(int(cur[0]))
            cur = np.diff(cur)
        # cur = order-th order finite differences (ECG residuals fit well in int16)
        score = int(np.sum(np.abs(cur)))
        if best_score is None or score < best_score:
            best_score = score
            best_order = order
            best_warmup = warmup
            best_r16 = cur.astype(np.int16)
    return best_order, best_warmup, best_r16


def _reconstruct(warmup, res16, order):
    """Invert order-P finite differences via P iterated cumsums."""
    cur = res16.astype(np.int64)
    for p in range(order - 1, -1, -1):
        cur = np.concatenate([[warmup[p]], warmup[p] + np.cumsum(cur)])
    return cur.astype(np.int16)


def encode(blob):
    n_chan, n_samp, sig = _parse(blob)
    orders, warmups, residuals = [], [], []

    for ch in range(n_chan):
        o, wm, r = _best_residuals(sig[ch])
        orders.append(o)
        warmups.append(wm)
        residuals.append(r)

    meta = bytes(orders)
    wm_bytes = b''.join(struct.pack(f'<{len(w)}h', *w) for w in warmups)
    res_bytes = b''.join(r.tobytes() for r in residuals)
    return blob[:HEAD] + meta + wm_bytes + lzma.compress(res_bytes, preset=6)


def decode(comp):
    head = comp[:HEAD]
    n_chan, n_samp = struct.unpack("<HI", head[4:HEAD])
    pos = HEAD

    orders = list(comp[pos:pos + n_chan])
    pos += n_chan

    warmups = []
    for o in orders:
        wm = list(struct.unpack(f'<{o}h', comp[pos:pos + o * 2]))
        pos += o * 2
        warmups.append(wm)

    raw = lzma.decompress(comp[pos:])
    channels, off = [], 0
    for ch in range(n_chan):
        o = orders[ch]
        n_res = n_samp - o
        r = np.frombuffer(raw[off:off + n_res * 2], dtype='<i2')
        off += n_res * 2
        channels.append(_reconstruct(warmups[ch], r, o))

    return head + np.array(channels, dtype='<i2').tobytes()
