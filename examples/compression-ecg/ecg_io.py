"""WFDB format-212 reader + the canonical ECG container for this experiment.

MIT-BIH signals ship in WFDB "format 212": two 12-bit signed samples packed into
3 bytes, with channels interleaved per time step. General-purpose compressors do
poorly on that packed/misaligned stream, so we unpack once to a fair, explicit
representation and let every codec (baselines and candidates) compete on THAT.

Canonical record file (`.ecg`) — lossless, self-describing:
    magic   b'ECG1'                 (4 bytes)
    n_chan  uint16 little-endian    (2 bytes)
    n_samp  uint32 little-endian    (4 bytes)   # samples per channel
    body    n_chan * n_samp int16 little-endian, channel-major (ch0..., ch1...)

Codecs compress these bytes and must reproduce them exactly (the gate).
"""
import struct

import numpy as np

MAGIC = b"ECG1"
HEADER_LEN = 10


def parse_header(hea_path):
    """Return (n_chan, n_samp, [format_codes]) from a WFDB .hea header."""
    lines = [ln.strip() for ln in open(hea_path)
             if ln.strip() and not ln.startswith("#")]
    rec = lines[0].split()
    n_chan, n_samp = int(rec[1]), int(rec[3])
    fmts = [lines[1 + i].split()[1] for i in range(n_chan)]
    return n_chan, n_samp, fmts


def unpack_212(dat_path, n_chan, n_samp):
    """Decode a format-212 .dat into an int16 array shaped [n_chan, n_samp]."""
    raw = np.frombuffer(open(dat_path, "rb").read(), dtype=np.uint8)
    total = n_chan * n_samp
    ngroups = (total + 1) // 2
    b = raw[:ngroups * 3].reshape(-1, 3).astype(np.int32)
    b0, b1, b2 = b[:, 0], b[:, 1], b[:, 2]
    s1 = b0 | ((b1 & 0x0F) << 8)
    s2 = b2 | ((b1 & 0xF0) << 4)
    s1 = np.where(s1 >= 2048, s1 - 4096, s1)   # sign-extend 12-bit
    s2 = np.where(s2 >= 2048, s2 - 4096, s2)
    inter = np.empty(ngroups * 2, dtype=np.int16)
    inter[0::2] = s1
    inter[1::2] = s2
    inter = inter[:total]
    # samples are channel-interleaved per time step -> [n_samp, n_chan] -> .T
    return np.ascontiguousarray(inter.reshape(n_samp, n_chan).T)


def build_canonical(sig):
    """Pack an int16 [n_chan, n_samp] array into the canonical .ecg blob."""
    n_chan, n_samp = sig.shape
    head = MAGIC + struct.pack("<HI", n_chan, n_samp)
    return head + np.ascontiguousarray(sig, dtype="<i2").tobytes()


def parse_canonical(blob):
    """Inverse of build_canonical: blob -> int16 [n_chan, n_samp]."""
    if blob[:4] != MAGIC:
        raise ValueError("not an ECG1 canonical blob")
    n_chan, n_samp = struct.unpack("<HI", blob[4:HEADER_LEN])
    body = np.frombuffer(blob[HEADER_LEN:HEADER_LEN + n_chan * n_samp * 2], dtype="<i2")
    return body.reshape(n_chan, n_samp)


def record_to_canonical(dat_path, hea_path):
    """Read a WFDB record (.dat + .hea) and return its canonical .ecg blob."""
    n_chan, n_samp, fmts = parse_header(hea_path)
    if any(f != "212" for f in fmts):
        raise ValueError(f"{dat_path}: only WFDB format 212 supported, got {fmts}")
    return build_canonical(unpack_212(dat_path, n_chan, n_samp))
