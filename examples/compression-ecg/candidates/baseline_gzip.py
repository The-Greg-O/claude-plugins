"""Baseline: gzip/DEFLATE level 9 on the canonical blob (no signal modeling)."""
import zlib


def encode(blob):
    return zlib.compress(blob, 9)


def decode(comp):
    return zlib.decompress(comp)
