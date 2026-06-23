"""Baseline: xz/LZMA preset 9 | EXTREME on the canonical blob (no modeling)."""
import lzma


def encode(blob):
    return lzma.compress(blob, preset=9 | lzma.PRESET_EXTREME)


def decode(comp):
    return lzma.decompress(comp)
