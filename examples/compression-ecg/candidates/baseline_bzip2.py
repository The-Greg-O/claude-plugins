"""Baseline: bzip2 -9 on the canonical blob. The strongest general-purpose
opponent on this data (BWT models ECG well) — the bar a real win must clear."""
import bz2


def encode(blob):
    return bz2.compress(blob, 9)


def decode(comp):
    return bz2.decompress(comp)
