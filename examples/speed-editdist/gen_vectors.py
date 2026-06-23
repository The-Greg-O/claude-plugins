"""Generate the frozen edit-distance test vectors + trusted expected answers.

The PRIMARY metric (fuel) is measured on these frozen vectors, so they are
committed and reproducible. Expected distances come from this trusted
Wagner-Fischer DP (the oracle) — candidates are graded against it, never the
other way around. Strings cluster near 64 bytes (one machine word) so the
Myers bit-parallel win is large; pairs are similar (a base string + a few random
edits) so distances are meaningful, but the O(n*m) vs O(n) fuel gap holds
regardless of the distance value.

Binary format (little-endian) consumed by fuel_host:
    u32 count
    per case: u32 a_len, a_bytes, u32 b_len, b_bytes, u32 expected
"""
import json
import os
import random
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = b"abcdefghijklmnop"   # 16 symbols
SEED = 20260619


def lev(a, b):
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def randstr(rng, n):
    return bytes(rng.choice(ALPHA) for _ in range(n))


def mutate(rng, s, k):
    s = bytearray(s)
    for _ in range(k):
        if not s:
            s.append(rng.choice(ALPHA))
            continue
        op = rng.randint(0, 2)
        i = rng.randrange(len(s))
        if op == 0:
            s[i] = rng.choice(ALPHA)            # substitute
        elif op == 1:
            s.insert(i, rng.choice(ALPHA))      # insert
        else:
            del s[i]                            # delete
    return bytes(s)


def build_cases():
    rng = random.Random(SEED)
    # strings capped at 64 bytes (one machine word) so single-word Myers is a
    # valid target; lengths cluster near 64 to make its bit-parallel win large.
    lengths = [56, 60, 62, 63, 64, 64, 64, 58, 52, 48, 40, 32, 24, 16]
    cases = []
    for _ in range(6):
        for L in lengths:
            a = randstr(rng, L)
            b = mutate(rng, a, rng.randint(1, max(1, L // 6)))
            cases.append((a[:64], b[:64]))
    cases += [
        (b"", b""), (b"", b"abc"), (b"abc", b""),
        (b"x" * 64, b"x" * 64),                 # identical, distance 0
        (randstr(rng, 64), randstr(rng, 64)),   # ~all different
        (b"kitten", b"sitting"),                # classic, distance 3
    ]
    return cases


def main():
    cases = build_cases()
    out = struct.pack("<I", len(cases))
    summary = []
    for a, b, in cases:
        d = lev(a, b)
        out += struct.pack("<I", len(a)) + a + struct.pack("<I", len(b)) + b + struct.pack("<I", d)
        summary.append({"a_len": len(a), "b_len": len(b), "dist": d})
    with open(os.path.join(HERE, "vectors.bin"), "wb") as f:
        f.write(out)
    with open(os.path.join(HERE, "vectors.summary.json"), "w") as f:
        json.dump({"seed": SEED, "count": len(cases),
                   "kitten_sitting_check": lev(b"kitten", b"sitting"),
                   "cases": summary}, f, indent=2)
    print(f"wrote vectors.bin: {len(cases)} cases, {len(out)} bytes; "
          f"kitten/sitting dist={lev(b'kitten', b'sitting')} (expect 3)")


if __name__ == "__main__":
    main()
