"""evaluate.py — trusted referee for the Rust edit-distance speed loop.

Contract (loop.py calls `python3 evaluate.py <candidate.rs>`):
  - candidate is a single .rs file exposing the experiment ABI:
        input_ptr() -> u32
        solve(a_ptr, a_len, b_ptr, b_len) -> u32     (the edit distance)
    It is compiled to wasm32-unknown-unknown with NO host imports, so it can
    only compute — the wasm boundary is the sandbox (no AST check needed).
  - GATE (hard): it compiles, AND returns the correct distance on every FROZEN
    vector AND on fresh host-generated random pairs (so it can't special-case the
    frozen set), AND the source is within the size cap (embed-the-answer guard).
  - METRIC: total wasmi FUEL consumed by solve over the frozen vectors
    (deterministic, instruction-proportional). MINIMIZE.
"""
import json
import os
import random
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
HOST = os.path.join(HERE, "fuel_host", "target", "release", "fuel_host")
VECTORS = os.path.join(HERE, "vectors.bin")
MAX_SOURCE_BYTES = 65536
COMPILE_TIMEOUT_S = 120
HOST_TIMEOUT_S = 120
FRESH_CASES = 40
ALPHA = b"abcdefghijklmnop"


def contract(gate_passed, gate_error, metrics):
    print(json.dumps({"gate_passed": gate_passed, "gate_error": gate_error, "metrics": metrics}))


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


def write_vectors(path, cases):
    out = struct.pack("<I", len(cases))
    for a, b in cases:
        out += struct.pack("<I", len(a)) + a + struct.pack("<I", len(b)) + b + struct.pack("<I", lev(a, b))
    with open(path, "wb") as f:
        f.write(out)


def fresh_cases(rng):
    cases = []
    for _ in range(FRESH_CASES):
        L = rng.randint(8, 64)
        a = bytes(rng.choice(ALPHA) for _ in range(L))
        s = bytearray(a)
        for _ in range(rng.randint(1, max(1, L // 5))):
            if not s:
                break
            op, i = rng.randint(0, 2), rng.randrange(len(s))
            if op == 0:
                s[i] = rng.choice(ALPHA)
            elif op == 1:
                s.insert(i, rng.choice(ALPHA))
            else:
                del s[i]
        cases.append((a[:64], bytes(s)[:64]))   # cap at 64 bytes (single-word Myers domain)
    return cases


def run_host(wasm_path, vectors_path):
    try:
        p = subprocess.run([HOST, wasm_path, vectors_path],
                           capture_output=True, text=True, timeout=HOST_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"correct": False, "error": f"host exceeded {HOST_TIMEOUT_S}s"}
    line = (p.stdout.strip().splitlines() or [""])[-1]
    try:
        return json.loads(line)
    except Exception:
        return {"correct": False, "error": f"host gave no result: {(p.stderr or line)[-300:]}"}


def main():
    if len(sys.argv) < 2:
        return contract(False, "usage: evaluate.py <candidate.rs>", {})
    src_path = os.path.abspath(sys.argv[1])
    if not os.path.isfile(src_path):
        return contract(False, f"no such candidate file: {sys.argv[1]}", {})
    if os.path.getsize(src_path) > MAX_SOURCE_BYTES:
        return contract(False, f"source > {MAX_SOURCE_BYTES} bytes (embed-the-answer guard)", {})
    if not os.path.isfile(HOST):
        return contract(False, "fuel_host not built — run: (cd fuel_host && cargo build --release)", {})

    with tempfile.TemporaryDirectory() as tmp:
        wasm = os.path.join(tmp, "cand.wasm")
        comp = subprocess.run(
            ["rustc", "--edition", "2021", "--target", "wasm32-unknown-unknown",
             "-O", "--crate-type=cdylib", src_path, "-o", wasm],
            capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S)
        if comp.returncode != 0 or not os.path.exists(wasm):
            tail = comp.stderr.strip().splitlines()[-4:]
            return contract(False, "compile failed: " + " | ".join(tail), {})

        frozen = run_host(wasm, VECTORS)
        if not frozen.get("correct"):
            return contract(False, "frozen vectors: " + json.dumps(frozen), {})

        rng = random.Random(int.from_bytes(os.urandom(8), "little"))
        fresh_path = os.path.join(tmp, "fresh.bin")
        write_vectors(fresh_path, fresh_cases(rng))
        fresh = run_host(wasm, fresh_path)
        if not fresh.get("correct"):
            return contract(False, "WRONG on fresh random inputs (overfit/hard-coded?): "
                                   + json.dumps(fresh), {})

        fuel = frozen["total_fuel"]
        metrics = {"fuel": fuel,
                   "fuel_per_case": round(fuel / frozen["n"], 1),
                   "wasm_bytes": os.path.getsize(wasm)}
        print(f"correct on {frozen['n']} frozen + {FRESH_CASES} fresh cases; "
              f"total fuel = {fuel:,}")
        contract(True, "", metrics)


if __name__ == "__main__":
    main()
