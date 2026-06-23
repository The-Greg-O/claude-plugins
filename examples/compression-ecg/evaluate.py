"""evaluate.py — trusted referee for the MIT-BIH ECG lossless-compression loop.

Contract (loop.py calls `python3 evaluate.py <candidate>`):
  - candidate is a directory (with codec.py) or a .py file providing:
        encode(blob: bytes) -> bytes      decode(blob: bytes) -> bytes
    operating on a canonical .ecg blob (see ecg_io.py).
  - GATE (hard, byte-exact): decode(encode(x)) == x on EVERY record, the
    candidate passes a static no-peeking import/IO check, and its source is
    within the size cap (embed-the-answer guard).
  - METRIC: lossless ratio = raw_bytes / compressed_bytes on the subject-disjoint
    HOLDOUT split (primary), plus the train ratio for overfitting visibility.
    Maximize.

The candidate runs in a subprocess (neutral cwd + timeout) and is handed only
in-memory bytes, never file paths — so it cannot read the holdout.
"""
import ast
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MANIFEST = os.path.join(HERE, "data.manifest.json")
RUNNER = os.path.join(HERE, "_codec_runner.py")

MAX_SOURCE_BYTES = 65536
RUNNER_TIMEOUT_S = 240
ALLOWED_IMPORTS = {"numpy", "struct", "zlib", "bz2", "lzma", "math", "array",
                   "collections", "itertools", "functools", "typing", "heapq"}
BANNED_CALLS = {"open", "eval", "exec", "compile", "__import__", "input"}


def static_check(src):
    """Return an error string if the candidate source is unsafe, else None."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return f"syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"banned import '{a.name}'"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return f"banned import-from '{node.module}'"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BANNED_CALLS:
                return f"banned call '{node.func.id}()'"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return f"banned dunder access '.{node.attr}'"
    return None


def resolve_codec(candidate):
    p = os.path.abspath(candidate)
    return os.path.join(p, "codec.py") if os.path.isdir(p) else p


def contract(gate_passed, gate_error, metrics):
    print(json.dumps({"gate_passed": gate_passed,
                      "gate_error": gate_error, "metrics": metrics}))


def run_split(codec_path, split):
    with tempfile.TemporaryDirectory() as tmp:
        p = subprocess.run(
            [sys.executable, RUNNER, codec_path, DATA, MANIFEST, split],
            capture_output=True, text=True, cwd=tmp, timeout=RUNNER_TIMEOUT_S)
    line = (p.stdout.strip().splitlines() or [""])[-1]
    try:
        return json.loads(line)
    except Exception:
        return {"gate_ok": False,
                "error": f"runner produced no valid result: {(p.stderr or line)[-300:]}"}


def main():
    if len(sys.argv) < 2:
        return contract(False, "usage: evaluate.py <candidate>", {})
    codec_path = resolve_codec(sys.argv[1])
    if not os.path.isfile(codec_path):
        return contract(False, f"no codec.py at {sys.argv[1]}", {})

    src = open(codec_path, "rb").read()
    if len(src) > MAX_SOURCE_BYTES:
        return contract(False, f"codec source {len(src)} > {MAX_SOURCE_BYTES} bytes "
                               f"(embed-the-answer guard)", {})
    bad = static_check(src.decode("utf-8", "replace"))
    if bad:
        return contract(False, f"static no-peeking check failed: {bad}", {})
    if not os.path.exists(MANIFEST):
        return contract(False, "data.manifest.json missing — run fetch_data.py first", {})

    try:
        hold = run_split(codec_path, "holdout")
    except subprocess.TimeoutExpired:
        return contract(False, f"holdout eval exceeded {RUNNER_TIMEOUT_S}s", {})
    if not hold.get("gate_ok"):
        return contract(False, hold.get("error", "holdout gate failed"), {})

    ratio_h = hold["total_raw"] / hold["total_enc"]
    metrics = {"ratio_holdout": round(ratio_h, 4),
               "compressed_kb_holdout": round(hold["total_enc"] / 1024, 1)}
    print(f"holdout: {hold['total_raw'] / 1024:.0f} KB -> {hold['total_enc'] / 1024:.0f} KB"
          f"  ratio={ratio_h:.4f}  over {len(hold['records'])} records")

    try:
        tr = run_split(codec_path, "train")
        if tr.get("gate_ok"):
            metrics["ratio_train"] = round(tr["total_raw"] / tr["total_enc"], 4)
            print(f"train:   ratio={metrics['ratio_train']:.4f}")
    except subprocess.TimeoutExpired:
        pass

    contract(True, "", metrics)


if __name__ == "__main__":
    main()
