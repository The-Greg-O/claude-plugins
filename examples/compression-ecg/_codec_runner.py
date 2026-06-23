"""Trusted sandboxed runner — applies a candidate codec to one split's records
and reports per-record sizes + byte-exact roundtrip status as a single JSON line.

evaluate.py launches this in a subprocess (timeout + neutral cwd). The candidate
only ever receives in-memory bytes here — it never sees a file path or the
manifest — so it cannot peek at the holdout even before the static import check.
This file is harness code, NOT candidate code.
"""
import importlib.util
import json
import os
import sys


def load_codec(path):
    spec = importlib.util.spec_from_file_location("candidate_codec", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def emit(obj):
    print(json.dumps(obj))


def main():
    codec_path, data_dir, manifest_path, split = sys.argv[1:5]
    manifest = json.load(open(manifest_path))
    records = sorted(r for r, info in manifest["records"].items()
                     if info["split"] == split)
    try:
        codec = load_codec(codec_path)
        encode, decode = codec.encode, codec.decode
    except Exception as e:
        return emit({"gate_ok": False, "error": f"candidate import failed: {e}"})

    total_raw = total_enc = 0
    per = []
    for r in records:
        info = manifest["records"][r]
        blob = open(os.path.join(data_dir, info["split"], f"{r}.ecg"), "rb").read()
        try:
            enc = encode(blob)
            if not isinstance(enc, (bytes, bytearray)):
                return emit({"gate_ok": False,
                             "error": f"encode returned {type(enc).__name__}, not bytes (rec {r})"})
            dec = decode(bytes(enc))
        except Exception as e:
            return emit({"gate_ok": False, "error": f"codec raised on rec {r}: {e}"})
        if bytes(dec) != blob:
            return emit({"gate_ok": False, "error": f"LOSSY: roundtrip mismatch on rec {r}"})
        total_raw += len(blob)
        total_enc += len(enc)
        per.append([r, len(blob), len(enc)])
    emit({"gate_ok": True, "total_raw": total_raw, "total_enc": total_enc, "records": per})


if __name__ == "__main__":
    main()
