"""Fetch MIT-BIH Arrhythmia records and build the canonical .ecg data set.

Public source: PhysioNet MIT-BIH Arrhythmia Database v1.0.0 (ODC-By 1.0, see
NOTICE). Downloads <record>.dat + <record>.hea per record (small, ~2 MB each;
the AWS Open Data S3 mirror is tried first as it is range-resumable and stable),
unpacks WFDB format-212 to the canonical int16 container (see ecg_io.py), writes
data/<split>/<record>.ecg, and pins the sha256 of every canonical file in
data.manifest.json.

    python3 fetch_data.py            # download + (re)write the manifest
    python3 fetch_data.py --verify   # re-hash existing files against the manifest

Record-disjoint split below is a curated 12-record subset. To run the loop on the
full standard de Chazal DS1/DS2 inter-patient split, replace SPLIT accordingly.
"""
import hashlib
import json
import os
import sys
import urllib.request

import ecg_io

MIRRORS = (
    "https://physionet-open.s3.amazonaws.com/mitdb/1.0.0",
    "https://physionet.org/files/mitdb/1.0.0",
)

SPLIT = {
    "train":   [101, 103, 105, 108, 209, 215, 230],
    "holdout": [100, 119, 207, 214, 232],
}

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MANIFEST = os.path.join(HERE, "data.manifest.json")


def download(name, dest):
    last = None
    for base in MIRRORS:
        try:
            with urllib.request.urlopen(f"{base}/{name}", timeout=90) as r:
                blob = r.read()
            with open(dest, "wb") as f:
                f.write(blob)
            return
        except Exception as e:   # try the next mirror
            last = e
    raise RuntimeError(f"could not download {name} from any mirror: {last}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    verify_only = "--verify" in sys.argv
    prev = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {"records": {}}
    manifest = {
        "source": "PhysioNet MIT-BIH Arrhythmia Database v1.0.0",
        "license": "ODC-By 1.0 (see NOTICE)",
        "doi": "10.13026/C2F305",
        "split": {k: list(v) for k, v in SPLIT.items()},
        "records": {},
    }
    rawdir = os.path.join(DATA, "_raw")
    for split, records in SPLIT.items():
        outdir = os.path.join(DATA, split)
        os.makedirs(outdir, exist_ok=True)
        for rec in records:
            ecg = os.path.join(outdir, f"{rec}.ecg")
            if not os.path.exists(ecg):
                if verify_only:
                    sys.exit(f"FATAL: {ecg} missing (run without --verify to fetch)")
                os.makedirs(rawdir, exist_ok=True)
                dat = os.path.join(rawdir, f"{rec}.dat")
                hea = os.path.join(rawdir, f"{rec}.hea")
                download(f"{rec}.dat", dat)
                download(f"{rec}.hea", hea)
                with open(ecg, "wb") as f:
                    f.write(ecg_io.record_to_canonical(dat, hea))
            digest = sha256(ecg)
            pinned = prev.get("records", {}).get(str(rec), {}).get("sha256")
            if pinned and pinned != digest:
                sys.exit(f"FATAL: {rec}.ecg sha256 drift {digest} != pinned {pinned}")
            manifest["records"][str(rec)] = {
                "split": split, "bytes": os.path.getsize(ecg), "sha256": digest,
            }
    if not verify_only:
        with open(MANIFEST, "w") as f:
            json.dump(manifest, f, indent=2)
    n = sum(len(v) for v in SPLIT.values())
    print(f"OK: {n} records ready ({len(SPLIT['train'])} train, "
          f"{len(SPLIT['holdout'])} holdout) in {DATA}")


if __name__ == "__main__":
    main()
