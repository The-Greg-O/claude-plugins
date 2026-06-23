#!/bin/bash
# verify.sh — reproduce the headline result independently.
# Evaluates the shipped champion codec on the holdout and shows its margin over
# the strongest general-purpose baseline (bzip2-9). Deterministic; needs network
# only on the first run to fetch + sha256-verify the public MIT-BIH data.
set -uo pipefail
cd "$(dirname "$0")" || exit 1

echo "1/3  ensuring MIT-BIH data is present (fetch + sha256 verify)..."
python3 fetch_data.py >/dev/null || { echo "data fetch/verify failed"; exit 1; }

ratio() {  # $1 = candidate file -> prints ratio_holdout from the contract JSON
  python3 evaluate.py "$1" 2>/dev/null | python3 -c \
    "import sys,json; print(json.loads(sys.stdin.read().splitlines()[-1])['metrics'].get('ratio_holdout','ERR'))"
}

echo "2/3  evaluating champion (candidates/champion.py)..."
CH=$(ratio candidates/champion.py)
echo "3/3  evaluating bzip2-9 baseline..."
BZ=$(ratio candidates/baseline_bzip2.py)

echo
echo "  champion  ratio_holdout : ${CH}x"
echo "  bzip2-9   ratio_holdout : ${BZ}x   (strongest general-purpose baseline)"
python3 -c "print(f'  margin over bzip2-9     : {(float(\"$CH\")/float(\"$BZ\")-1)*100:+.1f}%')" 2>/dev/null || true
echo
echo "Lossless guaranteed by evaluate.py's gate: byte-exact decode(encode(x))==x"
echo "on every record. Re-run with no args to reproduce."
