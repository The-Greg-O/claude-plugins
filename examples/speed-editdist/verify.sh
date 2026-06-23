#!/bin/bash
# verify.sh — reproduce the headline result independently.
# Builds the fuel host, compiles the champion + the naive seed to wasm, and
# reports each one's deterministic fuel on the frozen vectors + the speedup.
# Fully reproducible and offline (test vectors are generated deterministically).
set -uo pipefail
cd "$(dirname "$0")" || exit 1

echo "1/4  building fuel host (once; pulls wasmi)..."
cargo build --release --manifest-path fuel_host/Cargo.toml >/dev/null 2>&1 || { echo "host build failed"; exit 1; }
HOST=fuel_host/target/release/fuel_host

echo "2/4  ensuring frozen vectors..."
[ -f vectors.bin ] || python3 gen_vectors.py >/dev/null

fuelof() {  # $1 = .rs path -> prints fuel, or ERR / INCORRECT
  rustc --edition 2021 --target wasm32-unknown-unknown -O --crate-type=cdylib "$1" -o /tmp/verify.wasm 2>/dev/null \
    || { echo ERR; return; }
  "$HOST" /tmp/verify.wasm vectors.bin | python3 -c \
    "import sys,json; d=json.loads(sys.stdin.read()); print(d['total_fuel'] if d.get('correct') else 'INCORRECT')"
}

echo "3/4  champion (candidates/champion.rs)..."; CH=$(fuelof candidates/champion.rs)
echo "4/4  naive seed (candidates/seed.rs)...";     SD=$(fuelof candidates/seed.rs)

echo
echo "  champion fuel : ${CH}"
echo "  naive seed    : ${SD}"
python3 -c "print(f'  speedup       : {float(\"$SD\")/float(\"$CH\"):.1f}x fewer wasm instructions')" 2>/dev/null || true
echo
echo "Correctness is enforced by the gate (byte-exact distance on every frozen"
echo "vector, plus fresh random pairs during the loop). Fuel is deterministic."
