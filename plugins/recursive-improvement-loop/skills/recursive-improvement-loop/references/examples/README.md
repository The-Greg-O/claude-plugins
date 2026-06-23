# Worked examples

Case studies of **real** recursive-improvement loops — each from an actual run, not
an invented narrative: the objective, the gate, the seed, and the champion
trajectory with reproducible numbers. The full runnable experiments live at the
repo's top-level `examples/` and each ships a `verify.sh` that reproduces the
headline number.

| example | domain | result | verify |
|---|---|---|---|
| [compression-ecg](compression-ecg.md) | lossless ECG compression (Python) | 3.90× → **4.31×**, +12% past bzip2 | `examples/compression-ecg/verify.sh` |
| [speed-editdist](speed-editdist.md) | faster Rust algorithm, deterministic fuel | naive → **425× fewer** instructions (rediscovers Myers) | `examples/speed-editdist/verify.sh` |

Both show the same shape: a naive-but-correct seed, a hard correctness gate, and a
loop that discovers the advanced technique on its own — NLMS adaptive prediction in
one, Myers bit-parallelism (and then *beyond* it) in the other — documenting every
promotion and dead end in a living lab notebook.
