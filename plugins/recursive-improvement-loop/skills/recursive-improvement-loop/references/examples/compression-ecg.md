# Worked example — lossless ECG compression (MIT-BIH)

**Result:** a recursive-improvement loop evolved a lossless codec for MIT-BIH ECG
from a **3.90×** seed to **4.3133×** over 26 iterations — **+12.1% past bzip2-9
(3.85×)**, the strongest general-purpose baseline — by discovering, on its own,
variable-length zigzag packing + **NLMS adaptive prediction** + best-of-N entropy
backends. Every champion is byte-exact lossless and the headline number is
independently reproducible (`examples/compression-ecg/verify.sh`).

Runnable experiment: [`examples/compression-ecg/`](../../../../../../examples/compression-ecg).

## Problem
- **Objective:** maximize lossless compression ratio (`raw_bytes / compressed_bytes`).
- **Data:** MIT-BIH Arrhythmia v1.0.0 (PhysioNet, ODC-By), WFDB format-212 unpacked
  to the canonical **16-bit** signal — so a win is real signal modeling, not undoing
  the 12-bit packing. 12 records, **subject-disjoint** train (7) / holdout (5).
- **Primary metric:** ratio on the holdout (promotion is holdout-only).
- **Gate:** byte-exact `decode(encode(x)) == x` on every record + a static
  no-peeking import check; candidates run sandboxed.
- **Baselines (holdout):** gzip-9 2.26× · xz-9e 3.23× · **bzip2-9 3.85× ← the bar**.

## The run (9 promotions across 26 iterations)
| iter | champion | ratio | what changed |
|---|---|---|---|
| 1 | seed | 3.90× | per-channel delta + LZMA — only *ties* bzip2 |
| 6 | varzz_bz2 | 4.09× | variable-length zigzag pack halves the stream → bz2's BWT sees 2× the heartbeat periods |
| 8 | perchan_varzz_bz2 | 4.10× | per-channel bz2 (avoid the BWT block split) |
| 9 | nibble_adapt_bz2 | 4.16× | pack small-delta *pairs* into one byte, best-of-two vs varzz |
| 10–12 | nlms_bz2 | 4.17→4.28→**4.31×** | **NLMS adaptive linear prediction**, best-of-N over (order, μ) |
| 14 | rice_nlms | 4.3113× | Golomb–Rice as an alternate backend on near-iid residuals |
| 26 | **split_nlms** | **4.3133×** | split-tap NLMS (short lags + taps at the beat period) |

## What the loop discovered (the interesting part)
- **bzip2 wins on ECG because its BWT dictionary-matches the repeating ~385-sample
  heartbeat.** So *higher-order* differencing actually *hurt* (it fragments the
  periodicity) — the loop tried it, measured the regression, and explained why.
- **MIT-BIH is an *arrhythmia* database**, so naive beat-to-beat (period-lag)
  prediction *failed* (3.90× → 2.89×): irregular timing means `s[i-T]` is usually
  the wrong beat. The loop diagnosed this and pivoted to NLMS, which adapts to local
  morphology instead of assuming periodicity.
- **Cross-lead correlation is ~zero** in the delta domain here — the loop added a
  cross-lead NLMS tap, watched best-of-N never select it, and retired the idea.
- **An 11-iteration "0× wall"** (iters 15–25): LPC, cascade-NLMS, beat-templates,
  phase-stable packing — all gate-passed, none promoted, each with a documented
  reason bz2 already captured the structure. Iteration 26 finally broke through with
  split-tap NLMS reaching back a full beat period. The lab notebook
  (`examples/compression-ecg/LAB_NOTEBOOK.md`) reads like a real research log.

> Note: the loop was **still climbing when the 24-iteration budget ran out** (it
> promoted on the final iteration), so this isn't a converged optimum — more budget
> would likely push further toward the ~4.5–5× literature ceiling.

## Verify it yourself
```bash
cd examples/compression-ecg
./verify.sh        # fetches the public data (sha256-pinned), evaluates the
                   # champion + bzip2 → prints 4.3133x vs 3.8489x (+12.1%)
```
Frozen run artifacts (dashboard, leaderboard, key results):
`examples/compression-ecg/results/`.
