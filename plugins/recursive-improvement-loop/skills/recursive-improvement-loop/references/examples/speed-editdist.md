# Worked example — evolving a faster Rust algorithm (edit distance)

**Result:** a recursive-improvement loop drove a Rust edit-distance routine from a
naive **37,856,920**-instruction baseline down to **88,976** — a **425×** reduction
— across 10 promotions, by **rediscovering Myers bit-parallelism on its own** and
then beating the textbook algorithm. Cost is measured as deterministic **Wasmtime/
wasmi fuel** (retired wasm instructions), so the number is bit-for-bit reproducible
on any machine. Every champion stays byte-exact correct.

This is **[AlphaDev](https://www.nature.com/articles/s41586-023-06004-9)'s setup**
— minimize instruction count for a fixed kernel under a byte-exact correctness gate
— but with a deterministic fuel meter instead of noisy hardware latency.

Runnable experiment: [`examples/speed-editdist/`](../../../../../../examples/speed-editdist).

## Problem
- **Objective:** minimize total fuel to compute Levenshtein edit distance over a
  frozen set of 90 string pairs (≤ 64 bytes each).
- **Primary metric:** total wasmi fuel (deterministic). **Minimize.**
- **Gate:** the candidate compiles to `wasm32` **and** returns the correct distance
  on every frozen vector **and** on fresh random pairs generated each eval
  (anti-hard-coding). The wasm sandbox (no host imports) *is* the no-peeking
  guarantee — candidates can only compute.
- **Seed:** the textbook two-row Wagner-Fischer DP — correct, obvious, slow.

## The run (10 promotions)
| champion | fuel | the move |
|---|---|---|
| seed | 37,856,920 | naive O(n·m) DP |
| **myers01** | **193,983** | **Myers/Hyyrö bit-parallel — a 194× cut in one leap** |
| hybrid01–02 | 148,622 → 131,257 | trim common prefix/suffix (then word-at-a-time u64 scan) |
| hybrid03 | 123,052 | fuse the score update; bit-accumulator PEQ init |
| cpq/pt01 | 110,642 → 104,234 | compact PEQ, 2× unroll, pointer-only suffix loop |
| ha01 | 98,668 | **pattern in the top bits** → score = `ph>>63` (drops the mask) |
| **ha04** | **88,976** | **post-loop popcount invariant — deletes per-step score tracking** |

## What the loop discovered (the interesting part)
- **The algorithm-family jump dominates everything.** DP → bit-parallel was 194× in
  a single step; the loop correctly noted that the incremental DP micro-opts it had
  queued (unchecked indexing, single row, banding) were now irrelevant.
- **It beat the textbook.** The reference single-word Myers impl costs ~188,820 fuel;
  the loop's champion is **52.9% below that**, via prefix/suffix trimming (the test
  pairs share long common affixes) plus WASM-level micro-optimization.
- **It reasoned at the instruction level.** It eliminated a per-iteration
  `i32.wrap_i64` by keeping the score in `u64`; learned that **LLVM's auto-unroll
  beats manual unrolling** here (4× and 3× both regressed); and confirmed via
  wasm-tools that the PEQ lookup was already at minimum cost.
- **It found a closed form.** `ha04` replaced per-step score tracking with the
  invariant `edit_dist = n + popcount(pv) − popcount(mv)` — derived from the DP
  column semantics and compiled to two native `i64.popcnt` instructions. That's the
  loop discovering *new* math about the algorithm, not just porting it.
- It even **caught and corrected one of its own wrong insights** (an inferred input
  distribution it later measured directly from the committed generator and retracted).
  See `examples/speed-editdist/LAB_NOTEBOOK.md`.

## Verify it yourself
```bash
cd examples/speed-editdist
./verify.sh        # builds the wasmi host, compiles champion + naive seed to
                   # wasm, prints 88,976 vs 37,856,920 fuel = 425.5x
```
Frozen run artifacts (dashboard, leaderboard, key results):
`examples/speed-editdist/results/`.
