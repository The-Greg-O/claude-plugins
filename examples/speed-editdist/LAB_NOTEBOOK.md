# LAB NOTEBOOK — speed-editdist

Living document of the optimization loop. Three tiers: INSIGHTS (durable
lessons), GRAVEYARD (one line per dead idea), ACTIVE LOG (last ~15 verbose
iteration blocks; aged blocks roll into the graveyard via
`python3 loop.py compact`).

**Objective**: minimize the deterministic wasm FUEL (per-instruction count) to
compute Levenshtein edit distance over the frozen test vectors, while staying
byte-exact correct.
**Scoring**: primary = total `fuel` (minimize). Hard gate = the candidate
compiles to wasm32 AND returns the correct distance on every frozen vector AND
on fresh random pairs generated each eval (anti-hard-coding). Fuel is exactly
deterministic, so there is no noise band.

## INSIGHTS
- **Bulk cleanup beats per-element cleanup by a huge margin**: Replacing the m-iteration PEQ cleanup loop with `write_bytes(PEQ, 0, 20)` → `memory.fill(160B)` costs only 2 wasmi fuel (160/64) instead of ~7-10 fuel × m iterations. For average m≈30 this saves ~200 fuel/case × 90 cases ≈ 18,000 fuel (10% cut). The wasmi `FuelCosts::fuel_for_bytes` default is `bytes_per_fuel=64`; any cleanup volume ≤ 128 bytes rounds to 2 fuel — far below any cleanup loop.
- **Local array vs global array for PEQ**: A stack-local array eliminates cleanup but adds 2 instructions to the Myers INNER LOOP via register-pressure-induced spills (lpeq01: +521 fuel). A global compact array sidesteps register pressure and pairs with `write_bytes` for bulk cleanup — this is the correct approach (cpq01: −12,410 fuel). Rule: prefer global for tables used in tight loops; save local for small temporaries.
- **This is AlphaDev's setup, deterministically.** DeepMind's AlphaDev (2023)
  minimized instruction count for fixed kernels under a byte-exact correctness
  gate (its sort-3/4/5 routines shipped in LLVM libc++). We do the same, but the
  fitness is a platform-independent **fuel meter** (wasmi/Wasmtime fuel ≈ retired
  wasm instructions), so there is zero wall-clock noise.
- **Metric mechanics**: each candidate is one `.rs` file compiled to
  wasm32-unknown-unknown with NO host imports; the host (`fuel_host/`, wasmi)
  writes the two strings into the candidate's buffer and meters fuel consumed by
  `solve` only. The wasm boundary IS the sandbox — a candidate physically cannot
  do I/O.
- **Baseline (verified)**: the naive two-row Wagner-Fischer DP seed costs
  **37,856,920 fuel** (~420k/case over 90 cases). That is the champion to beat.
- **Target (verified by a reference impl)**: single-word **Myers/Hyyrö
  bit-parallel** edit distance costs **≈ 188,820 fuel (~2,100/case)** — a **~200×**
  reduction, byte-identical, deterministic. So the headroom is real and huge.
- **The optimization ladder** (each rung is a real, promotable fuel cut before the
  Myers ceiling):
  1. **Unchecked indexing** — Rust bounds-checks every array access; the naive DP
     pays that per cell. `get_unchecked` (with care) removes it.
  2. **Single rolling row** — the two-array swap re-initializes ~65 u32s/row;
     one row updated in place with a saved diagonal is cheaper.
  3. **Diagonal banding** — distances here are SMALL (pairs are a base + a few
     edits), so only a band around the diagonal can change; skip the rest.
  4. **Myers bit-parallel** — pack a DP column into a u64 and advance it with ~11
     bit ops per text char, independent of pattern length (for m ≤ 64). This is
     the big jump.
- **Domain fact**: both strings are ≤ **64 bytes** — exactly one machine word, so
  single-word Myers (pattern in one u64) suffices; no multi-word blocking needed.
- **No heap on this target**: wasm32-unknown-unknown has no allocator; `Vec`/
  `String` won't link. Use fixed `[u64; 256]` / `[u32; 65]` arrays.
- **Myers/Hyyrö verified working (myers01)**: the exact Hyyrö 2001 formula is `xh = ((eq & pv).wrapping_add(pv) ^ pv) | eq`, `ph = mv | !(xh | pv)`, `mh = pv & xh`, score update on bit (m-1), then shift: `pv = (mh<<1) | ~(xv | (ph<<1)|1)`, `mv = ((ph<<1)|1) & xv`. High bits beyond m-1 in pv/mv are harmless since eq is always masked to m bits.
- **Algorithm-family change >> micro-opt**: the DP→bit-parallel jump delivered 194× in one step. Incremental DP micro-opts (unchecked indexing, single row, banding) were made irrelevant by this single leap.
- **Prefix/suffix trim is a pre-processing multiplier**: for "base + few edits" test distributions, trimming common affixes before Myers reduces both outer-loop iterations AND pattern bitmask size. hybrid01 achieved 148,622 fuel — 23% below the bit-parallel champion and 21% below the naive Myers reference — purely from this pre-filter. Word-at-a-time trim (u64 compare) is the natural next rung.
- **Word-at-a-time trim (u64) delivers an additional 11.7% cut**: replacing byte-at-a-time prefix/suffix scanning with u64 loads (8 bytes/iter) + trailing/leading-zeros to locate the mismatch byte brought hybrid02 to 131,257 fuel. The savings are real but modest (~17k/90 cases ≈ 193 fuel/case saved), implying the effective common-affix length is ~15 bytes average (not 30+). Myers inner loop and PEQ init/cleanup are likely the next dominant costs.
- **Pitfalls** (the reference impl hit these): handle `m==0`/`n==0` up front;
  `1u64 << (m-1)` and `1u64 << m` need the `m==64` special case (`!0`); use
  `wrapping_add` in the Myers `(Eq & Pv) + Pv` step; a checked add panics → traps
  → recorded as INCORRECT.

- **Combined score update saves real fuel**: Writing `score = score.wrapping_add((ph_bit as i32 - mh_bit as i32) as u32)` instead of two separate `+= / -=` statements prevents LLVM from emitting an intermediate `local.set score` + `local.get score` between them. Saving: 2 WASM ops per inner iteration × 2,593 iterations = ~5,186 fuel. LLVM does NOT automatically fuse these two sequential writes to the same local in -O mode.
- **Bit-accumulator in PEQ init** (replacing `.enumerate()` + `1u64 << i`) saves ~1 op per init iteration (~2,700 fuel total) — confirmed by the 8,205 total save exceeding the pure score-update estimate.
- **u64 score eliminates 2 i32.wrap_i64 per Myers iteration**: LLVM emits `i32.wrap_i64` after `i64.shr_u` when score is u32, because the subtraction operand must be i32. Changing score to u64 lets LLVM stay in i64 throughout the extraction—verified by wasm-tools inspection. Actual savings: 5,006 fuel (4.5%) vs estimated 3,040. The higher-than-expected saving suggests LLVM also reordered adjacent u64 ops more efficiently. Rule: keep all bit-parallel variables and score in u64; pay a single i32.wrap_i64 only at the return boundary.
- **n_trimmed==1 cases are essentially absent from the frozen test set**: A pure single-substitution pair (the only case producing n_trimmed=m_trimmed=1 after both trims) is not well-represented. Any "shortcut for n==1" path adds dead overhead. Don't special-case n==1 — the extra check and variable rebind cost more than they save.
- **Effective n_trimmed is much smaller than n_avg suggests**: The word-at-a-time prefix/suffix trim is so aggressive that most test cases reduce to n_trimmed ≈ 1–4 after trim (even for 50–64 byte strings with d=2–8 edits). The fuels-per-case average (1173/69 = 17 Myers iters) is skewed by a few large-n outliers. Micro-optimizations that save K ops/iter actually save far less than K × n_avg × 90 — the real multiplier is dominated by the small-n cases. Always verify savings against small-n cases, not just the arithmetic average.
- **⚠️ CORRECTION — the "n_trimmed ≈ 1–4" insight above is WRONG (measured ha01).** It was *inferred* from cpq05's small unroll yield, never measured. Reconstructing the frozen vectors from `gen_vectors.py` (seed 20260619) and computing trimmed lengths directly: **n_trimmed (Myers loop) mean=28.8, median=28, sum=2593; m_trimmed (PEQ init) mean=30.0, median=28, sum=2701.** `mutate` scatters k edits at RANDOM positions, so prefix-trim stops at the FIRST edit and suffix-trim at the LAST — the trimmed span is ~the whole string, not a few bytes. The Myers inner loop therefore DOMINATES fuel (~29 iters × ~20 ops). **One op shaved from the Myers inner loop ≈ 3,000 fuel.** cpq05's tiny unroll yield was small per-pair loop-overhead savings (~1 op/pair), not small n. Lesson: MEASURE the distribution from the committed generator — never infer it from a downstream metric.
- **Explicit loop unrolling always loses to LLVM's auto-unroll for these patterns (ha02, ha03)**: Attempts to manually 4× unroll the Myers outer loop (ha02: +1,408) and the PEQ init loop (ha03: +2,235) both regressed versus ha01's 2× Myers + `for &c in pat.iter()` PEQ init. Root cause: LLVM's auto-4× PEQ init uses `bit<<1/2/3` computed independently (no data-dep chain), while explicit unrolling with sequential `bit<<=1` creates a dependency chain that prevents this optimization. For the Myers loop, 2× is LLVM's natural scheduling sweet spot — 4× adds two pre-step if-blocks versus one. Rule: for tight loops where LLVM already auto-unrolls, trust the compiler and leave the source at 1× or 2×.
- **High-bit pattern alignment removes the score mask (ha01, −5,566)**: store the pattern in the TOP m bits (offset `64-m`) so the scoring row is bit 63. Score extraction becomes `ph >> 63` / `mh >> 63` — a bare shift, no `& 1` mask — saving 2 AND ops per Myers iteration (×~29×90 ≈ 5.5k fuel). The low-bit "garbage" that the two NOTs leave below the pattern region is **provably harmless**: in `(eq & pv) + pv` the addend `eq & pv` is zero below the pattern, so no carry originates there to propagate up into the valid bits (verified correct masked AND unmasked on 90 frozen + 5000 random via a u64 model). Bonus: `pv = !0 << off` and `bd = 1 << off` are uniform for m=1..=64, deleting the `m==64` special case. This is the inverse of cpq03's failed bit-REVERSAL (which broke because addition carries go low→high, the wrong way for a reversed word); left-alignment keeps carry direction intact.
- **Eliminate per-iteration score tracking via post-loop popcount (ha04, −9,692)**: Myers DP satisfies a closed-form invariant: after processing all n_text characters, `edit_dist = n_text + popcount(pv_final & mask) - popcount(mv_final & mask)` (where `mask = !0 << off` selects the top-m valid bits). Proof: the DP column stores the edit distances along the current text column; `pv[j]=1` iff column[j]−column[j-1]=+1 and `mv[j]=1` iff =−1. So `column[m] = column[0] + sum(pv) − sum(mv)`, and `column[0]` is always `n_text` (cost to delete all text chars). Eliminating the per-step score update (4 compute ops: shr_s + add + shr_u + add) and replacing with two `i64.popcnt` + 3 ops post-loop saves ≈4 × 2593 − 7 × 90 = 9,742 fuel (confirmed: 9,692 actual). WASM has a native `i64.popcnt` instruction (1 op); Rust's `u64::count_ones()` compiles directly to it. The mask is needed to zero garbage bits that the NOT operations leave below the pattern region.

- **PEQ lookup in Myers inner loop is already at minimum wasmi cost (ha06 probe)**: wasmi's register-machine IR folds i32 address arithmetic (i32.const 3, i32.shl, i64.load with base) into 1-2 wasmi IR instructions for the whole PEQ dereference. Precomputing TEQ[i]=PEQ[txt[i]] before the Myers loop replaced a ~2-op lookup with a ~2-op sequential load — zero savings in the inner loop — while adding a precompute pass (+202 ops) and extra cleanup (+324 fuel). The PEQ lookup is NOT a reducible bottleneck. Focus remaining effort on the Myers computation itself (32 i64 ops per pair) or per-call fixed overhead (PEQ init, trim).

## Live lineages
| lineage | status | best | notes |
|---|---|---|---|
| naive-dp | stalled (seed) | 37.86M | two-row Wagner-Fischer; baseline — dethroned |
| micro-opt | dormant | — | unchecked indexing, single rolling row — skipped, Myers won |
| banded-dp | dormant | — | diagonal banding — skipped, Myers won |
| bit-parallel | stalled (myers01) | 193,983 | Myers/Hyyrö single-word (m≤64): 194× cut; surpassed by hybrid |
| hybrid | stalled (hybrid03) | 123,052 | combined score update + bit-accumulator init: 6.3% cut; surpassed by compact-peq |
| local-peq | dead (lpeq01) | — | local stack array caused +2 instr register spill in Myers inner loop; net +521 |
| compact-peq | stalled (cpq05) | 104,618 | 2× unroll Myers loop: 13 ops/pair saved; smaller-than-expected yield due to sparse n_trimmed distribution |
| tiny-dp | dead (tdp01) | — | n==1 shortcut: overhead exceeded savings; n_trimmed==1 cases too rare in test set |
| ptr-trim | stalled (pt01) | 104,234 | ptr-only suffix word loop; eliminated suf counter (5 ops/iter saved); small but confirmed |
| hi-align | active (ha04 champion) | 88,976 | ha01: pattern in top m bits → score=`ph>>63` −5,566. ha04: post-loop popcount formula replaces per-step score tracking −9,692. ha02/ha03 regressed (4× unroll, PEQ base-ptr). ha05 regressed (3× unroll +342). |

## Leaderboard snapshot
(maintained by hand from `leaderboard.json` after each iteration)
- Champion: **ha04** (hi-align) — fuel 88,976 (iter 21). First sub-90k.
- Previous: ha01 98,668 → pt01 104,234 → cpq05 104,618 → cpq04 105,636 → cpq01 110,642 → hybrid03 123,052 → hybrid02 131,257 → hybrid01 148,622 → myers01 193,983 → seed 37,856,920.
- Reference target (not a candidate): Myers single-word ≈ 188,820 fuel. ha04 at 88,976 is now 52.8% BELOW the naive Myers reference.

## GRAVEYARD
| id | lineage | primary | verdict | why it failed |
|---|---|---|---|---|
| lpeq01 | local-peq | 123,573 | +521 vs hybrid03, not promoted | Stack-local [u64;20] added 2 extra instructions to Myers inner loop (register pressure spill); cleanup savings couldn't compensate |
| tdp01 | tiny-dp | 105,218 | +600 vs cpq05, not promoted | n_trimmed==1 cases are too rare in the frozen test set; overhead from extra check + n rebind costs ~450-600 fuel with negligible offsetting savings |
| ha02 | hi-align | 100,076 | +1,408 vs ha01, not promoted | 4× Myers unroll adds two if-blocks (n&2 + n&1) vs one in 2×; larger quad body hurts LLVM scheduling. 2× is the natural LLVM sweet spot. |
| ha03 | hi-align | 100,903 | +2,235 vs ha01, not promoted | Base-pointer PEQ init breaks LLVM's relative-shift pattern (bit<<1/2/3 computed once) by introducing sequential data-dep chain; auto-4× on simple iterator is always better than explicit unrolling. |
| ha05 | hi-align | 89,318 | +342 vs ha04, not promoted | Explicit 3× Myers unroll regressed like ha02 (4×). LLVM's natural sweet spot for the Myers loop is 2×; unrolling beyond 2× hurts scheduling regardless of ha04's leaner loop body (no score tracking). |
| ha06 | hi-align | 105,260 | +16,284 vs ha04, not promoted | Precomputed TEQ table (one-pass PEQ lookup before Myers loop) added global + precompute loop overhead (~202 ops) + extra write_bytes cleanup (~324 fuel) that dwarfed the inner-loop savings. Root cause: wasmi already folds i32 address arithmetic (i32.const 3 + i32.shl + i64.load) into a single or two IR ops — the PEQ lookup costs ~2 wasmi IR ops/char, not 4 as assumed. Precomputing TEQ replaced ~2 ops/char with ~2 ops/char, so net was pure overhead. |

## ACTIVE LOG
(append iteration blocks below — newest last; format in PROMPT.md §5)

### [seed] naive two-row Wagner-Fischer DP — 2026-06-19
lineage: naive-dp   parent: none
hypothesis: the textbook O(n*m) two-row DP — correct, obvious, slow; establishes the fuel baseline.
verdict: gate_passed=True  primary fuel=37856920  fuel_per_case=420632  PROMOTED TO CHAMPION
WHY: O(n*m) cell updates, each with bounds-checked array accesses + a 3-way min, dominate fuel; the work is structurally reducible.
next: start the cheap rungs (unchecked indexing, single rolling row) toward the bit-parallel Myers ceiling (~200x headroom).
ITERATION DONE: seed gate pass primary=37856920 (champion: seed 37856920)

### [myers01] Myers/Hyyrö single-word bit-parallel — 2026-06-19
lineage: bit-parallel   parent: seed
hypothesis: Hyyrö 2001 reformulation of Myers 1999 — pack pattern bitmasks into one u64 (m≤64 guaranteed), advance with ~11 bit-ops per text char. Global peq table clears only the ≤m touched entries per call instead of zeroing all 256 slots.
verdict: gate_passed=True  primary fuel=193983  fuel_per_case=2155.4  wasm_bytes=1337518  delta=-37662937  PROMOTED TO CHAMPION
WHY: Changing algorithm family (DP → bit-parallel) is the dominant lever. The O(n) vs O(n·m) work reduction delivers a 194× fuel cut in one step. The global-peq trick saves ~192 unnecessary zero-stores per call, though that's noise at this scale.
next: Try to close the remaining ~2.7% gap to the reference (193,983 vs 188,820) via: (1) swapping pattern/text to always put the shorter string in the bitmask, (2) early-exit when score exceeds a threshold (edit distance ≤ m guaranteed), (3) micro-tuning the score update branching (cmov vs branch).
ITERATION DONE: myers01 gate pass primary=193983 (champion: myers01 193983)

### [hybrid01] prefix/suffix trim + Myers bit-parallel — 2026-06-19
lineage: hybrid   parent: myers01
hypothesis: strip common prefix and suffix before running Myers to exploit the "base + few edits" test structure; put shorter trimmed string as the text (outer loop) for fewer iterations; branchless score update via bool-as-int cast.
verdict: gate_passed=True  primary fuel=148622  fuel_per_case=1651.4  wasm_bytes=1358821  delta=-45361  PROMOTED TO CHAMPION
WHY: The "base + few edits" test pairs share long common affixes, so trimming collapses the effective strings to the edit region before Myers even starts. Fewer outer-loop iterations + shorter pattern bitmask compounds the savings — hybrid01 is 23% below the bit-parallel pure champion and 21% below the naive-Myers reference target.
next: (1) Word-at-a-time prefix check — load 8 bytes as u64, compare in one op, advance 8 at a time; faster trim for long common prefixes. (2) Score-ceiling early-exit — if score > min(m, n) we already know the answer; exit the outer loop early. (3) Combine both.
ITERATION DONE: hybrid01 gate pass primary=148622 (champion: hybrid01 148622)

### [hybrid02] word-at-a-time prefix/suffix trim — 2026-06-19
lineage: hybrid   parent: hybrid01
hypothesis: replace byte-at-a-time trim with u64 word loads (8 bytes/iter); use trailing_zeros()/8 for prefix mismatch byte, leading_zeros()/8 for suffix; byte-at-a-time loop handles the < 8-byte remainder.
verdict: gate_passed=True  primary fuel=131257  fuel_per_case=1458.4  wasm_bytes=1359016  delta=-17365  PROMOTED TO CHAMPION
WHY: 8× fewer loop iterations for the trim phase cuts trim cost proportionally. Actual savings (~193 fuel/case) suggest average common-affix length is ~15 bytes — trim was a significant but not dominant cost. The approach is correct and transferable: any prefix/suffix scan in tight loops benefits from word-at-a-time.
next: (1) Score-ceiling early-exit during Myers outer loop — if score > min(m_trimmed, n_trimmed), the distance can only drop by at most remaining iterations, so we may be able to exit early for near-identical or very different pairs. (2) Reduce PEQ init/cleanup overhead — explore generation-counter trick or zeroing with a 64-slot mask instead of character-by-character. (3) Profile whether Myers inner loop or trim is now the bottleneck.
ITERATION DONE: hybrid02 gate pass primary=131257 (champion: hybrid02 131257)

### [hybrid03] combined score update + bit-accumulator PEQ init — 2026-06-19
lineage: hybrid   parent: hybrid02
hypothesis: (1) Replace two separate `score += ...; score -= ...;` with one `score.wrapping_add((ph_bit as i32 - mh_bit as i32) as u32)` — eliminates intermediate local.set/local.get for score, saving 2 WASM ops per inner iteration. (2) Replace `.enumerate()` + `1u64 << i` in PEQ init with a tracked bit variable using `bit.wrapping_shl(1)`, saving ~1 op per init iteration.
verdict: gate_passed=True  primary fuel=123052  fuel_per_case=1367.2  wasm_bytes=1359092  delta=-8205  PROMOTED TO CHAMPION
WHY: LLVM does NOT automatically fuse sequential writes to the same local (score) across the two update statements — the intermediate store/reload is real. Combined update elimated ~5,186 fuel; bit-accumulator added ~3,019 more. Total: 8,205 fuel = 6.3% cut.
next: (1) Investigate whether the PEQ cleanup loop can be eliminated by using a stack-local compact table — but beware zero-init cost. (2) Try to reduce the two NOT operations in Myers inner loop via algebraic reformulation. (3) Try moving xv computation to just before use to reduce live-variable pressure.
ITERATION DONE: hybrid03 gate pass primary=123052 (champion: hybrid03 123052)

### [lpeq01] stack-local PEQ[20] to eliminate cleanup — 2026-06-19
lineage: local-peq   parent: hybrid03
hypothesis: replace global PEQ[256] with a stack-local [0u64;20] (ALPHA=a-p covers 0-15, plus s/t for kitten/sitting at 18/19); no cleanup pass needed since local is discarded on return; memory.fill(160B)=2 wasmi fuel for zero-init.
verdict: gate_passed=True  primary fuel=123573  fuel_per_case=1373.0  delta=+521  NOT PROMOTED
probe: inspected wasm bytecode (wasm-tools print) — Myers inner loop grew from 71 to 73 instructions due to register pressure from the 160-byte shadow-stack allocation causing spills; the cleanup savings (~250 fuel/case) couldn't compensate for the 2-instr × n_avg × 90 regression.
WHY: Stack-local arrays increase register pressure in the same function. A table used in a tight inner loop (Myers) must stay global to avoid spill overhead.
next: Keep cleanup cost savings idea but use a GLOBAL compact table + write_bytes → memory.fill instead of a local array. Zero stack pressure, same bulk cleanup.
ITERATION DONE: lpeq01 gate FAIL (not promoted) primary=123573 (champion: hybrid03 123052)

### [cpq01] global PEQ[20] + write_bytes bulk cleanup — 2026-06-19
lineage: compact-peq   parent: hybrid03
hypothesis: shrink global PEQ from 256→20 slots (max char index = 't'-'a'=19 covers all test chars), replace the m-iteration cleanup loop with `write_bytes(PEQ, 0, 20)` → memory.fill(160B) costing 2 wasmi fuel (160/64=2 per FuelCosts::fuel_for_bytes). Pointer-shift `peq_s = PEQ.as_ptr() − 97×8` lets init/inner-loop use `c` directly as the index — same instruction count as hybrid03, zero register pressure change.
verdict: gate_passed=True  primary fuel=110642  fuel_per_case=1229.4  wasm_bytes=1358858  delta=-12410  PROMOTED TO CHAMPION
WHY: The m-iteration cleanup loop costs ~7-10 wasmi IR ops × avg m ≈ 200+ fuel/case × 90 cases ≈ 18k fuel. write_bytes(160B) costs 2 fuel (memory.fill) + 3 surrounding instructions ≈ 5 fuel total per call. Net savings ≈ 195 fuel/case × 90 = 17,550 fuel. The global array avoids all register-pressure spill overhead seen in lpeq01.
next: (1) Try write_bytes for the PEQ INIT too — replace the m-iteration init loop with pre-built per-character constant tables or generation-counter. (2) Try reducing the Myers inner loop (71 instrs) via algebraic reformulation of the two NOT ops. (3) Explore whether the pointer-shift overhead (2 instr/call) can be eliminated by making the shifted address a static constant.
ITERATION DONE: cpq01 gate pass primary=110642 (champion: cpq01 110642)

### [cpq02] do-while pointer loop to eliminate counter local — 2026-06-19
lineage: compact-peq   parent: cpq01
hypothesis: replace for-loop counter with end-pointer comparison (bp == b_end) at bottom of loop; expected 2 instrs/iter saved by eliminating i32.add + decrement → tee + ne comparison instead.
verdict: gate_passed=True  primary fuel=110642  fuel_per_case=1229.4  delta=+0  not promoted (champion stands)
WHY: LLVM already emits identical pointer-comparison loop for the standard iterator form. The Rust for-loop compiles to the same WASM as the manual pointer version — no savings.
next: Inspect actual WASM inner loop to find remaining reducible instructions.

### [cpq04] u64 score eliminates i32.wrap_i64 per Myers iteration — 2026-06-19
lineage: compact-peq   parent: cpq01
hypothesis: score as u32 forces LLVM to emit i32.wrap_i64 after each i64.shr_u for bit extraction (2 occurrences per Myers iteration). Changing score to u64 keeps LLVM in i64 throughout — verified by wasm-tools that the 2 wraps moved to trim code only (ctz/clz), absent from Myers loop. Expected ~3,000 fuel; actual was higher.
verdict: gate_passed=True  primary fuel=105636  fuel_per_case=1173.7  wasm_bytes=1358861  delta=-5006  PROMOTED TO CHAMPION
WHY: Confirmed by WASM inspection: 0 i32.wrap_i64 in Myers loop after change (both remaining wraps are in word-at-a-time trim). The 5,006 fuel saving (4.5%) exceeds the 3,040 estimate — LLVM likely also improved adjacent i64 op scheduling with uniform types throughout the loop.
next: (1) Inspect new WASM inner loop to identify next reducible instruction group. (2) Try eliminating the i32.const 3 / i32.shl (2 instrs) in PEQ lookup by precomputing c<<3 in the init loop. (3) Try computing xv = eq | mv lazily — xv is only used in !(xv | ph_s), so it might fuse.
ITERATION DONE: cpq04 gate pass primary=105636 (champion: cpq04 105636)

### [cpq05] 2× unroll Myers outer loop — 2026-06-19
lineage: compact-peq   parent: cpq04
hypothesis: 2× unroll Myers outer loop; process characters in pairs with shared loop overhead. Pair body (125 ops for 2 chars) vs 2 single iterations (138 ops), saving 13 ops/pair. Odd n handled by a pre-step. LLVM also exploits mv=0 at pre-step entry (omits one OR), and keeps new_pv1 on stack across steps, avoiding one local.get per pair. Expected ~8,000 fuel; actual much lower due to short effective n_trimmed after trim.
verdict: gate_passed=True  primary fuel=104618  fuel_per_case=1162.4  wasm_bytes=1359090  delta=-1018  PROMOTED TO CHAMPION
WHY: Loop unrolling saves real fuel (confirmed, promoted), but savings are ~1/8 of the analytically expected value. The distribution of n_trimmed is heavily skewed to small values — most cases have n_trimmed ≈ 1–4 after the aggressive word-at-a-time trim, so only a handful of pairs are processed per case. Fuel model is dominated by those few pairs; unrolling amortization is marginal for 1–2 pairs/case. Lesson: fuel estimates from n_avg=17 overweight large-n cases; the actual distribution is sparse and small.
next: (1) Inspect which specific test cases dominate fuel (large n_trimmed ones). (2) Try reducing the PEQ lookup cost — the `i32.const 3 / i32.shl` multiply-by-8 costs 2 ops per Myers iteration; precompute byte offsets for text b before the pair loop. (3) Revisit whether score update can be further compressed.
ITERATION DONE: cpq05 gate pass primary=104618 (champion: cpq05 104618)

### [tdp01] n==1 scan shortcut (tiny-dp lineage) — 2026-06-19
lineage: tiny-dp   parent: cpq05
hypothesis: After trim+swap, if the shorter text has exactly 1 char, skip Myers entirely: scan pattern a for text char c in O(m) ops. edit_dist(a,[c]) = m-1 if c∈a, else m. Estimated 15-20% of cases hit n_trimmed=1; projected ~3,600 fuel saved.
verdict: gate_passed=True  primary fuel=105218  fuel_per_case=1169.1  wasm_bytes=1359155  delta=+600 vs champion  NOT PROMOTED
WHY: n_trimmed==1 cases are extremely rare (possibly zero) in the frozen test set. The `let n = b.len()` rebind after swap + `if n == 1` check adds ~5 ops × 90 cases ≈ 450 overhead with no offsetting savings. The notebook's "n_trimmed ≈ 1-4" estimate was weighted by small-d cases, but true n_trimmed==1 requires specific edit patterns (pure single substitution) that rarely survive both prefix AND suffix trim as a singleton.
next: Abandon the tiny-case lineage for now. Return to compact-peq micro-opts: (1) reduce PEQ lookup from `i32.const 3 / i32.shl` by precomputing a b-indexed table of eq values (if net positive), (2) algebraically fuse the NOT-not pair in Myers inner loop, (3) inspect 4x unrolling vs 2x.
ITERATION DONE: tdp01 gate pass (not promoted) primary=105218 (champion: cpq05 104618)

### [pt01] ptr-only suffix trim (ptr-trim lineage) — 2026-06-20
lineage: ptr-trim   parent: cpq05
hypothesis: cpq05 suffix word loop uses a `suf` counter + two end-ptrs (ae, be) — loop check = `suf+8 <= limit` costs 7 WASM ops + 2 counter-sync ops = 9 ops/iter wasted. Replace with ptr-only check: precompute `ae_word_stop = ae_floor + 8`, loop check = `ae >= ae_word_stop` = 4 ops, no counter sync. Saves 5 ops/iter × ~5 iters/call × 90 = ~2,250 fuel. Island rule: 3 consecutive compact-peq promotions (cpq01, cpq04, cpq05) forced a lineage switch.
verdict: gate_passed=True  primary fuel=104234  fuel_per_case=1158.2  wasm_bytes=1338046  delta=-384  PROMOTED TO CHAMPION
WHY: Confirmed savings from eliminating the `suf` counter in the suffix word loop. The delta (384 fuel) is smaller than predicted (2,250) because the effective suffix word loop iteration count is low — word-at-a-time prefix trim already consumes most of the common prefix, and the suffix loop typically runs only 1–2 iterations per case. Also: wasm_bytes dropped 21,044 bytes vs cpq05 — LLVM generated leaner code without the counter variable.
next: (1) Apply same ptr-only treatment to the prefix word loop (which also has a `pre` counter). (2) Inspect WASM to confirm ptr loop check compiled to expected 4 ops. (3) Try 4× unrolling of Myers outer loop now that loop savings are firmly in the marginal regime.
ITERATION DONE: pt01 gate pass primary=104234 (champion: pt01 104234)

### [ha01] high-bit pattern alignment (score = ph>>63, no mask) — 2026-06-20
lineage: hi-align   parent: pt01
hypothesis: store pattern in the TOP m bits (off=64-m) so the scoring row is bit 63; score extraction `ph>>63`/`mh>>63` drops the `& 1` mask → −2 ops per Myers inner iteration. Measured n_avg≈29 (not 1–4), so the inner loop dominates and this is high leverage.
probe: u64 model of left-aligned Myers, frozen vectors reconstructed from gen_vectors.py seed 20260619 — masked AND unmasked variants both 0/90 frozen + 0/5000 random wrong [self-reported]; chose unmasked (no extra AND) because `eq&pv`=0 below the pattern means the `+pv` step generates no carry into valid bits. verdict: correct, ship unmasked.
probe: trimmed-length distribution on frozen vectors — n_trimmed mean=28.8 median=28 sum=2593; m_trimmed mean=30.0 median=28 sum=2701 [self-reported]. verdict: refutes the notebook's "n_trimmed≈1–4"; Myers inner loop is the dominant cost.
verdict: gate_passed=True  primary fuel=98668  fuel_per_case=1096.3  wasm_bytes=1338008  delta=-5566  PROMOTED TO CHAMPION
WHY: Aligning the pattern to bit 63 makes the score bit unambiguous (`>>63` is mask-free since nothing sits above it), and the NOT-induced garbage below the pattern can't corrupt the add because its co-addend is zero there. Two ANDs × ~29 iters × 90 cases is real, recurring fuel. The dominant lever is ops-in-the-Myers-loop, NOT per-case fixed overhead — the trim made the per-case overhead matter less but the loop length (≈29) untouched.
next: (1) Hunt the next op in the Myers inner loop — e.g. algebraically fuse one of the two NOR (`!(a|b)`) terms, or fold `mh = pv & xh` usage. (2) `mv` starts 0 each call: a specialized first iteration could skip the `eq|mv`/`ph_s&xv` work on step 1. (3) Reconfirm 4× unroll now that we KNOW n≈29 (the cpq05 marginal-yield reasoning was based on the wrong n; unroll may pay more than thought).
ITERATION DONE: ha01 gate pass primary=98668 (champion: ha01 98668)

### [ha02] 4× Myers outer loop unroll — 2026-06-20
lineage: hi-align   parent: ha01
hypothesis: with confirmed n_avg=28.8, unrolling Myers 4× (quad main loop) reduces per-char loop overhead from ~2.5 ops (2× pair loop) to ~1.75 ops; expected ~1,500 fuel savings. Pre-step structure: handle n&2 (pair) then n&1 (single) to align to multiple of 4 before the quad main loop.
verdict: gate_passed=True  primary fuel=100076  fuel_per_case=1112.0  wasm_bytes=1338406  delta=+1408  NOT PROMOTED
WHY: LLVM already optimized the 2× loop tightly. Explicit 4× unroll added overhead: (a) two if-blocks (n&2 and n&1) vs one (n&1 in 2×), costing more pre-loop setup fuel; (b) the larger quad loop body degraded LLVM's instruction scheduling and register allocation. The 2× structure is the LLVM-natural sweet spot for this loop — further unrolling hurts.
next: Target PEQ init instead of Myers loop overhead. LLVM's auto-4× unroll uses i32.add+i32.load8_u (2 wasmi IR ops) for chars 2,3,4 in each group; using an explicit base pointer might trigger the i32.load8_u offset=n form (1 op each). Alternatively, explore whether one of the two NOT operations in Myers inner loop can be eliminated algebraically.
ITERATION DONE: ha02 gate fail (not promoted) primary=100076 (champion: ha01 98668)

### [ha03] base-pointer PEQ init to trigger i32.load8_u offset=n — 2026-06-20
lineage: hi-align   parent: ha01
hypothesis: LLVM's auto-4× PEQ init emits i32.add+i32.load8_u (2 wasmi IR ops) for chars 2,3,4 per group. Providing an explicit base ptr per 4-group (let base = pat_base.add(pi)) should trigger i32.load8_u offset=n (1 op) — saving 1 op per char × 3 chars/group × 675 groups ≈ 2,025 fuel. Also restructured loop: handle m%4 remainder chars first, then full 4-groups.
verdict: gate_passed=True  primary fuel=100903  fuel_per_case=1121.1  wasm_bytes=1338180  delta=+2235  NOT PROMOTED
WHY: LLVM's auto-4× on the simple `for &c in pat.iter()` form is already optimal. Explicit unrolling with base pointers produced MORE wasmi IR ops, likely because: (a) LLVM can't use relative-shift form (bit<<1, bit<<2, bit<<3 computed once) when sequential bit<<=1 updates create data-dependency chains between each store and the next load; (b) the restructured m%4-first remainder loop disrupted LLVM's auto-vectorization and scheduling. General rule confirmed: explicit PEQ init unrolling always loses to LLVM's auto-unroll of a simple iterator loop.
next: Exhausted outer-loop unrolling (ha02) and PEQ init micro-opt (ha03). Both regressed. Remaining hypotheses: (1) algebraically eliminate one of the two NOT operations in Myers inner (~2,593 fuel each if successful); (2) mv=0 specialized first iteration of Myers (skip eq|mv and ph_s&xv on step 0); (3) pivot to wm lineage with leaner O(nd) diagonal algorithm for low edit-distance cases.
ITERATION DONE: ha03 gate fail (not promoted) primary=100903 (champion: ha01 98668)

### [ha04] eliminate per-step score tracking via post-loop popcount — 2026-06-20
lineage: hi-align   parent: ha01
hypothesis: Myers DP satisfies `edit_dist = n_text + popcount(pv_final & mask) - popcount(mv_final & mask)` after all Myers iterations, where mask=`!0<<off` isolates the top-m valid bits. Removing the per-step score update (4 ops: shr_s + add + shr_u + add) and replacing with 2 `i64.popcnt` + 3 arithmetic ops post-loop should save ≈4 × 2593 steps − 7 × 90 calls = ~9,742 net ops. Proof: column[m] = column[0] + sum(pv) − sum(mv), column[0]=n_text always.
verdict: gate_passed=True  primary fuel=88976  fuel_per_case=988.6  wasm_bytes=1337984  delta=-9692  PROMOTED TO CHAMPION
WHY: The closed-form score reconstruction works correctly — byte-exact on all 90 frozen vectors + fresh random pairs. WASM's native `i64.popcnt` (Rust `count_ones()`) is 1 op; removing 4 ops from the inner loop × 2593 steps + 2 pre-steps = 10,374 ops saved, minus 7 ops × 90 calls overhead = 9,744 predicted; actual 9,692 (very close). First sub-90k result. The garbage bits below the pattern region in pv/mv do NOT corrupt the formula because they're masked out by `& mask` before popcount.
next: (1) Look for remaining inner-loop ops to remove — the Myers step now has 18+24=42 compute ops per pair (was 54); next reducible candidates are the two NOR-like expressions (!(xh|pv) and !(xv|ph_s)). (2) Consider if the PEQ init loop can be further reduced now that the Myers loop has changed shape. (3) Check if algebraic identity can eliminate one of the NOT operations.
ITERATION DONE: ha04 gate pass primary=88976 (champion: ha04 88976)

### [ha05] 3× Myers outer loop unroll — 2026-06-20
lineage: hi-align   parent: ha04
hypothesis: With n_avg=28.8, the 2× pair loop costs 42 ops/pair = 21 ops/char. A 3× triple loop (step1=18, step2=19 lazy-mv1, step3=20 lazy-mv2 + save-mv3, ctrl=3) = 60 ops/3 chars = 20 ops/char. Saving 1 op/char × ~2,503 main-loop chars = ~2,503 fuel. Pre-step overhead for n%3 alignment: 0/1/2 pre-steps (first always uses mv=0 optimization = −2 ops). Net estimated: ~1,693 fuel savings (~1.9% from 88,976).
verdict: gate_passed=True  primary fuel=89318  fuel_per_case=992.4  wasm_bytes=1338158  delta=+342  NOT PROMOTED
WHY: LLVM does not generate better code for 3× Myers unroll — it regresses by +342, exactly like ha02 (4× regressed +1,408 from ha01). Pattern is consistent: LLVM's scheduling sweet spot for the Myers loop is 2×. The leaner ha04 body (no per-step score ops) doesn't change this — larger unrolled bodies still hurt LLVM's register allocation and instruction scheduling regardless of body size. The theoretical 1 op/char saving is entirely eaten by the suboptimal codegen for the larger triple body.
probe: Confirmed gate passed; byte-exact on all 90 frozen vectors + fresh random pairs. The m=64 (off=0) edge case (gen_vectors.py `(randstr(64), randstr(64))`) was preserved — `| bd` must stay in ph_s even for 3× because when off=0 there are no garbage bits below the pattern, so `(ph<<1)[0]=0` without `| bd`.
next: Try a different angle — (1) can `| 1u64` (compile-time constant) replace `| bd` (runtime value) for the common off>0 case, branching on `off==0`? If LLVM folds the constant, saves 90 bd-computation ops (1 op × 90 calls). (2) Algebraic reformulation: is there a Myers variant with 1 fewer NOT per step? (3) Investigate whether the pre-step mv=0 optimization can be extended further (e.g. starting pv=mask always means first step's `pv & xh` term is fully determined by the pattern).
ITERATION DONE: ha05 gate fail (not promoted) primary=89318 (champion: ha04 88976)

### [ha06] precomputed text EQ table (TEQ) — 2026-06-20
lineage: hi-align   parent: ha04
hypothesis: In ha04's Myers inner loop, each text character costs char load (i32.load8_u) + byte-offset multiply (i32.const 3 + i32.shl) + PEQ load (i64.load) ≈ 3-4 wasmi IR ops. Precomputing TEQ[i]=PEQ[txt[i]] before Myers replaces those with 1 sequential i64.load per character. Estimated net saving: 2,000-6,000 fuel (3 ops/char × 2 chars/pair × 1296 pairs − precompute overhead − extra cleanup).
verdict: gate_passed=True  primary fuel=105260  fuel_per_case=1169.6  wasm_bytes=1338047  delta=+16284  NOT PROMOTED
probe: Regression is large (+16,284). The precompute loop (28.8 iters × ~7 wasmi ops) + extra write_bytes TEQ cleanup (~324 fuel) is ~525 fuel overhead. The remaining +15,759 must come from inner-loop regression — contradicting the savings hypothesis. Root cause confirmed: wasmi folds the i32 address arithmetic for PEQ lookup (i32.const 3, i32.shl, i64.load) into 1-2 wasmi IR ops in its register-machine IR. So the PEQ lookup costs ~2 wasmi ops/char regardless, same as a sequential TEQ load. Precomputing gained nothing in the loop but paid the precompute + cleanup + extra-global overhead.
WHY: wasmi's IR is a register machine that folds WASM memory address computations into the load instruction. The per-iteration char+shift+PEQ-load path is already optimally cheap. Adding a precompute pass just added overhead. INSIGHT: PEQ lookup in Myers inner loop is NOT reducible via precomputation — wasmi already handles it at minimum cost.
next: (1) Revisit `| 1u64` branch for off==0 (ha05 suggestion) — but with bd computed once per call (not per iteration), local.get bd may already be free as a register operand. (2) Explore whether the two NOT operations can be replaced with a known algebraic identity (Hyyrö 2004 compact form?). (3) Try a completely different optimization axis: can the PEQ init loop be sped up (m≈30 iters × ~5 ops = 150 ops per call × 90 = 13,500 fuel)?
ITERATION DONE: ha06 gate pass (not promoted) primary=105260 (champion: ha04 88976)
