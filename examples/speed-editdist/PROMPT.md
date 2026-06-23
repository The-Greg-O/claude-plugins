# Iteration task — speed-editdist improvement loop

You are one iteration of a long-running optimization loop. Durable state
lives in files, not in your context. Run ONE experiment end-to-end, document
it, then STOP.

## Protocol (exactly one pass)

### 1. Reconstruct state (bounded read — do not exceed this set)
- `experiment.json` — objective, primary metric, direction
- `leaderboard.json` — champion + baselines
- `LAB_NOTEBOOK.md` — INSIGHTS, GRAVEYARD, Live lineages, last ACTIVE blocks
- `tail -n 30 results.jsonl` — recent attempts including failures
- First time here? Also read `references/` if present.

### 2. Hypothesize ONE improvement
- One or two sentences; name its lineage and parent candidate.
- Island rule: if one lineage produced the last 3 promotions, advance a
  DIFFERENT or dormant lineage. Every ~5th iteration, revive a GRAVEYARD
  idea fused with a champion insight.
- Never repeat a GRAVEYARD idea without saying why it will work this time.

### 3. Implement
- Create `candidates/<id>.rs` (a fresh id; check existing files in
  `candidates/` AND recent results.jsonl). Never edit an evaluated candidate.
- Self-check it compiles first:
  `rustc --edition 2021 --target wasm32-unknown-unknown -O --crate-type=cdylib candidates/<id>.rs -o /tmp/c.wasm`

### Subagent rules (if you fan out)
Subagents are for RESEARCH, offline PROBES on tune/train data, and
DEVIL'S-ADVOCATE review of your design. Only YOU — the primary agent — may
invoke the harness eval or produce the candidate that gets evaluated.
Never delegate the official experiment or the notebook entry; you must be
able to defend every line you submit.

### 4. Test — the harness is the only referee
```
python3 loop.py eval --candidate candidates/<id>.rs \
  --meta '{"lineage":"<lineage>","parent":"<id|null>","hypothesis":"<one line>"}'
```
- NEVER edit loop.py, evaluate.py, results.jsonl, leaderboard.json, or the
  frozen evaluation data.
- NEVER self-report numbers — copy the printed HARNESS VERDICT.
- If you believe the harness/evaluator is wrong, write the suspicion under
  INSIGHTS and stop; a human adjudicates.

### 5. Document in LAB_NOTEBOOK.md, then stop
Append to **ACTIVE LOG** (newest last), exact format:

```
### [<id>] <one-line title> — <ISO date>
lineage: <lineage>   parent: <id|none>
hypothesis: <what and why>
verdict: <PASTE the harness verdict lines>
WHY: <1-2 sentences — the transferable lesson>
next: <single most promising follow-up>
```

- Ran offline probes (subagent fan-out)? Add one `probe:` line per variant
  inside the block — variant, split used (tune/train data ONLY; holdout is
  harness-only), self-measured number marked `[self-reported]`, verdict —
  plus a GRAVEYARD row per probe-killed idea. Probes are telemetry; only
  harness verdicts are evidence.
- Durable lesson learned? Add ONE bullet to INSIGHTS.
- Update "Live lineages" if a lineage was born/revived/stalled.
- If ACTIVE LOG exceeds ~15 blocks: `python3 loop.py compact`.
- End with one line:
  `ITERATION DONE: <id> <gate pass/fail> primary=<x> (champion: <id> <x>)`

## Standing objectives
1. **Minimize `fuel`** = total wasmi fuel (deterministic instruction count) to
   compute edit distance over the frozen vectors — the ONLY promotion criterion.
   The candidate MUST stay byte-exact correct (the gate): wrong on any frozen OR
   fresh random pair ⇒ rejected regardless of fuel.
2. Keep the source small and readable (it ships as the result).

### Bars to beat
- **seed (naive Wagner-Fischer two-row DP), current champion: 37,856,920 fuel.**
- Reference target: single-word **Myers/Hyyrö bit-parallel** ≈ 190,000 fuel
  (~200× less). Intermediate rungs exist and each is a real cut: unchecked
  indexing, a single rolling row, diagonal banding (distances here are small).
- Floor: O(n) work per pair vs the DP's O(n·m).

### How to write a candidate (the contract)
- ONE file `candidates/<id>.rs`, compiled to **wasm32-unknown-unknown**
  (`-O --crate-type=cdylib`) with **NO host imports** — you only compute.
- Export EXACTLY these, unchanged:
  - `input_ptr() -> u32` — pointer to a buffer the host writes `a` then `b` into.
  - `solve(a_ptr, a_len, b_ptr, b_len) -> u32` — return the edit distance.
- Both strings are ≤ **64 bytes** (one machine word) — that is why single-word
  Myers is viable. Read inputs via `core::slice::from_raw_parts`.
- **No heap**: this target has no allocator — use fixed-size stack/static arrays
  (`[u64; 256]`, `[u32; 65]`…), NOT `Vec`/`String` (they won't link). Use
  `wrapping_add`/`wrapping_shl` where overflow is intentional.
- Fuel = retired wasm instructions. Bounds checks, array re-init, and per-cell
  work all cost fuel; fewer ops ⇒ lower fuel.
- See `LAB_NOTEBOOK.md` INSIGHTS for the algorithm ladder and pitfalls.
