# Iteration task — compression-ecg improvement loop

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
- Create `candidates/<id>.py` (a fresh id; check existing files in
  `candidates/` AND recent results.jsonl). Never edit an evaluated candidate.
- Self-check it imports and round-trips on a tiny array before the harness.

### Subagent rules (if you fan out)
Subagents are for RESEARCH, offline PROBES on tune/train data, and
DEVIL'S-ADVOCATE review of your design. Only YOU — the primary agent — may
invoke the harness eval or produce the candidate that gets evaluated.
Never delegate the official experiment or the notebook entry; you must be
able to defend every line you submit.

### 4. Test — the harness is the only referee
```
python3 loop.py eval --candidate candidates/<id>.py \
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
1. **Maximize `ratio_holdout`** = raw_bytes / compressed_bytes (lossless,
   byte-exact) on the subject-disjoint HOLDOUT records — the ONLY promotion
   criterion. Watch `ratio_train` vs `ratio_holdout` for overfitting.
2. Keep the codec source small and readable (it ships as the result).

### Bars to beat (holdout)
- gzip-9: 2.26×  ·  xz-9e: 3.23×  ·  **bzip2-9: 3.85× ← the bar to beat**
- seed (per-channel delta + LZMA), current champion: **3.90×** (only ties bzip2)
- A *convincing* win clearly beats bzip2-9 — target **4.3×+** on holdout, toward
  the literature ceiling (~4.5–5× via prediction + Rice/arithmetic residual
  coding). Naive delta only ties bzip2; winning needs real signal modeling.

### How to write a candidate (the contract)
- A candidate is ONE file `candidates/<id>.py` exposing
  `encode(blob: bytes) -> bytes` and `decode(blob: bytes) -> bytes`, lossless
  (`decode(encode(x)) == x`), operating on a canonical `.ecg` blob.
- Canonical blob = 10-byte header (`b'ECG1'` + uint16 n_chan + uint32 n_samp)
  then channel-major int16 little-endian samples. Parse it; model each channel.
- Allowed imports ONLY: numpy, struct, zlib, bz2, lzma, math, array,
  collections, itertools, functools, typing, heapq. No file/network/os access
  (statically enforced — you receive bytes, never paths).
- See `LAB_NOTEBOOK.md` INSIGHTS for the data profile and starting lineages.
