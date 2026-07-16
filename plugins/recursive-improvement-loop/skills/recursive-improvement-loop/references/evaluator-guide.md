# Writing a trustworthy evaluate.py

The evaluator is the experiment. The harness (loop.py) is generic and
trusted; the iterating agent is capable and *motivated by construction* to
find holes in your metric. Everything below is a hole someone's loop
actually found.

## The contract
`<eval_cmd> <candidate_path>` → final stdout line is one JSON object:
```json
{"gate_passed": true,
 "gate_error": "",
 "metrics": {"score_holdout": 4.01, "score_train": 4.13,
             "latency_ms": 12.3, "artifact_bytes": 4096}}
```
- `gate_passed` — hard validity/correctness, never a score.
- `metrics` must include the `primary_metric` named in experiment.json.
- Nonzero exit, timeout, or malformed JSON ⇒ recorded as a gate failure.
- Print human-readable detail BEFORE the JSON line freely (it shows in the
  terminal); only the last line must be the contract object.

## Gate design — the difference between a loop and a liability
1. **Pass/fail, binary, non-negotiable.** "Mostly correct" is FAIL.
2. **Test adversarial inputs, not typical ones.** A good compression
   gate runs all-zero, all-max, full-range jumps, single-element spikes,
   and pure random — because a widen-then-clip codec passes every smooth
   input and silently corrupts extreme values. Build the equivalent set for your
   domain (empty input, max size, unicode/null bytes, boundary values,
   malformed-but-parseable).
3. **Verify end-to-end, never trust internal consistency.** Check
   decode(encode(x))==x against the ORIGINAL x; diff output against ground
   truth; run the real test suite. A candidate that validates itself is a
   candidate that grades itself.
   For SET-VALUED outputs (SQL result sets, JSON arrays without ordering
   semantics), canonicalize before comparing — sort rows / normalize types
   unless an explicit ORDER BY makes order part of correctness. Otherwise
   the gate false-fails exactly the plan/index changes a query-perf loop
   exists to find.
4. **Plant a bug to test the gate** before looping: a deliberately invalid
   candidate MUST fail. If it passes, the loop will optimize into the hole.

## Frozen data + holdout discipline
- Snapshot evaluation data once; record sha256 in a manifest; FAIL LOUDLY
  if it changes mid-campaign (otherwise the trend line you're watching is
  comparing incomparable numbers).
- Split: tune/train data the agent may study vs **holdout** that decides
  promotion. Different files/seeds/scenarios — not a random shuffle of the
  same pool, or memorization transfers.
- Report both (`score_train`, `score_holdout`); set primary_metric to the
  HOLDOUT one. A growing train↔holdout gap = overfitting, visible on the
  dashboard. Keep the `<stem>_train` / `<stem>_holdout` naming convention —
  `loop.py meta-stats` pairs metrics by those suffixes to compute the
  overfit-gap readout.

## Anti-reward-hacking checklist
- **Sandbox the candidate**: run it in a subprocess with a wall-clock
  timeout and a neutral CWD; pass data in-memory or via temp paths, never
  the canonical data location.
- **No peeking**: static-check candidate source — ban file/network/process
  access and the names/paths of your data and manifest. (AST import
  whitelist + regex ban-list.)
- **No embedding the answer**: cap candidate size (`max_candidate_bytes`,
  enforced by loop.py) so a candidate can't smuggle the holdout data in as
  a literal. If the agent can read the holdout files at iteration time,
  also grep candidates for long literals, or generate holdout fresh from a
  seed the agent never sees.
- **Determinism**: same candidate + same data ⇒ same score (seed RNGs).
  loop.py's champion re-verify will abort the loop on drift — that's a
  feature; fix the nondeterminism rather than disabling the guard.
- **Resource caps**: per-stage timeouts so an O(n³) candidate can't hang
  the loop; record TIMEOUT as a gate failure with the stage named.

## Timing & noisy metrics
- Measure best-of-K (K≥5) after a warmup rep; report the MIN (least
  contaminated estimate of true cost).
- Measure your machine's noise floor once (re-time the same candidate ~10
  times); treat differences inside ~2× that band as TIES. Prefer a
  deterministic promotion metric when one exists, with noisy metrics as
  banded tie-breakers.
- Never define plateau on raw noise (the loop will chase ghosts forever).

## When the primary metric IS noisy or LLM-judged
Sometimes the thing you're optimizing is inherently noisy (p95 latency) or
judge-scored (prompt quality). The loop still works — configure it honestly:
1. **Aggregate inside the evaluator** until the number is stable enough to
   rank on: median/min of K≥5 timed replays for latency; mean of M judge
   replicates with a PINNED judge model, temperature 0, and responses
   cached by (candidate hash, case id) for pass-rate metrics.
2. **Size the thresholds to the noise**: set `promote_eps` /
   `promote_eps_rel` ≥ 2× your measured noise band so promotions are real;
   for pass rates, `promote_eps` ≥ 1/N_cases (one case).
3. **Relax the drift guard to match**: `reverify_rel_tol` ≥ 2× the noise
   band, or `reverify_champion: "gate-only"` (gate still checked; the
   primary-drift comparison skipped). Default 0.1% tolerance assumes a
   deterministic evaluator and will false-FATAL on noisy ones.
4. **Mind eval cost**: every candidate eval also re-runs the champion. For
   minutes-long LLM-judge suites set `reverify_every: 5` (or more) and/or
   "gate-only", and consider a cheap deterministic proxy metric (e.g.
   EXPLAIN cost, token count) as a cascade stage before the expensive one.

## Cascade expensive evaluations
Order stages cheap→dear, fail fast:
1. Static checks + adversarial gate on a tiny input (ms)
2. Sanity score on a small slice — reject degenerate candidates (s)
3. Full train+holdout scoring + timing (the expensive part)
A bad candidate should cost milliseconds, not minutes.

## Metric design
- Primary metric: ONE number, deterministic, computed on holdout. Resist
  composite weighted scores at first — optimize the thing you actually
  want, report the rest as secondary metrics, and impose other dimensions
  as gate constraints (budgets) instead.
- Include cost/footprint proxies relevant to deployment as secondary
  metrics (binary size, memory peak, dependency tier) so the dashboard
  shows the whole trade-space even though promotion is single-objective.
- If you genuinely need multi-objective: keep primary for promotion,
  and record dominance ("pareto_front": true) as a metric for analysis.
