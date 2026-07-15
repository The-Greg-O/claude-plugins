# Intake interview — defining the experiment with the user

The interview is the highest-leverage step of the whole setup: a loop
pointed at a badly-defined objective optimizes the wrong thing for hours,
unattended. Run this BEFORE scaffolding anything. Your job is not to
transcribe the user's answers — it is to push back until every answer
would survive the loop.

How to run it:
- Use the structured-question UI (AskUserQuestion) for the closed forks
  (direction, noise class, gate type, stop policy); free-form discussion
  for domain specifics.
- For every red-flag answer below, probe — don't proceed on a vague
  answer, and say plainly why it won't survive the loop.
- End by writing the **Experiment Charter** (template at the bottom) and
  getting an explicit "yes, that's the experiment" BEFORE running
  `loop.py init`. The charter then seeds experiment.json, evaluate.py,
  POLICY.md objectives, and the notebook header.

## Phase 0 — Sharpen the objective

**Q: "What exactly should get better, and how would a script know?"**
- Why: "two domain experts would independently get the same number" is
  the bar. Everything else builds on this.
- Red flags: "make it better/faster/smarter", "improve quality",
  "accuracy" with no measurement procedure, any adjective without a unit.
- Probe: *better by what number, computed by what command, on what data?*
  Worked transformations:
  - "make queries faster" → "minimize p95 latency in ms over the frozen
    50-query replay set, gate = canonicalized result rows identical"
  - "better compression" → "maximize lossless ratio (raw/compressed
    bytes) on the held-out split, gate = byte-exact roundtrip"
  - "better prompts" → "maximize pass rate over the 40-case eval suite,
    judge model pinned at temp 0, gate = output schema validates"
- If after two rounds there is still no computable number, STOP — this is
  not (yet) a loop problem; offer normal iterative development instead.

## Phase 1 — Primary metric

**Q: "Of everything measurable, which ONE number decides promotion?"**
- Why: the ratchet promotes on exactly one metric; everything else is
  reporting. Composite weighted scores hide regressions and invite
  gaming — push the user toward one honest primary plus gate constraints.
- Red flags: "ratio AND speed AND size" (which wins a conflict?); a
  metric the user can't compute today; percent-improvement targets with
  no baseline measured yet.
- Also fix: direction (maximize/minimize), units, secondary metrics worth
  plotting, and any meaningful axis anchors for the dashboard
  (`primary_axis_range`, e.g. [0, 1024] bytes = perfect→none).
- Sets: `primary_metric`, `direction`, `secondary_metrics`,
  `primary_axis_range`.

## Phase 2 — The hard gate

**Q: "What makes a candidate WRONG — not worse, wrong?"**
- Why: the gate is what makes unattended optimization safe. Score 0 vs
  FAIL is the difference between "bad attempt" and "would have corrupted
  production data".
- Red flags: "it should mostly match", "we'd eyeball it", a gate that is
  itself a score threshold, trusting the candidate's own self-check.
- Probe: *what are the nastiest inputs?* Build the adversarial list with
  the user: empty/zero, max-range, boundary values, malformed-but-
  parseable, the case that broke production last time. A compression
  gate catches a silently-lossy codec only when full-range jumps are
  in the adversarial set.
- Set-valued outputs (SQL rows, unordered JSON): agree on
  canonicalization now, or the gate will false-fail legitimate wins.
- Commit the user to the rail test: a deliberately-broken candidate must
  be REJECTED before the loop starts.

## Phase 3 — Data: frozen + holdout

**Q: "What data does the evaluator run against, and can we freeze it?"**
- Why: an unfrozen corpus makes every number incomparable; without a
  holdout, the loop optimizes memorization.
- Red flags: "we'll use live data" (it drifts — snapshot it), "just split
  randomly" (random splits of correlated data leak — split by unit:
  sensor, day, customer, query family), "we don't have data yet" (then
  the FIRST task is capturing a gold set, not looping).
- Probe: *is the holdout genuinely out-of-sample?* Different seed,
  different physical unit, different time window — not a shuffle.
- Sets: the data-capture step, the split rule, holdout-only promotion.

## Phase 4 — Baselines and bounds

**Q: "What's the status quo, and what's the dumbest thing that works?"**
- Why: baselines turn the dashboard from numbers into race positions;
  without them nobody can say whether 3.3× is good. A theoretical bound
  (entropy, speed-of-light latency, 100% pass rate) gives the asymptote.
- Red flags: no current system to compare against ("what would you ship
  if this loop never ran?"); refusing the trivial baseline (it often
  embarrasses sophisticated approaches and that's the point).
- Probe: *what's the bar a candidate must beat to be worth deploying?*
  This becomes the POLICY.md "bars to beat" line.

## Phase 5 — Noise and eval cost (the config trap)

**Q: "If you ran the evaluator twice on the same candidate, how different
would the numbers be? And what does one eval cost?"**
- Why: the two most common silent failures are promoting on noise and
  burning the budget re-verifying champions. The defaults assume a fast,
  deterministic evaluator — anything else needs config.
- Red flags: "it's timing-based but probably fine" (measure the noise
  floor FIRST: re-time one candidate ~10×); "the judge is an LLM"
  (pin model + temp 0 + cache, aggregate M replicates); "one eval takes
  20 minutes" (design the cascade now: cheap gate → sanity slice → full).
- Sets: `promote_eps`/`promote_eps_rel` (≥2× noise band; pass rates:
  ≥1/N_cases), `reverify_rel_tol` or `reverify_champion: "gate-only"`,
  `reverify_every`, `eval_timeout_s`.

## Phase 6 — Constraints and deployability

**Q: "What must a winning candidate respect to actually ship?"**
- Why: an unconstrained loop happily wins with a 700 MB-RAM, GPL-encum-
  bered, network-calling candidate. Constraints are gate rules or
  tracked tiers, decided now.
- Probe: dependency whitelist? memory/binary ceiling? runtime target
  (MCU? ARM? lambda)? latency budget? license? Tracking an A/B
  portability tier, say, is cheap and can shape the endgame.
- Sets: evaluator ban-lists, secondary footprint metrics, POLICY.md
  constraints section, `max_candidate_bytes`.

## Phase 7 — Lineages (the hypothesis space)

**Q: "Name 3-5 structurally DIFFERENT ways this could be attacked."**
- Why: the island rule needs seeded diversity, or the loop converges on
  the first idea-family and stalls. The user's domain knowledge is the
  cheapest source of diversity — harvest it now into the notebook.
- Probe: *what's the obvious approach? what would a contrarian try? what
  does the literature/production do today? what's the long shot?* One
  line each in the notebook's lineage table, plus every known fact
  (data quirks, prior-art numbers, past failures) as seed INSIGHTS —
  facts the loop shouldn't have to rediscover at $/iteration.

## Phase 8 — Stopping, budgets, oversight

**Q: "When is this DONE enough, and how unattended is it allowed to be?"**
- Why: maximize-forever needs an explicit stop policy and a money/time
  ceiling, or it runs until someone notices.
- Fix: plateau N (default 10-15 evals without promotion), hard iteration
  cap, target-good-enough (optional: stop early if X reached), model/
  effort mix (cheap routine + strong deep pass), and how often a human
  reviews the notebook (recommend: after the first 3 iterations, then
  every ~10).

## The Experiment Charter

Write this, show it, get explicit confirmation, THEN scaffold:

```markdown
# Experiment charter — <name>
**Objective**: <one sentence, plain language>
**Primary metric**: <name> (<maximize|minimize>, unit) — promotion on
  <holdout description> only
**Gate**: <what makes a candidate invalid> — adversarial cases: <list>
**Data**: <source>, frozen <how>, split <rule>, holdout <what makes it
  out-of-sample>
**Baselines**: <status quo + trivial + any bound> — bar to beat: <X>
**Noise/cost**: evaluator is <deterministic|noisy ±X%>, one eval ≈ <t>;
  config: promote_eps=<>, reverify=<mode/every>
**Constraints**: <deps, memory, runtime, license, latency>
**Lineages**: <3-5 one-liners>
**Stop**: plateau <N>, hard cap <M> iterations, target <optional>;
  human review cadence: <when>
```

Charter → artifacts: every field maps directly — metric/direction/noise
→ experiment.json; gate/data/constraints → evaluate.py; baselines/bars →
POLICY.md objectives + `--baseline` runs; lineages/known facts →
LAB_NOTEBOOK.md tables and INSIGHTS. If a charter field is still vague,
the interview isn't done.
