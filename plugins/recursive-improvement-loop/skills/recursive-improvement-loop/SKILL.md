---
name: recursive-improvement-loop
description: Set up and run recursive-improvement experiments — long-running agentic optimization loops where Claude iteratively hypothesizes, implements, tests against a trusted measurement harness, and documents results in a living lab notebook across hundreds of fresh-context iterations. Use whenever someone wants to optimize anything measurable through iterative experimentation - compression ratios, query/pipeline performance, prompt quality, model parameters, algorithm tuning, cost reduction - or mentions "improvement loop", "optimization loop", "recursive improvement", "experiment loop", "let the agent iterate on this overnight", or wants Claude to keep improving something against a metric.
---

# Improvement Loop

Run optimization as a **loop of disposable agents around durable artifacts**:
each iteration is a fresh `claude -p` process that reconstructs state from a
lab notebook, runs ONE experiment through a trusted harness, documents the
result, and exits. The loop survives hundreds of iterations with zero
context rot, can run unattended on a subscription, stops itself on
statistical plateau, and leaves a complete audit trail + live dashboard.

It was first built for a real lossless-compression optimization, where a hard
losslessness gate made cheating structurally impossible while the loop invented
and discarded codec designs unattended. The harness itself is domain-agnostic —
see `references/examples/` for two worked case studies (lossless ECG compression
beating bzip2, and a Rust hot path sped up ~425×).

## The architecture (memorize this shape)

```
runner.sh                one fresh `claude -p` per iteration (the "Ralph" loop)
  └─ iteration agent     reads notebook → ONE hypothesis → implement →
       └─ loop.py        TRUSTED HARNESS: sole writer of every trusted record
            └─ evaluate.py   THE ONLY DOMAIN CODE — gate + metrics contract
```

Five load-bearing principles (full reasoning in
`references/design-principles.md`):
1. **Verifiable reward** — a hard pass/fail gate + numeric metrics measured
   by code the agent cannot edit. The agent never grades itself.
2. **Fresh context, durable artifacts** — state lives in LAB_NOTEBOOK.md
   (three-tier: INSIGHTS / GRAVEYARD / ACTIVE LOG) + results.jsonl, never in
   the context window. In-session loops (/loop, Stop-hook loops) accumulate
   context and cannot run hundreds of iterations.
3. **Champion ratchet** — best-so-far is immutable, regression-guarded, and
   promotion requires strictly beating it on the primary metric (holdout
   where applicable).
4. **Statistical plateau stop** — stop after N evals without a promotion,
   never on a single miss (flat iterations are expected noise) and never on
   a model-uttered "done".
5. **Anti-reward-hacking by construction** — sandboxed evaluation, frozen +
   hashed data, no-peeking checks, candidate size caps, holdout-only
   promotion. Assume the loop WILL find any hole in the metric.

## Setting up a new experiment

### Step 1 — Qualify the problem (do not skip)
An improvement loop only works when ALL THREE hold:
- **Measurable**: a numeric primary metric a script can compute. ("Two
  domain experts would independently get the same number.")
- **Gateable**: a hard correctness check separating valid from invalid
  candidates (tests pass, output byte-exact, schema valid, build succeeds).
- **Iterable**: one eval is fast enough to run dozens-to-hundreds of times.

If the user's goal fails any of these, STOP and redesign the objective with
them first (e.g. "make it better" → "reduce p95 latency on the frozen
replay set, gate = identical responses"). This conversation is the highest
-leverage part of the whole setup.

### Step 2 — Interview, then fill the scaffold
Run the structured intake interview in `references/intake-interview.md`:
a phase-by-phase question bank (objective → metric → gate → data →
baselines → noise/cost → constraints → lineages → stop policy), each
phase with the red-flag answers to push back on and the config it sets.
It ends with an **Experiment Charter** the user explicitly confirms
BEFORE anything is scaffolded — every charter field maps 1:1 onto the
artifacts below.

Scaffold the experiment directory:
```bash
python3 "$(dirname <this-skill>)/scripts/loop.py" init <experiment-dir>
```
(or copy `scripts/loop.py` + `scripts/templates/` manually). Then, with the
user, fill in:
- `experiment.json` — name, eval_cmd, **primary_metric**, direction
  (maximize/minimize), secondary_metrics, eval timeout.
- `evaluate.py` — THE critical artifact; follow
  `references/evaluator-guide.md` (gate on adversarial inputs, frozen+hashed
  data, holdout split, subprocess+timeout isolation, noise-banded timing).
- `POLICY.md` — the mutable strategy layer (read set, island-rule
  constants, Standing Objectives): fill the objectives with concrete bars
  to beat (baselines, status quo, theoretical bound if computable). The
  frozen protocol + trust rules live in `PROMPT_CORE.md` — scaffolded,
  never edited; the runner concatenates the two into each iteration's
  prompt, so POLICY.md is hot-swappable mid-campaign.
- `LAB_NOTEBOOK.md` — seed INSIGHTS with everything already known (data
  profile, prior art numbers, constraints) and 3-5 starting **lineages**
  (structurally different solution families, for island-style diversity).
- `CLAUDE.md` — generated; review the trust boundaries.

### Step 3 — Seed and baseline
1. Write a trivial-but-valid seed candidate; `python3 loop.py eval
   --candidate candidates/<seed>` — it should pass the gate and become the
   first champion.
2. Record reference points: `python3 loop.py eval --candidate <ref>
   --baseline` for each obvious alternative (the status quo, the standard
   library approach, the current production system). Baselines draw as
   dashed lines on the dashboard and define "worth doing".
3. Sanity-test the gate with a deliberately INVALID candidate — it must be
   rejected. If the gate can't catch a planted bug, fix the gate before
   looping (a loop with a weak gate optimizes toward the hole).

### Step 4 — Run
```bash
./runner.sh -n 25 -p 10                  # 25 iters max, stop at 10 w/o promotion
./runner.sh -m opus -e ultracode -n 30   # strongest: opus + multi-agent fan-out
./runner.sh -n 100 -p 15 -M 10           # + meta-pass every 10 iters: a bounded
                                         #   agent may propose ONE POLICY.md edit,
                                         #   trialled blind and auto-reverted
                                         #   unless it beats the incumbent window
                                         #   (worthwhile on long campaigns)
./runner.sh --help                       # model/effort/turns/sleep/meta flags
```
- Terminal streams the agent's live thinking/tool-calls/verdicts
  (`_stream_view.py`); `dashboard.html` auto-refreshes with metric trends,
  baselines, audit; `checkpoints/cp_*.png` is the time-lapse.
- Runs on the operator's logged-in Claude subscription (headless `claude
  -p`), `--dangerously-skip-permissions` inside the experiment dir — keep
  the experiment in its own directory, with version control.
- Audit: `loop_audit.jsonl` records wall time / turns / tokens / active
  `policy_sha` per iteration; `results.jsonl` records every attempt +
  per-metric deltas.

### Step 5 — Operate
- Watch the train/holdout gap (overfitting) and gate-failure rate (prompt
  or evaluator clarity problem) on the dashboard.
- The loop stops itself on plateau; restart anytime — state is on disk.
- Between campaigns, curate the notebook: promote ACTIVE-LOG lessons into
  INSIGHTS, run `loop.py compact`, prune stale lineages.
- `python3 loop.py status` for a one-screen summary;
  `python3 loop.py meta-stats --window 15` for the meta-fitness readout
  (promotions/eval, champion delta per Mtoken, gate-fail rate, lineage
  entropy, hypothesis-repeat rate, train↔holdout gap, per-policy
  attribution) — the raw material for tuning POLICY.md between windows;
  interpret it with the failure-mode table below. The meta-loop design
  lives in the plugin's `docs/meta-loop-design.md`.

## Failure modes to check when a loop misbehaves
| Symptom | Likely cause | Fix |
|---|---|---|
| Absurd metric jump | reward hacking / gate hole | inspect champion; tighten gate + bans |
| Train≫holdout | overfitting to eval data | promote on holdout only; refresh holdout |
| Endless tiny "wins" | measuring noise | promote_eps up; noise-band timings |
| Same idea repeats | notebook not read / GRAVEYARD thin | enforce bounded-read protocol; richer WHY lines |
| One lineage dominates then stalls | premature convergence | island rule; forced dormant-lineage revival |
| Iterations slow/balloon | context bloat in-session | verify fresh-process driver; compact notebook |
| Champion suddenly worse | data/env drift | reverify_champion=true (regression guard aborts) |

## References
- `references/intake-interview.md` — the structured user interview that
  defines the experiment: question bank with red-flag answers, worked
  vague-to-sharp objective transformations, and the Experiment Charter
  template confirmed before scaffolding
- `references/design-principles.md` — why each piece exists (FunSearch/
  AlphaEvolve lineage, verified failure modes, the research behind it)
- `references/evaluator-guide.md` — writing a trustworthy evaluate.py:
  gates, frozen data, holdouts, anti-hacking, timing methodology
- `references/examples/` — two real worked case studies: compression-ecg
  (3.90×→4.31× lossless ECG, +12% past bzip2) and speed-editdist (naive→~425×
  fewer instructions, rediscovers Myers), each backed by a runnable experiment
  under the repo's top-level `examples/` with a `verify.sh`
