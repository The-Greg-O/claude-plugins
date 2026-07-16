# Greg Overton's Claude Code plugins

A personal [Claude Code](https://code.claude.com) plugin marketplace.

```
/plugin marketplace add The-Greg-O/claude-plugins
/plugin install recursive-improvement-loop@the-greg-o
```

Updates are picked up with `/plugin marketplace update`.

---

## recursive-improvement-loop

Run **recursive-improvement experiments**: long-running agentic optimization loops
where each iteration is a *fresh-context* Claude that reads a living lab notebook,
forms ONE hypothesis, implements it, gets judged by a **trusted measurement
harness**, documents the result, and exits. Hundreds of iterations, zero context
rot, unattended on your subscription — with a hard anti-cheating gate, a champion
ratchet, a statistical plateau stop, a live dashboard, and a full audit trail.

The shape is **a loop of disposable agents around durable artifacts**:

```
runner.sh                one fresh `claude -p` per iteration (the "Ralph" loop)
  └─ iteration agent     reads notebook → ONE hypothesis → implement →
       └─ loop.py        TRUSTED HARNESS: sole writer of every trusted record
            └─ evaluate.py   THE ONLY DOMAIN CODE — gate + metrics contract
```

The **harness**, not the agent, owns scoring and the stop decision. The agent never
grades itself: it submits a candidate, a gate it cannot edit checks correctness, and
a metric it cannot edit measures progress. The only domain-specific piece is a single
`evaluate.py` — and **Claude drafts it with you** (see below).
[Full plugin docs →](plugins/recursive-improvement-loop)

---

## See it work

Two real runs, each captured as a reproducible case study with a `verify.sh` that
re-derives the headline number on any machine. The charts below are the actual
champion trajectories.

### 1 · Lossless ECG compression — beats bzip2 by 12%

![compression-ecg champion trajectory](examples/compression-ecg/results/trajectory.png)

From a 3.90× seed (delta + LZMA), the loop evolved a lossless codec for **MIT-BIH
ECG** to **4.31× — +12% past bzip2-9**, the strongest general-purpose baseline. It
earned it: it discovered that bzip2 wins by dictionary-matching the repeating
heartbeat, ruled out beat-to-beat prediction (it's an *arrhythmia* database),
survived an 11-iteration plateau, and broke through with **NLMS adaptive
prediction** — documenting every dead end in its
[lab notebook](examples/compression-ecg/LAB_NOTEBOOK.md).

```bash
cd examples/compression-ecg && ./verify.sh      # → champion 4.3133× vs bzip2 3.85×
```
[Case study →](plugins/recursive-improvement-loop/skills/recursive-improvement-loop/references/examples/compression-ecg.md)

### 2 · A faster Rust algorithm — 425× fewer instructions

![speed-editdist champion trajectory](examples/speed-editdist/results/trajectory.png)

This is [AlphaDev](https://www.nature.com/articles/s41586-023-06004-9)'s setup —
minimize instruction count for a fixed kernel under a byte-exact correctness gate —
but the fitness is **deterministic Wasmtime/wasmi fuel**, so the number is identical
on every machine. From a naive Wagner-Fischer DP (37.9M fuel, off-chart at iteration
1), the loop **rediscovered Myers bit-parallelism on its own** (194× in a single
leap), then *beat the textbook*: affix-trimming, WASM-level micro-optimization, and a
closed-form popcount invariant for the score — reaching **88,976 fuel, 425× below
naive.**

```bash
cd examples/speed-editdist && ./verify.sh        # → champion 88,976 fuel vs 37.9M naive
```
[Case study →](plugins/recursive-improvement-loop/skills/recursive-improvement-loop/references/examples/speed-editdist.md)

---

## Start your own — Claude sets it up

You don't drive the harness by hand. Point Claude at a goal and it acts as the
**orchestrator** — interviewing you, writing the code, and running the loop:

```
/recursive-improvement-loop:init
```

…or just ask: *"set up a recursive-improvement experiment to optimize &lt;X&gt;."*
Claude then:

1. **Interviews you** to pin down what actually makes a good loop — the objective, the
   single promotion **metric**, the hard correctness **gate**, the frozen **data** +
   holdout, the **baselines** to beat, deployment **constraints**, and the **stop**
   policy. It pushes back on anything too vague to survive an unattended run, and
   confirms an **Experiment Charter** with you before building anything.
2. **Writes the code** — it scaffolds the experiment and drafts the one domain-specific
   piece, `evaluate.py` (your gate + metrics), then seeds a candidate and records the
   baselines.
3. **Runs it** — it launches the loop (a fresh `claude -p` per iteration) and you just
   watch `dashboard.html`; it stops itself on a statistical plateau.

Your job is to answer the interview and read the results — you generally never touch
`loop.py` or `runner.sh` yourself. A problem is a good fit when three things hold: it's
**measurable**, **gateable**, and **iterable** (one eval runs in seconds-to-minutes),
and the interview exists mostly to confirm that before burning hours on it.

<details>
<summary><b>Under the hood</b> — the commands Claude runs for you</summary>

```bash
python3 <plugin>/skills/recursive-improvement-loop/scripts/loop.py init my-experiment
cd my-experiment
# evaluate.py → final stdout line: {"gate_passed": true, "metrics": {"score_holdout": 4.01}}
python3 loop.py eval --candidate candidates/seed     # seed + record baselines
./runner.sh -n 30 -p 12                               # fresh claude -p per iteration
```

Everything but `evaluate.py` — the loop, the champion ratchet, the plateau stop, the
lab notebook, the dashboard, the audit — is generic. Deeper docs: the
[evaluator guide](plugins/recursive-improvement-loop/skills/recursive-improvement-loop/references/evaluator-guide.md)
and the
[design principles](plugins/recursive-improvement-loop/skills/recursive-improvement-loop/references/design-principles.md).
</details>

---

## The loop that improves the loop

New (experimental): an outer **meta-loop** that optimizes *how the loop iterates*,
with the same referee discipline the inner loop applies to candidates. The
iteration prompt is split into a frozen protocol (`PROMPT_CORE.md` — trust rules,
never mutated) and a mutable strategy layer (`POLICY.md` — read set, island-rule
constants, objectives). Run with `-M K` and every K iterations a bounded
meta-agent reads a harness-computed fitness readout (`loop.py meta-stats`:
promotions per eval, champion improvement **per million tokens**, lineage
entropy, hypothesis-repeat rate…) and may propose ONE small `POLICY.md` edit —
which then runs **blind** for the next K iterations and is auto-reverted via git
unless its window measurably beats the incumbent's. The evaluator and harness
stay frozen at both levels: the meta-loop tunes strategy, never the referee.
Design and phasing: [`docs/meta-loop-design.md`](plugins/recursive-improvement-loop/docs/meta-loop-design.md).

> **Safety:** `runner.sh` drives an autonomous agent with
> `--dangerously-skip-permissions`. Run it in a dedicated, version-controlled
> experiment directory. Trust boundaries are enforced in software — the agent cannot
> edit the harness or its evaluator, never scores itself, and candidate code runs
> sandboxed. The meta-agent is boundary-checked the same way: any write outside
> `POLICY.md` is detected and rolled back by the runner.

## License

[MIT](LICENSE). The ECG example uses the MIT-BIH Arrhythmia Database (ODC-By 1.0; see
[`examples/compression-ecg/NOTICE`](examples/compression-ecg/NOTICE)). The data is
fetched on demand, not redistributed here.
