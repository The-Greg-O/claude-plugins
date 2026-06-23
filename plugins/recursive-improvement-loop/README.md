# recursive-improvement-loop

Run **recursive-improvement experiments**: long-lived agentic optimization
loops where each iteration is a fresh-context Claude that reads a living
lab notebook, forms ONE hypothesis, implements it, gets judged by a trusted
measurement harness, documents the result, and exits. Hundreds of
iterations, zero context rot, unattended on your Claude subscription, with
a hard anti-cheating gate, champion ratchet, statistical plateau stop, live
dashboard, and a complete audit trail.

Distilled from the FunSearch/AlphaEvolve literature, the open-source
evolutionary-agent ecosystem, and research on why in-session loops don't
survive long campaigns. It was first built for a real lossless-compression
optimization, and the harness is domain-agnostic.

## Install

```
/plugin marketplace add The-Greg-O/claude-plugins
/plugin install recursive-improvement-loop@the-greg-o
```

## What you get

- **Skill `recursive-improvement-loop`** — the methodology: qualify the problem
  (measurable / gateable / iterable), scaffold, write a trustworthy
  evaluator, seed + baseline, run, operate. With references:
  - `design-principles.md` — the research-backed why
  - `evaluator-guide.md` — gates, holdouts, anti-reward-hacking, timing
  - `examples/` — two worked case studies from real runs (compression-ecg,
    speed-editdist), each with a runnable top-level experiment + `verify.sh`
- **`/recursive-improvement-loop:init [dir]`** — interview-driven experiment setup
- **`/recursive-improvement-loop:status`** — one-screen health check of a running loop
- **Scripts** (copied into each experiment so they version with it):
  - `loop.py` — generic harness: eval contract, champion ratchet +
    regression guard, plateau detection, notebook compaction, dashboard
    (matplotlib optional), embed-the-answer size guard
  - `runner.sh` — fresh-`claude -p`-per-iteration driver with model/effort
    flags (incl. `-e ultracode` for multi-agent iterations), audit logging
  - `_stream_view.py` — live terminal rendering of the agent's thinking,
    tool calls, and harness verdicts

## The one domain-specific piece

`evaluate.py` — your gate + metrics, to a 10-line JSON contract — which **Claude
drafts with you** from the intake interview:

```json
{"gate_passed": true, "gate_error": "", "metrics": {"score_holdout": 4.01}}
```

Everything else — the loop, the ratchet, the notebook, the dashboard, the
audit — is generic.

## Good fits

Compression/codec design, query and pipeline performance (frozen replay
sets), detection/algorithm parameter tuning against labeled captures,
prompt/agent-tool optimization against eval suites, cost-per-result
reduction, binary-size or memory squeezing — anything with a number to
move, a correctness gate, and an eval you can run in seconds-to-minutes.

## Quick start — let Claude run it

You don't drive the harness by hand. Ask Claude to set one up:

```
/recursive-improvement-loop:init
```

It **interviews you** (objective, metric, gate, data + holdout, baselines,
constraints, stop policy), confirms an **Experiment Charter**, then scaffolds the
experiment, drafts `evaluate.py`, records seed + baselines, and launches the loop —
one fresh `claude -p` per iteration. You answer the interview and watch
`dashboard.html`; `/recursive-improvement-loop:status` is a one-screen health check.
You generally never run `loop.py` / `runner.sh` yourself.

<details>
<summary>The commands it runs under the hood</summary>

```bash
python3 <plugin>/skills/recursive-improvement-loop/scripts/loop.py init my-experiment
cd my-experiment
python3 loop.py eval --candidate candidates/seed --meta '{"lineage":"seed"}'
python3 loop.py eval --candidate <reference> --baseline
./runner.sh -n 25 -p 10        # fresh claude -p per iteration; watch dashboard.html
```
</details>

## Safety

`runner.sh` drives an autonomous, unattended agent with
`--dangerously-skip-permissions`, so it can edit files and run commands without
prompting. Run it only inside a dedicated, version-controlled experiment
directory — ideally a container or throwaway VM.

Trust boundaries are enforced in software, not by the OS: `loop.py` (the
harness) and your `evaluate.py` are the trusted referee. The iterating agent
cannot edit them, never scores itself, and every result is recorded by the
harness. Candidate code the agent writes is run sandboxed in a subprocess with
a timeout (see [`evaluator-guide.md`](skills/recursive-improvement-loop/references/evaluator-guide.md)).
