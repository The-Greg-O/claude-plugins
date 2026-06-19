---
description: "Scaffold a new recursive-improvement experiment (interview → charter → scaffold → evaluator → seed → baselines)"
argument-hint: "[experiment-directory]"
---

Set up a new recursive-improvement-loop experiment. Follow the
`recursive-improvement-loop` skill's setup procedure (invoke it now if not
already loaded), and run the intake interview from
`references/intake-interview.md` — do not skip to scaffolding.

1. **Interview me** using the question bank in
   `${CLAUDE_PLUGIN_ROOT}/skills/recursive-improvement-loop/references/intake-interview.md`:
   sharpen the objective until it is measurable / gateable / iterable,
   then cover metric, gate + adversarial cases, frozen data + holdout,
   baselines, noise + eval cost, constraints, lineages, and stop policy.
   Use AskUserQuestion for the closed forks (direction, noise class, gate
   type, stop policy); push back on every red-flag answer the reference
   lists. If the objective fails measurable/gateable/iterable after two
   rounds of sharpening, stop and tell me a loop is the wrong tool.
2. **Write the Experiment Charter** (template at the end of
   intake-interview.md), show it to me, and get my explicit confirmation
   before touching the filesystem.
3. **Scaffold**: run
   `python3 "${CLAUDE_PLUGIN_ROOT}/skills/recursive-improvement-loop/scripts/loop.py" init $1`
   (default to a sensible directory name from the charter if I didn't
   give one), then fill every artifact FROM the charter: experiment.json
   (metric/direction/noise config), evaluate.py (per
   references/evaluator-guide.md), PROMPT.md standing objectives + bars
   to beat, LAB_NOTEBOOK.md lineages + seed INSIGHTS.
4. **Verify the rails**: a seed candidate must pass and become champion; a
   deliberately-broken candidate MUST be rejected by the gate; record
   baselines with `--baseline`.
5. Show me the dashboard and the exact `./runner.sh` command to start.
