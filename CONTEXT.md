# Domain glossary — recursive-improvement-loop

The shared vocabulary for this repo. Code, docs, and reviews should use
these terms exactly.

## Roles

- **Referee / trusted harness** — `loop.py` plus the experiment's
  `evaluate.py`: the only code allowed to produce a score or write a
  trusted record. Frozen from the iteration agent's perspective.
- **Iteration agent** — one fresh headless Claude process per iteration.
  Proposes exactly one candidate, submits it to the referee, documents
  the verdict, exits. Never grades itself.
- **Runner** — `runner.sh`. Spawns iteration agents, measures wall
  clock/exit codes, orchestrates the meta-pass. Measures and plumbs;
  never assembles a trusted record.
- **Meta-pass** — a bounded agent the runner spawns every K iterations
  (`-M K`) that may propose ONE small edit to the policy surface.

## Artifacts

- **Candidate** — one file in `candidates/`, immutable once evaluated.
- **Champion** — the best gate-passing candidate so far; promotion
  requires beating it by an epsilon (the **champion ratchet**).
- **Trusted records** — `results.jsonl`, `leaderboard.json`,
  `loop_audit.jsonl` (the **audit log**), `meta_state.json`. Written by
  the harness only (the **sole-writer principle**).
- **Lab notebook** — `LAB_NOTEBOOK.md`: INSIGHTS / GRAVEYARD / lineages /
  ACTIVE LOG. The loop's memory; agent-written, verdicts copied verbatim.
- **Lineage** — a structurally distinct solution family; the island rule
  forces rotation when one lineage dominates promotions.
- **Policy surface** — `POLICY.md` (read set, island-rule constants,
  standing objectives): the only file the meta-pass may mutate. The
  frozen protocol lives in `PROMPT_CORE.md`.
- **Blind window** — the K iterations a trial policy runs before the
  meta-ratchet keeps (commits) or auto-reverts it on harness-computed
  fitness it cannot see coming.

## Architectural principles

- **Sole writer** — every trusted byte on disk is written by tested
  harness python; bash passes measurements to `loop.py audit-append`.
- **Traveling harness** — `loop.py init` copies `loop.py`, `runner.sh`,
  `_stream_view.py`, and `_dashboard.py` into each experiment so the
  referee freezes with the results it produced. Never "fix" an old
  experiment's copy mid-campaign.
- **Display is not the referee** — rendering lives in `_dashboard.py`
  behind a guard that can never fail an eval.
- **Frozen referee at every level** — neither the iteration agent nor
  the meta-pass may touch `loop.py`, `evaluate.py`, frozen data, or the
  records; the meta-loop mutates strategy, never scoring
  (`plugins/recursive-improvement-loop/docs/meta-loop-design.md`).
