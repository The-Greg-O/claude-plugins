# POLICY — {{NAME}} strategy layer

This is the mutable strategy layer: the loop's policy surface. A meta-pass
(or a human) may revise this file between iterations; the runner re-reads
it fresh every iteration and stamps its hash into loop_audit.jsonl as
`policy_sha`. Iteration agents FOLLOW this file but never edit it. The
frozen protocol in PROMPT_CORE.md always takes precedence.

## Read set (protocol step 1 — do not exceed)
- `experiment.json` — objective, primary metric, direction
- `leaderboard.json` — champion + baselines
- `LAB_NOTEBOOK.md` — INSIGHTS, GRAVEYARD, Live lineages, last ACTIVE blocks
- `tail -n 30 results.jsonl` — recent attempts including failures
- First time here? Also read `references/` if present.

## Strategy heuristics (protocol step 2)
- Island rule: if one lineage produced the last 3 promotions, advance a
  DIFFERENT or dormant lineage.
- Every ~5th iteration, revive a GRAVEYARD idea fused with a champion
  insight.
- Never repeat a GRAVEYARD idea without saying why it will work this time.

## Notebook curation (protocol step 5)
- If ACTIVE LOG exceeds ~15 blocks: `python3 loop.py compact`.

## Standing objectives
1. <PRIMARY: e.g. "maximize holdout score" — the only promotion criterion>
2. <secondary objectives, in priority order>
<Concrete bars to beat: baselines, theoretical bounds, the status quo.>
