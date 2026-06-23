# speed-editdist — standing rules for every Claude session in this directory

This is a long-running recursive-improvement-loop experiment (recursive-improvement-loop
plugin). The iteration protocol lives in PROMPT.md. These rules apply to
EVERY session here, interactive or headless.

## Trust boundaries (non-negotiable)
- `loop.py` (harness) and `evaluate.py` (evaluator) are the trusted
  referees. Never modify either.
- `results.jsonl`, `leaderboard.json`, `loop_audit.jsonl`, `checkpoints/`,
  and the frozen evaluation data are harness/runner-owned. Never write.
- All metrics come from the printed HARNESS VERDICT. Never estimate or
  self-report numbers. If notebook and results.jsonl disagree,
  results.jsonl wins.
- Candidates are immutable once evaluated — new ids only.

## Memory model
Context dies with the session; these files are the memory:
- `LAB_NOTEBOOK.md` — INSIGHTS / GRAVEYARD / Live lineages / ACTIVE LOG
- `leaderboard.json` + `results.jsonl` — machine truth
Read those first; keep reads bounded (PROMPT.md §1).
