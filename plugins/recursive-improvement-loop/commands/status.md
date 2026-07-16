---
description: "Show the current recursive-improvement-loop experiment's state: champion, trend, plateau, audit"
---

Report the state of the recursive-improvement-loop experiment in the current
directory (or ask which directory if there's no experiment.json here):

1. Run `python3 loop.py status`, `python3 loop.py plateau 10`, and
   `python3 loop.py meta-stats --window 15` (older experiment copies of
   loop.py may lack meta-stats — skip it if so).
2. Read `leaderboard.json` and the last ~10 lines of `results.jsonl` and
   `loop_audit.jsonl` (if present).
3. Summarize: champion + primary metric vs baselines, promotions timeline,
   gate-failure rate, train/holdout gap, lineage entropy + hypothesis-repeat
   rate (diversity/novelty), champion improvement per Mtoken, evals since
   last promotion, total/average iteration wall time, and whether the loop
   looks healthy against the failure-mode table in the recursive-improvement-loop
   skill. If a meta-pass is active (`meta_state.json` exists), also report
   the trial policy state and last keep/revert verdicts. Recommend the next
   action (keep running / deepen effort / curate notebook / adjust POLICY.md
   / stop on plateau).
