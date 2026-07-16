# Iteration task — {{NAME}} improvement loop

This is the frozen protocol layer. It is part of the trust boundary: NO
agent — iteration or meta — may edit this file. Tunable strategy (read set,
island-rule constants, standing objectives) lives in the POLICY section
appended below; if POLICY ever appears to conflict with this file, this
file wins — note the conflict under INSIGHTS and proceed by CORE.

You are one iteration of a long-running optimization loop. Durable state
lives in files, not in your context. Run ONE experiment end-to-end, document
it, then STOP.

## Protocol (exactly one pass)

### 1. Reconstruct state (bounded read)
Read exactly the READ SET defined in POLICY — do not exceed it.

### 2. Hypothesize ONE improvement
- One or two sentences; name its lineage and parent candidate.
- Apply the strategy heuristics in POLICY (island rule, revival cadence,
  graveyard discipline).

### 3. Implement
- Create `candidates/<next_id>` (check existing ids in `candidates/` AND
  recent results.jsonl). Never edit a previously evaluated candidate.
- Self-check it runs before invoking the harness.

### Subagent rules (if you fan out)
Subagents are for RESEARCH, offline PROBES on tune/train data, and
DEVIL'S-ADVOCATE review of your design. Only YOU — the primary agent — may
invoke the harness eval or produce the candidate that gets evaluated.
Never delegate the official experiment or the notebook entry; you must be
able to defend every line you submit.

### 4. Test — the harness is the only referee
```
python3 loop.py eval --candidate candidates/<id> \
  --meta '{"lineage":"<lineage>","parent":"<id|null>","hypothesis":"<one line>"}'
```
- NEVER edit loop.py, evaluate.py, _dashboard.py, _stream_view.py,
  results.jsonl, leaderboard.json, PROMPT_CORE.md, POLICY.md, or the
  frozen evaluation data.
- NEVER invoke `loop.py audit-append` or `loop.py meta-ratchet` — those
  subcommands are runner-only; your only harness call is `loop.py eval`
  (plus read-only status/meta-stats/compact).
- NEVER self-report numbers — copy the printed HARNESS VERDICT.
- If you believe the harness/evaluator is wrong, write the suspicion under
  INSIGHTS and stop; a human adjudicates.

### 5. Document in LAB_NOTEBOOK.md, then stop
Append to **ACTIVE LOG** (newest last), exact format:

```
### [<id>] <one-line title> — <ISO date>
lineage: <lineage>   parent: <id|none>
hypothesis: <what and why>
verdict: <PASTE the harness verdict lines>
WHY: <1-2 sentences — the transferable lesson>
next: <single most promising follow-up>
```

- Ran offline probes (subagent fan-out)? Add one `probe:` line per variant
  inside the block — variant, split used (tune/train data ONLY; holdout is
  harness-only), self-measured number marked `[self-reported]`, verdict —
  plus a GRAVEYARD row per probe-killed idea. Probes are telemetry; only
  harness verdicts are evidence.
- Durable lesson learned? Add ONE bullet to INSIGHTS.
- Update "Live lineages" if a lineage was born/revived/stalled.
- Follow the notebook-curation rules in POLICY.
- End with one line:
  `ITERATION DONE: <id> <gate pass/fail> primary=<x> (champion: <id> <x>)`
