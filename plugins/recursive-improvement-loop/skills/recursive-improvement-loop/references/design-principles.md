# Design principles — why the loop is built this way

Distilled from DeepMind's FunSearch/AlphaEvolve, Anthropic's agent
engineering guidance, the open-source evolutionary-agent ecosystem
(OpenEvolve, ShinkaEvolve, AIDE), and a validated real-world compression
loop. Every rule below exists because its absence is a documented failure
mode.

## 1. The loop is generate → evaluate → select → feed back
(FunSearch, Nature 2024; AlphaEvolve 2025.) An LLM proposes a mutation, an
automated evaluator scores it on a VERIFIABLE metric, survivors enter a
durable database, and the best exemplars are fed back into the next
proposal prompt. The model is the mutation operator; the evaluator is the
selection pressure; the database is the memory. Our LAB_NOTEBOOK.md +
leaderboard.json + results.jsonl ARE that database, in human-readable form.

Key transfers:
- **Evolve inside a fixed skeleton.** FunSearch mutates one function, not
  the harness; AlphaEvolve uses EVOLVE-BLOCK markers. Constrain what the
  agent may change; keep the evaluator out of reach.
- **Best-shot prompting.** Show the current champion (and 1-2 diverse
  runners-up) WITH their scores in the iteration context — the notebook's
  leaderboard snapshot does this.
- **Parsimony pressure.** Prefer the simpler of two equal-scoring
  candidates (FunSearch biases toward shorter programs).

## 2. Fresh context per iteration; durable artifacts as memory
The "Ralph pattern": `while :; do cat PROMPT.md | claude -p ...; done` —
one fresh process per iteration, state reconstructed from disk each time.
Confirmed in practice, and consistent with published agent research:
- In-session loops (Claude Code `/loop`, Stop-hook loops like the official
  ralph-wiggum plugin) run iterations in ONE conversation: context
  accumulates, auto-compaction lossily summarizes, and Stop-hook loops
  force-stop after 8 consecutive blocks. Fine for short bursts; wrong for
  hundreds of iterations.
- A fresh process reads a BOUNDED set (notebook tiers + leaderboard + log
  tail) so iteration 300 starts exactly as sharp as iteration 1.

**Three-tier notebook** keeps the read bounded forever: INSIGHTS (curated
durable lessons — the reverse-engineered WHY, which FunSearch showed is the
real product), GRAVEYARD (one line per dead idea so failures still inform),
ACTIVE LOG (last ~15 verbose blocks; `loop.py compact` rolls aged ones into
the graveyard).

## 3. The agent never grades itself
Numbers exist only if the harness printed them. The iterating agent is
*statistically expected* to be optimistic about its own work — separating
the worker from the referee is what makes unattended runs trustworthy.
Corollaries:
- results.jsonl / leaderboard.json / loop_audit.jsonl are harness-written
  only (the runner passes measurements to `loop.py audit-append`; it never
  assembles a record itself); the notebook copies the verdict verbatim;
  on disagreement the JSONL wins.
- An adversarial "second opinion" (review subagent, or ultracode fan-out
  probing variants before the official eval) raises per-iteration quality
  but never replaces the harness verdict.

## 4. Champion ratchet + regression guard
Evaluated candidates are immutable (new ids only). Promotion requires
strictly beating the champion's primary by more than epsilon. Every run
re-verifies the stored champion and ABORTS on drift — this catches silent
corruption of evaluation data, environment changes, and nondeterministic
evaluators before they poison the trend line you're watching.

## 5. Statistical plateau, never "done"
Optimization has no DONE state — a completion-string stop (Ralph-plugin
style) either never fires or fires falsely. Per-attempt success is well
below 100% in working FunSearch-class loops, so a single flat iteration is
noise. Stop = N consecutive evals without a promotion (default 10-15),
with a hard iteration cap as runaway insurance. Before accepting a final
plateau, force one dormant-lineage revival; only if THAT fails is the
plateau real.

## 6. Island lineages against premature convergence
If every iteration builds on the single global best, the loop collapses
into one idea-family and stalls (the failure the FunSearch island model
prevents). Maintain 3-5 structurally different lineages; if one lineage
wins 3 in a row, the next iteration must advance another; revive dead
lineages periodically, fused with champion insights.

## 7. Assume the loop attacks the metric
Any exploitable gap between "scores well" and "is actually good" WILL be
found eventually — not from malice, from optimization pressure. Defenses
(detail in evaluator-guide.md): hard adversarial gate, frozen+hashed data,
holdout-only promotion, sandboxed eval with timeouts, no-peeking bans,
candidate size caps, noise-banded timing. Reward hacking observed in the
wild includes: special-casing test inputs, exploiting harness bugs,
silently-lossy "wins", timing noise harvesting, and embedding evaluation
answers in the candidate.

## 8. The meta-loop: improve the process, never the referee
(Weco AIDE², 2026 — first credible evidence that an outer loop rewriting the
inner agent's iteration machinery compounds.) Our inner agent's "machinery"
is its policy surface: the mutable half of the prompt (POLICY.md — read
set, island-rule constants, standing objectives), cleanly split from the
frozen protocol (PROMPT_CORE.md) so no optimization pressure can loosen
trust language. A meta-pass may propose ONE small policy edit per window,
judged blind on the NEXT window's harness-computed, cost-normalized
fitness (champion delta per Mtoken), and auto-reverted unless it wins by
an epsilon sized to window noise. Sobering priors transfer: ~9/10 outer
proposals fail; winners are simple compositions; nested loops compound
noise. The referee — loop.py, evaluate.py, frozen data, and the meta-stats
computation itself — is never mutable at any level, for the same reason
the inner agent never grades itself. Full design: the plugin's
`docs/meta-loop-design.md`.

## When NOT to use a loop
- No computable metric (taste, strategy, one-shot designs) → use normal
  iterative development with human review.
- Eval is minutes-long and can't be cascaded → fix the evaluator first
  (cheap gate stage → expensive scoring stage), or budget very few
  iterations.
- The objective is "explore", not "optimize" → use research workflows; a
  ratchet needs a direction.

## Pointers
- FunSearch: Romera-Paredes et al., Nature 625 (2024)
- AlphaEvolve: DeepMind technical report (2025)
- Anthropic: "Building effective agents"; "Demystifying evals for AI
  agents"; Claude Code best practices ("give Claude a way to verify its
  work"; subagent task spec: objective, output format, tool guidance,
  boundaries)
- Open engines worth knowing: OpenEvolve, ShinkaEvolve (native Claude,
  Claude-Code skills, headless-subscription provider), AIDE (journal/tree)
