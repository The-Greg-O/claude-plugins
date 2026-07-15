# Meta-loop design: improvement of the improvement process

*Gap analysis against Weco AIDE² ("First Evidence of Recursive Self-Improvement", 2026-07-14), drafted 2026-07-15.*

*Status 2026-07-15: **Phase 0 implemented** (PROMPT_CORE/POLICY split, `policy_sha` audit attribution, `loop.py meta-stats`) and **Phase 1 prototyped** (`runner.sh -M K`: blind-window meta-pass with `loop.py meta-ratchet` keep/revert via git, write boundary enforced by runner diff inspection, `phase=meta` audit rows excluded from window fitness). Steal-now #3 (holdout-primary scaffold default) done. Phase 2 and steal-now #1/#2 remain. As of this date Weco has published only the blog post — no technical report, no AIDE85 release, no third-party replication.*

## Where this plugin already stands

Structurally, this plugin **is** Weco's inner loop: fresh-context iteration agent (their AIDE₀ ≈ our `claude -p` per iteration), trusted referee (`loop.py` + `evaluate.py` ≈ their private-score selection), champion ratchet with epsilon (their hill-climb), plateau-based stop. Several AIDE² "discoveries" are already present in hand-written form:

| AIDE² evolved mechanism | Plugin equivalent today |
|---|---|
| Minimal per-operator context (16× compression) | Bounded read set in PROMPT.md ("do not exceed this set") |
| Bandit over draft subtrees / fork-on-stall | Island rule + every-5th graveyard revival + plateau-forced dormant revival |
| Private selection score | Holdout-as-primary-metric guidance in evaluator-guide |
| Anti-reward-hacking stack | Worker/referee split, planted-bug rail test, byte cap, no-peeking checks |

What the plugin does **not** have is the outer loop: the policy that decides *how to iterate* (PROMPT.md heuristics, model schedule, notebook curation) is fixed prompt text, tuned only by a human between campaigns.

## The mapping

Weco's outer loop rewrites the inner agent's *code*. Our inner agent's "code" is the **policy surface**: `PROMPT.md` (protocol + island-rule constants + standing objectives), runner knobs (`-m/-d/-D/-e/-t`), and notebook curation. Crucially, `runner.sh` re-reads `PROMPT.md` fresh every iteration — the policy is hot-swappable with zero harness changes. The referee stays frozen for the meta-loop exactly as it does for the inner loop; otherwise metric comparability dies.

The three AIDE² mechanisms translate as:

1. **Public–private split → blind future windows.** A policy variant must be judged on inner iterations that happen *after* it's proposed (it can't game history), scored only from harness-written files (`results.jsonl`, `loop_audit.jsonl`), never from agent self-reports.
2. **Fixed cost budget → per-token meta-fitness.** Meta-metric must be normalized: champion-delta or promotions **per million tokens** (both already logged in `loop_audit.jsonl`), else the meta-loop trivially learns "always opus, more turns."
3. **Heterogeneous portfolio → cross-campaign tier.** Within one experiment there is one task; generality pressure must come from scoring template changes across *multiple campaigns* (see Phase 2).

## Sobering priors from their negative results

- ~9/10 outer proposals rejected; winning mechanisms were "compositions of rather simple mechanisms." Expect most policy mutations to lose; bias toward small diffs.
- Their per-candidate outer evaluation = several full cost-capped inner runs. Our analog: a meta-eval costs a window of K inner iterations. **Within-campaign meta-looping only pays off on long campaigns (≳100 iterations); short campaigns should rely on the cross-campaign tier.**
- Nested loops compound noise — they couldn't even show their efficiency claim was significant. Meta-promotion epsilon must be ≥ 2× the observed window-to-window noise band, same discipline `design-principles.md` already demands of the inner ratchet.
- No ignition found. The goal here is a better plugin, not a takeoff; L1-style "net positive vs hand-tuning" is the honest target.

## Proposed design (phased)

### Phase 0 — plumbing (trusted layer, small)
- **Split the prompt**: `PROMPT_CORE.md` (frozen: trust boundary, never-edit list, no-self-reporting, probe accounting) + `POLICY.md` (mutable: island-rule constants, hypothesis heuristics, standing objectives, read-set size). `runner.sh` concatenates. *This is the anti-reward-hacking move: the meta-agent must never be able to "improve" throughput by loosening trust language.* Weco's lesson says an unguarded meta-loop will find that hole not from malice but from pressure.
- **Attribution**: add `policy_sha` (hash of POLICY.md) to every `loop_audit.jsonl` record — one-line runner change; without it meta-fitness can't be credited.
- **`loop.py meta-stats --window K`**: promotions/eval, mean promoted delta, champion-delta per Mtoken, gate-fail rate, lineage entropy, fuzzy hypothesis-repeat rate, train↔holdout gap. All computable from existing files; the dashboard's "agent economics" panel already plots the raw ingredients. The SKILL.md failure-mode table becomes the interpretation rubric.

### Phase 1 — within-campaign meta-pass
- Every K inner iterations (default 10–15, aligned with plateau windows), `runner.sh` spawns a **meta-pass**: `claude -p META_PROMPT.md` reading meta-stats + notebook, proposing ONE bounded edit to `POLICY.md` (diff-size cap, like the candidate byte cap).
- **Meta-ratchet**: the edit runs blind for the next K iterations; keep iff meta-stats beat the incumbent window by ≥ meta-epsilon, else auto-revert (`git` in the experiment dir is the cheap mechanism). Alternating A/B windows if noise demands it.
- This generalizes two things that already exist: the deep-pass schedule (a hard-coded two-tier policy) and `/status` (a read-only meta-evaluator that "recommends the next action" — promotion from advisor to actor).

### Phase 2 — cross-campaign template evolution
- Each campaign ends by emitting a standardized `campaign-report.json` (final meta-stats + charter shape: gate strictness, lineage count, noise config, iterations-to-plateau).
- A repo-level loop treats `PROMPT.template.md` / interview phases as candidates, scored across campaign reports — the heterogeneous-portfolio pressure AIDE² used to force generality.
- **Human ratifies template promotions via PR.** This tier moves slowly and matters most; it's where "the plugin gets better at making loops" actually lives.

### Non-goals
- Meta-mutation of `loop.py`/`evaluate.py`/frozen data — never; referee integrity is the product.
- Chasing ignition (meta-meta loops) — Weco looked and found nothing there; noise compounds faster than insight.
- Raw-throughput meta-fitness — cost-normalized only.

## Steal-now items (no meta-loop required)
1. **Lineage scoreboard injection**: have `loop.py` compute per-lineage stats from `results.jsonl` and inject a 5-line scoreboard into the prompt — upgrades the island rule from prose to data (their bandit result says this is where the win was).
2. **Context-compression experiment**: try halving the bounded read set (`tail -n 15` instead of 30) on a live campaign and watch improvement-per-token — their 16× compression finding says leaner context + more iterations beats richer context.
3. **Public–private hardening**: make holdout-primary the default in `evaluate.template.py` rather than guidance in the evaluator guide.
