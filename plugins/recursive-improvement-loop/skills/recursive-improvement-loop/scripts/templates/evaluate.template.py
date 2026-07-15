#!/usr/bin/env python3
"""evaluate.py — the ONLY domain-specific code in a {{NAME}} experiment.

Contract (loop.py calls this as `<eval_cmd> <candidate_path>`):
  - Run the candidate against FROZEN evaluation data / scenarios.
  - Print exactly ONE JSON object as the FINAL stdout line:
      {"gate_passed": bool,      # hard correctness/validity gate
       "gate_error": "...",      # why, when gate_passed is false
       "metrics": {"<primary>": <num>, ...}}   # everything measurable
  - Exit 0 (a nonzero exit or bad JSON is recorded as a gate failure).

DESIGN RULES (see references/evaluator-guide.md for the full reasoning):
  1. The GATE is pass/fail correctness, never a score. Test adversarial /
     edge inputs, not just typical ones. If you can't gate it, you can't
     loop on it.
  2. Evaluate on FROZEN data (hash it; fail loudly if it changes). Keep a
     HOLDOUT slice the candidates are never tuned against, and make the
     primary metric the HOLDOUT number — the scaffolded default is
     score_holdout. Name split metrics <stem>_train / <stem>_holdout so
     `loop.py meta-stats` can watch the overfit gap. (No meaningful
     holdout in your domain — e.g. a deterministic exhaustive metric?
     Rename to a single metric and update experiment.json.)
  3. Anti-reward-hacking: run candidates in a subprocess with a timeout;
     forbid them from reading the evaluation data or this file (static
     checks / sandboxing); never let the candidate print its own metrics.
  4. Timing metrics: best-of-K after a warmup, and treat differences inside
     your measured noise band as ties.
"""

import json
import sys


def main():
    candidate_path = sys.argv[1]

    # --- load the candidate (subprocess/sandbox it for untrusted code!) ---
    # --- run it against your frozen scenarios ---------------------------
    # --- compute gate + metrics ------------------------------------------

    result = {
        "gate_passed": False,
        "gate_error": f"evaluate.py is still the unedited template (got {candidate_path})",
        "metrics": {"score_holdout": 0.0, "score_train": 0.0},
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
