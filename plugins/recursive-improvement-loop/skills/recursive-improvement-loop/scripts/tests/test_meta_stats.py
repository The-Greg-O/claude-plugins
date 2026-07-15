"""Tests for loop.py compute_meta_stats — the P0 meta-fitness readout.

Windowing contract:
- window = the last K SUCCESSFUL runner iterations (audit rows, exit==0);
  the window opens at the ts_start of the earliest of those rows.
- token cost inside the window counts EVERY audit row from the open
  timestamp on, including failed iterations (failures burn tokens too).
- evals in the window = non-baseline results with ts >= the open timestamp.
- with no audit data at all, fall back to the last K evals; token-based
  stats become None.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loop  # noqa: E402

CFG = {"name": "t", "eval_cmd": "x", "primary_metric": "score",
       "direction": "maximize"}
CFG_MIN = dict(CFG, direction="minimize")


def ts(minute, hour=10):
    return f"2026-07-15T{hour:02d}:{minute:02d}:00"


def result(i, when, promoted=False, gate=True, lineage="L1", primary=None,
           delta=None, hyp="", metrics=None):
    rec = {"id": f"c{i:03d}", "iter": i, "ts": when, "lineage": lineage,
           "gate_passed": gate, "promoted": promoted, "primary": primary,
           "hypothesis": hyp, "metrics": metrics or {}}
    if delta is not None:
        rec["delta_vs_champion"] = {"score": delta}
    return rec


def audit(i, when, exit_code=0, tin=0, tout=0, sha=None, wall=60):
    rec = {"iter": i, "ts_start": when, "exit": exit_code, "wall_s": wall,
           "in_tokens": tin, "out_tokens": tout}
    if sha:
        rec["policy_sha"] = sha
    return rec


class TestEmpty(unittest.TestCase):
    def test_no_data(self):
        s = loop.compute_meta_stats(CFG, [], [], window=10)
        self.assertEqual(s["evals"], 0)
        self.assertEqual(s["promotions"], 0)
        self.assertIsNone(s["promotions_per_eval"])
        self.assertIsNone(s["gate_fail_rate"])
        self.assertIsNone(s["mean_promoted_delta"])
        self.assertIsNone(s["hypothesis_repeat_rate"])
        self.assertIsNone(s["lineage_entropy_bits"])


class TestWindowing(unittest.TestCase):
    def test_window_opens_at_kth_last_successful_iteration(self):
        audits = [audit(1, ts(0)), audit(2, ts(10)), audit(3, ts(20))]
        results = [result(1, ts(1)), result(2, ts(11)), result(3, ts(21))]
        s = loop.compute_meta_stats(CFG, results, audits, window=2)
        self.assertEqual(s["window_iters"], 2)
        self.assertEqual(s["evals"], 2)          # evals at ts(11), ts(21)

    def test_failed_iterations_do_not_shrink_window_but_cost_counts(self):
        audits = [audit(1, ts(0), tin=100, tout=10),
                  audit(2, ts(10), tin=100, tout=10),
                  audit(3, ts(15), exit_code=1, tin=70, tout=7),   # failure inside
                  audit(3, ts(20), tin=100, tout=10)]
        results = [result(1, ts(1)), result(2, ts(11)), result(3, ts(21))]
        s = loop.compute_meta_stats(CFG, results, audits, window=2)
        self.assertEqual(s["window_iters"], 2)                 # successes only
        self.assertEqual(s["tokens"]["total"], 100 + 10 + 70 + 7 + 100 + 10)

    def test_baselines_never_counted_as_evals(self):
        audits = [audit(1, ts(0))]
        results = [dict(result(0, ts(1)), lineage="baseline"),
                   result(1, ts(2))]
        s = loop.compute_meta_stats(CFG, results, audits, window=5)
        self.assertEqual(s["evals"], 1)

    def test_fallback_without_audit_uses_last_k_evals(self):
        results = [result(i, ts(i)) for i in range(1, 6)]
        s = loop.compute_meta_stats(CFG, results, [], window=3)
        self.assertIsNone(s["window_iters"])
        self.assertEqual(s["evals"], 3)
        self.assertIsNone(s["tokens"])
        self.assertIsNone(s["champion_improvement_per_mtoken"])


class TestRates(unittest.TestCase):
    def test_promotions_per_eval_and_mean_promoted_delta(self):
        audits = [audit(1, ts(0))]
        results = [result(1, ts(1), promoted=True, primary=1.0, delta=0.10),
                   result(2, ts(2)),
                   result(3, ts(3), promoted=True, primary=1.3, delta=0.30),
                   result(4, ts(4))]
        s = loop.compute_meta_stats(CFG, results, audits, window=5)
        self.assertEqual(s["promotions"], 2)
        self.assertAlmostEqual(s["promotions_per_eval"], 0.5)
        self.assertAlmostEqual(s["mean_promoted_delta"], 0.20)

    def test_gate_fail_rate(self):
        audits = [audit(1, ts(0))]
        results = [result(1, ts(1), gate=False),
                   result(2, ts(2)),
                   result(3, ts(3), gate=False),
                   result(4, ts(4))]
        s = loop.compute_meta_stats(CFG, results, audits, window=5)
        self.assertAlmostEqual(s["gate_fail_rate"], 0.5)


class TestChampionImprovement(unittest.TestCase):
    def test_maximize_uses_pre_window_champion_as_base(self):
        audits = [audit(2, ts(10))]
        results = [result(1, ts(1), promoted=True, primary=1.0),   # pre-window
                   result(2, ts(11), promoted=True, primary=1.2),
                   result(3, ts(12), promoted=True, primary=1.5)]
        s = loop.compute_meta_stats(CFG, results, audits, window=1)
        self.assertAlmostEqual(s["champion_improvement"], 0.5)

    def test_minimize_is_signed_positive_when_better(self):
        audits = [audit(2, ts(10))]
        results = [result(1, ts(1), promoted=True, primary=10.0),
                   result(2, ts(11), promoted=True, primary=8.0)]
        s = loop.compute_meta_stats(CFG_MIN, results, audits, window=1)
        self.assertAlmostEqual(s["champion_improvement"], 2.0)

    def test_first_ever_promotion_measures_from_first_champion(self):
        audits = [audit(1, ts(0))]
        results = [result(1, ts(1), promoted=True, primary=1.0),
                   result(2, ts(2), promoted=True, primary=1.4)]
        s = loop.compute_meta_stats(CFG, results, audits, window=5)
        self.assertAlmostEqual(s["champion_improvement"], 0.4)

    def test_no_promotions_means_zero_improvement(self):
        audits = [audit(1, ts(0))]
        results = [result(1, ts(1)), result(2, ts(2))]
        s = loop.compute_meta_stats(CFG, results, audits, window=5)
        self.assertEqual(s["champion_improvement"], 0.0)

    def test_per_mtoken_normalization(self):
        audits = [audit(1, ts(0), tin=400_000, tout=100_000)]
        results = [result(1, ts(1), promoted=True, primary=1.0),
                   result(2, ts(2), promoted=True, primary=1.25)]
        s = loop.compute_meta_stats(CFG, results, audits, window=5)
        # 0.25 improvement over 0.5 Mtoken -> 0.5 per Mtoken
        self.assertAlmostEqual(s["champion_improvement_per_mtoken"], 0.5)


class TestDiversity(unittest.TestCase):
    def test_lineage_entropy_uniform_two_lineages_is_one_bit(self):
        audits = [audit(1, ts(0))]
        results = [result(1, ts(1), lineage="a"), result(2, ts(2), lineage="b"),
                   result(3, ts(3), lineage="a"), result(4, ts(4), lineage="b")]
        s = loop.compute_meta_stats(CFG, results, audits, window=5)
        self.assertAlmostEqual(s["lineage_entropy_bits"], 1.0)
        self.assertEqual(s["lineage_counts"], {"a": 2, "b": 2})

    def test_single_lineage_entropy_zero(self):
        audits = [audit(1, ts(0))]
        results = [result(1, ts(1)), result(2, ts(2))]
        s = loop.compute_meta_stats(CFG, results, audits, window=5)
        self.assertAlmostEqual(s["lineage_entropy_bits"], 0.0)

    def test_hypothesis_repeat_rate_fuzzy_match_against_prior_history(self):
        audits = [audit(2, ts(10))]
        results = [
            result(1, ts(1), hyp="use delta encoding on the residual stream"),
            # in-window: near-duplicate of the pre-window hypothesis
            result(2, ts(11), hyp="use delta encoding on residual stream"),
            # in-window: genuinely new
            result(3, ts(12), hyp="switch to arithmetic coder with adaptive model"),
        ]
        s = loop.compute_meta_stats(CFG, results, audits, window=1)
        self.assertAlmostEqual(s["hypothesis_repeat_rate"], 0.5)


class TestTrainHoldoutGap(unittest.TestCase):
    def test_mean_gap_per_metric_stem(self):
        audits = [audit(1, ts(0))]
        results = [
            result(1, ts(1), metrics={"score_train": 0.9, "score_holdout": 0.8}),
            result(2, ts(2), metrics={"score_train": 0.7, "score_holdout": 0.7}),
            result(3, ts(3), gate=False,
                   metrics={"score_train": 9.9, "score_holdout": 0.0}),  # excluded
        ]
        s = loop.compute_meta_stats(CFG, results, audits, window=5)
        self.assertAlmostEqual(s["train_holdout_gap"]["score"], 0.05)

    def test_no_pairs_gives_empty_dict(self):
        audits = [audit(1, ts(0))]
        results = [result(1, ts(1), metrics={"score": 1.0})]
        s = loop.compute_meta_stats(CFG, results, audits, window=5)
        self.assertEqual(s["train_holdout_gap"], {})


class TestMetaRowExclusion(unittest.TestCase):
    def test_meta_rows_not_iterations_and_their_cost_tracked_separately(self):
        audits = [audit(1, ts(0), tin=100, tout=10),
                  dict(audit(0, ts(5), tin=50, tout=5), phase="meta"),
                  audit(2, ts(10), tin=100, tout=10)]
        results = [result(1, ts(1)), result(2, ts(11))]
        s = loop.compute_meta_stats(CFG, results, audits, window=2)
        self.assertEqual(s["window_iters"], 2)
        self.assertEqual(s["tokens"]["total"], 220)
        self.assertEqual(s["meta_tokens"], 55)

    def test_meta_rows_excluded_from_policy_breakdown(self):
        audits = [audit(1, ts(0), tin=100, tout=10, sha="aaa"),
                  dict(audit(0, ts(5), tin=50, tout=5, sha="bbb"), phase="meta")]
        results = [result(1, ts(1))]
        s = loop.compute_meta_stats(CFG, results, audits, window=2)
        self.assertEqual([p["policy_sha"] for p in s["policies"]], ["aaa"])


class TestMetaRatchet(unittest.TestCase):
    def test_first_check_baselines_incumbent(self):
        v, st = loop.decide_meta_ratchet(None, 1.0, eps=0.1)
        self.assertEqual(v, "baseline")
        self.assertEqual(st["incumbent_fitness"], 1.0)
        self.assertFalse(st["pending"])

    def test_pending_trial_kept_when_it_beats_incumbent_by_eps(self):
        state = {"incumbent_fitness": 1.0, "pending": True}
        v, st = loop.decide_meta_ratchet(state, 1.2, eps=0.1)
        self.assertEqual(v, "keep")
        self.assertEqual(st["incumbent_fitness"], 1.2)
        self.assertFalse(st["pending"])

    def test_pending_trial_reverted_when_within_eps(self):
        state = {"incumbent_fitness": 1.0, "pending": True}
        v, st = loop.decide_meta_ratchet(state, 1.05, eps=0.1)
        self.assertEqual(v, "revert")
        self.assertEqual(st["incumbent_fitness"], 1.0)
        self.assertFalse(st["pending"])

    def test_pending_trial_reverted_when_worse(self):
        state = {"incumbent_fitness": 1.0, "pending": True}
        v, st = loop.decide_meta_ratchet(state, 0.4, eps=0.1)
        self.assertEqual(v, "revert")
        self.assertEqual(st["incumbent_fitness"], 1.0)

    def test_no_pending_trial_refreshes_incumbent_estimate(self):
        state = {"incumbent_fitness": 2.0, "pending": False}
        v, st = loop.decide_meta_ratchet(state, 1.5, eps=0.1)
        self.assertEqual(v, "baseline")
        self.assertEqual(st["incumbent_fitness"], 1.5)


class TestPolicyBreakdown(unittest.TestCase):
    def test_evals_and_tokens_attributed_by_timestamp(self):
        audits = [audit(1, ts(0), tin=10, tout=1, sha="aaa"),
                  audit(2, ts(10), tin=20, tout=2, sha="aaa"),
                  audit(3, ts(20), tin=30, tout=3, sha="bbb"),
                  audit(4, ts(30), tin=40, tout=4, sha="bbb")]
        results = [result(1, ts(1)), result(2, ts(11), promoted=True, primary=1.0),
                   result(3, ts(21), gate=False), result(4, ts(31))]
        s = loop.compute_meta_stats(CFG, results, audits, window=10)
        by_sha = {p["policy_sha"]: p for p in s["policies"]}
        self.assertEqual(by_sha["aaa"]["iters"], 2)
        self.assertEqual(by_sha["aaa"]["evals"], 2)
        self.assertEqual(by_sha["aaa"]["promotions"], 1)
        self.assertEqual(by_sha["aaa"]["tokens_total"], 33)
        self.assertEqual(by_sha["bbb"]["evals"], 2)
        self.assertEqual(by_sha["bbb"]["gate_fails"], 1)
        self.assertEqual(by_sha["bbb"]["tokens_total"], 77)

    def test_missing_sha_grouped_as_unattributed(self):
        audits = [audit(1, ts(0), tin=5, tout=5)]
        results = [result(1, ts(1))]
        s = loop.compute_meta_stats(CFG, results, audits, window=10)
        self.assertEqual(s["policies"][0]["policy_sha"], "unattributed")


if __name__ == "__main__":
    unittest.main()
