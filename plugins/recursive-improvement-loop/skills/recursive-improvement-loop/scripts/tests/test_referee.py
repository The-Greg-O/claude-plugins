"""In-process tests for the referee's core: the champion ratchet
(eval_candidate), the drift guard (verify_champion), the evaluator
contract (run_evaluator), and the plateau counter — exercised against
tempdir fixtures through the same functions the CLI dispatches to.

The fixture's evaluator is an echo: each candidate file CONTAINS the
contract JSON the evaluator prints, so every test controls its own
gate/metrics. Special candidate bodies trigger evaluator crash/timeout.
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loop  # noqa: E402

EVALUATOR = """\
import sys, time
body = open(sys.argv[1]).read()
if "CRASH" in body:
    sys.exit(3)
if "SLEEP" in body:
    time.sleep(5)
print(body.strip().splitlines()[-1])
"""


class RefereeFixture(unittest.TestCase):
    config = {}          # per-class experiment.json overrides

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.prev = os.getcwd()
        os.chdir(self.tmp)
        os.makedirs("candidates")
        with open("evaluate.py", "w") as f:
            f.write(EVALUATOR)
        cfg = {"name": "fixture", "eval_cmd": "python3 evaluate.py",
               "primary_metric": "score", "direction": "maximize"}
        cfg.update(self.config)
        with open("experiment.json", "w") as f:
            json.dump(cfg, f)
        self.cfg = loop.load_config()
        self._dash = mock.patch.object(loop, "_refresh_dashboard")
        self._dash.start()

    def tearDown(self):
        self._dash.stop()
        os.chdir(self.prev)
        shutil.rmtree(self.tmp)

    def write_candidate(self, cid, body=None, gate=True, score=None, error=""):
        if body is None:
            metrics = {} if score is None else {"score": score}
            body = json.dumps({"gate_passed": gate, "gate_error": error,
                               "metrics": metrics})
        path = os.path.join("candidates", cid)
        with open(path, "w") as f:
            f.write(body)
        return path

    def run_eval(self, cid, baseline=False, meta=None, **cand_kw):
        path = self.write_candidate(cid, **cand_kw)
        with contextlib.redirect_stdout(io.StringIO()):
            return loop.eval_candidate(self.cfg, path, meta or {},
                                       baseline=baseline)

    def leaderboard(self):
        return json.load(open("leaderboard.json"))


class TestChampionRatchet(RefereeFixture):
    def test_first_gate_passing_candidate_becomes_champion(self):
        rec = self.run_eval("seed", score=1.0)
        self.assertTrue(rec["promoted"])
        lb = self.leaderboard()
        self.assertEqual(lb["champion"]["id"], "seed")
        self.assertEqual(lb["champion"]["primary"], 1.0)
        self.assertEqual(len(lb["history"]), 1)

    def test_equal_score_does_not_promote(self):
        self.run_eval("seed", score=1.0)
        rec = self.run_eval("c2", score=1.0)
        self.assertFalse(rec["promoted"])
        self.assertEqual(self.leaderboard()["champion"]["id"], "seed")

    def test_better_score_promotes_and_appends_history(self):
        self.run_eval("seed", score=1.0)
        rec = self.run_eval("c2", score=1.5)
        self.assertTrue(rec["promoted"])
        lb = self.leaderboard()
        self.assertEqual(lb["champion"]["id"], "c2")
        self.assertEqual([h["id"] for h in lb["history"]], ["seed", "c2"])

    def test_delta_vs_champion_per_metric(self):
        self.run_eval("seed", body=json.dumps(
            {"gate_passed": True, "metrics": {"score": 1.0, "aux": 2.0}}))
        rec = self.run_eval("c2", body=json.dumps(
            {"gate_passed": True, "metrics": {"score": 1.5, "aux": 1.8}}))
        self.assertAlmostEqual(rec["delta_vs_champion"]["score"], 0.5)
        self.assertAlmostEqual(rec["delta_vs_champion"]["aux"], -0.2)

    def test_gate_fail_recorded_and_champion_stands(self):
        self.run_eval("seed", score=1.0)
        rec = self.run_eval("bad", gate=False, error="planted bug")
        self.assertFalse(rec["gate_passed"])
        self.assertFalse(rec["promoted"])
        self.assertEqual(rec["gate_error"], "planted bug")
        self.assertEqual(self.leaderboard()["champion"]["id"], "seed")

    def test_gate_pass_without_primary_metric_becomes_gate_fail(self):
        rec = self.run_eval("nometric", score=None)
        self.assertFalse(rec["gate_passed"])
        self.assertIn("missing primary_metric", rec["gate_error"])

    def test_baseline_recorded_without_promotion(self):
        rec = self.run_eval("ref", score=9.9, baseline=True)
        self.assertFalse(rec["promoted"])
        self.assertEqual(rec["iter"], 0)
        lb = self.leaderboard()
        self.assertEqual(lb["baselines"]["ref"], 9.9)
        self.assertIsNone(lb["champion"])

    def test_results_jsonl_records_every_attempt(self):
        self.run_eval("seed", score=1.0)
        self.run_eval("bad", gate=False)
        rows = [json.loads(l) for l in open("results.jsonl")]
        self.assertEqual([r["id"] for r in rows], ["seed", "bad"])
        self.assertEqual([r["iter"] for r in rows], [1, 2])

    def test_leaderboard_write_leaves_no_tmp_behind(self):
        self.run_eval("seed", score=1.0)
        self.assertFalse(os.path.exists("leaderboard.json.tmp"))

    def test_plateau_counts_evals_since_last_promotion(self):
        self.run_eval("seed", score=1.0)
        self.run_eval("c2", score=0.5)
        self.run_eval("c3", score=0.4)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(loop.plateau(3), 0)   # 2 misses < 3
            self.run_eval("c4", score=0.3)
            self.assertEqual(loop.plateau(3), 1)   # 3 misses >= 3


class TestAbsoluteEpsilon(RefereeFixture):
    config = {"promote_eps": 0.5}

    def test_improvement_within_epsilon_is_a_tie(self):
        self.run_eval("seed", score=1.0)
        self.assertFalse(self.run_eval("c2", score=1.4)["promoted"])
        self.assertTrue(self.run_eval("c3", score=1.6)["promoted"])


class TestRelativeEpsilon(RefereeFixture):
    config = {"promote_eps_rel": 0.1}

    def test_epsilon_scales_with_champion(self):
        self.run_eval("seed", score=10.0)
        self.assertFalse(self.run_eval("c2", score=10.9)["promoted"])
        self.assertTrue(self.run_eval("c3", score=11.2)["promoted"])


class TestMinimizeDirection(RefereeFixture):
    config = {"direction": "minimize"}

    def test_lower_is_better(self):
        self.run_eval("seed", score=10.0)
        self.assertFalse(self.run_eval("worse", score=11.0)["promoted"])
        self.assertTrue(self.run_eval("faster", score=9.0)["promoted"])


class TestEmbedTheAnswerGuard(RefereeFixture):
    config = {"max_candidate_bytes": 64}

    def test_oversized_candidate_rejected_before_evaluator(self):
        body = json.dumps({"gate_passed": True, "metrics": {"score": 99.0}})
        rec = self.run_eval("fat", body=body + "#" * 200)
        self.assertFalse(rec["gate_passed"])
        self.assertIn("embed-the-answer", rec["gate_error"])
        self.assertFalse(os.path.exists("leaderboard.json"))  # never promoted


class TestDriftGuard(RefereeFixture):
    # each test asserts on the abort MESSAGE, not just SystemExit — a bare
    # SystemExit assertion cannot tell the intended abort from an
    # unrelated FATAL (mutation-verified during review)

    def test_champion_primary_drift_aborts(self):
        self.run_eval("seed", score=1.0)
        self.write_candidate("seed", score=2.0)   # environment "drifted"
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                loop.eval_candidate(self.cfg, self.write_candidate(
                    "c2", score=3.0), {})
        self.assertIn("drift", str(cm.exception))

    def test_champion_gate_regression_aborts(self):
        self.run_eval("seed", score=1.0)
        self.write_candidate("seed", gate=False, error="rot")
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                loop.eval_candidate(self.cfg, self.write_candidate(
                    "c2", score=3.0), {})
        self.assertIn("no longer passes the gate", str(cm.exception))

    def test_missing_champion_file_aborts(self):
        self.run_eval("seed", score=1.0)
        os.remove(os.path.join("candidates", "seed"))
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                loop.eval_candidate(self.cfg, self.write_candidate(
                    "c2", score=3.0), {})
        self.assertIn("missing", str(cm.exception))


class TestGateOnlyReverify(RefereeFixture):
    config = {"reverify_champion": "gate-only"}

    def test_primary_drift_tolerated_when_gate_still_passes(self):
        self.run_eval("seed", score=1.0)
        self.write_candidate("seed", score=2.0)   # noisy primary drifted
        rec = self.run_eval("c2", score=1.5)      # must not abort
        self.assertTrue(rec["promoted"])          # vs STORED champion 1.0

    def test_gate_regression_still_aborts_in_gate_only_mode(self):
        # gate-only skips the drift COMPARISON, never the gate CHECK —
        # it is the only protection gate-only mode retains
        self.run_eval("seed", score=1.0)
        self.write_candidate("seed", gate=False, error="rot")
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                loop.eval_candidate(self.cfg, self.write_candidate(
                    "c2", score=1.5), {})
        self.assertIn("no longer passes the gate", str(cm.exception))


class TestEvaluatorContract(RefereeFixture):
    def test_invalid_contract_json_is_gate_fail(self):
        rec = self.run_eval("garbled", body="not json{")
        self.assertFalse(rec["gate_passed"])
        self.assertIn("not valid contract JSON", rec["gate_error"])

    def test_evaluator_crash_is_gate_fail(self):
        rec = self.run_eval("boom", body="CRASH")
        self.assertFalse(rec["gate_passed"])
        self.assertIn("evaluator crashed", rec["gate_error"])

    def test_evaluator_timeout_is_gate_fail(self):
        with open("experiment.json") as f:
            cfg = json.load(f)
        cfg["eval_timeout_s"] = 0.4
        with open("experiment.json", "w") as f:
            json.dump(cfg, f)
        self.cfg = loop.load_config()
        rec = self.run_eval("slow", body="SLEEP")
        self.assertFalse(rec["gate_passed"])
        self.assertIn("exceeded", rec["gate_error"])

    def test_missing_candidate_file_aborts_without_recording(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                loop.eval_candidate(self.cfg, "candidates/ghost", {})
        self.assertFalse(os.path.exists("results.jsonl"))


if __name__ == "__main__":
    unittest.main()
