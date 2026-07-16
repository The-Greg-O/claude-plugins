"""Tests for loop.py build_audit_record — the harness as sole writer of
loop_audit.jsonl. Record assembly (field extraction from the claude CLI
result event, policy hashing, measured-field passthrough) must be tested
python, not bash-embedded generated source.
"""

import hashlib
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loop  # noqa: E402


class TestBuildAuditRecord(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.policy = os.path.join(self.tmp.name, "POLICY.md")
        with open(self.policy, "w") as f:
            f.write("policy content v1\n")
        self.result = os.path.join(self.tmp.name, "result.json")
        with open(self.result, "w") as f:
            json.dump({"type": "result", "is_error": False,
                       "duration_api_ms": 1234, "num_turns": 7,
                       "result": "x" * 300,
                       "usage": {"input_tokens": 111, "output_tokens": 22}}, f)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, **over):
        kw = dict(iter_n=5, model="sonnet", ts_start="2026-07-15T10:00:00",
                  wall_s=61, exit_code=0, policy_path=self.policy,
                  result_path=self.result)
        kw.update(over)
        return loop.build_audit_record(**kw)

    def test_assembles_measured_and_parsed_fields(self):
        rec = self.build()
        self.assertEqual(rec["iter"], 5)
        self.assertEqual(rec["model"], "sonnet")
        self.assertEqual(rec["ts_start"], "2026-07-15T10:00:00")
        self.assertEqual(rec["wall_s"], 61)
        self.assertEqual(rec["exit"], 0)
        self.assertEqual(rec["api_ms"], 1234)
        self.assertEqual(rec["turns"], 7)
        self.assertEqual(rec["in_tokens"], 111)
        self.assertEqual(rec["out_tokens"], 22)
        self.assertEqual(rec["result_tail"], "x" * 200)   # last 200 chars

    def test_policy_sha_is_sha256_prefix_of_policy_file(self):
        expected = hashlib.sha256(
            open(self.policy, "rb").read()).hexdigest()[:12]
        self.assertEqual(self.build()["policy_sha"], expected)

    def test_phase_meta_included_only_when_given(self):
        self.assertEqual(self.build(phase="meta")["phase"], "meta")
        self.assertNotIn("phase", self.build())

    def test_missing_result_file_keeps_measured_fields(self):
        rec = self.build(exit_code=1,
                         result_path=os.path.join(self.tmp.name, "absent.json"))
        for k in ("api_ms", "turns", "in_tokens", "out_tokens"):
            self.assertIsNone(rec[k])
        self.assertEqual(rec["result_tail"], "")
        self.assertEqual(rec["exit"], 1)
        self.assertEqual(rec["wall_s"], 61)

    def test_malformed_result_file_tolerated(self):
        bad = os.path.join(self.tmp.name, "bad.json")
        with open(bad, "w") as f:
            f.write("not json{")
        rec = self.build(result_path=bad)
        self.assertIsNone(rec["in_tokens"])
        self.assertEqual(rec["result_tail"], "")

    def test_missing_policy_file_yields_null_sha_not_crash(self):
        rec = self.build(policy_path=os.path.join(self.tmp.name, "gone.md"))
        self.assertIsNone(rec["policy_sha"])

    def test_no_policy_file_omits_the_key_entirely(self):
        # meta-pass rows carry no policy_sha (schema-compatible with
        # pre-refactor rows; meta spend is not attributable to a policy)
        rec = self.build(policy_path=None, phase="meta")
        self.assertNotIn("policy_sha", rec)

    def test_json_serializable_single_line(self):
        line = json.dumps(self.build())
        self.assertNotIn("\n", line)
        self.assertEqual(json.loads(line)["iter"], 5)


if __name__ == "__main__":
    unittest.main()
