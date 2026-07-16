#!/bin/bash
# test_runner_integration.sh — end-to-end test of runner.sh's trusted plumbing
# using a fake `claude` binary on PATH. Covers:
#   1. fresh scaffolds get the PROMPT_CORE.md + POLICY.md split (no PROMPT.md)
#   2. the per-iteration prompt is CORE + POLICY concatenated
#   3. loop_audit.jsonl records policy_sha = sha256(POLICY.md)[:12]
#   4. legacy experiments (monolithic PROMPT.md) still run, attributed to
#      sha256(PROMPT.md)[:12]
set -uo pipefail

SCRIPTS="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
FAILURES=0
fail() { echo "FAIL: $1"; FAILURES=$((FAILURES + 1)); }
pass() { echo "  ok: $1"; }

sha12() {
  python3 -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()[:12])" "$1"
}
audit_sha() {
  python3 -c "import json,sys;print(json.loads(open(sys.argv[1]).readline()).get('policy_sha',''))" "$1" 2>/dev/null
}

# fake claude: captures the -p prompt into the cwd, emits a valid result
# event. On a META-PASS prompt it edits POLICY.md (and, when
# FAKE_META_VIOLATE is set, illegally touches PROMPT_CORE.md too).
mkdir -p "$WORK/bin"
cat > "$WORK/bin/claude" <<'FAKE'
#!/bin/bash
PROMPT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p) PROMPT="$2"; printf '%s' "$2" > "${FAKE_CAPTURE:-prompt_capture.txt}"; shift 2 ;;
    *) shift ;;
  esac
done
if [[ "$PROMPT" == *"META-PASS"* ]]; then
  echo "TRIAL-EDIT marker $RANDOM" >> POLICY.md
  [[ -n "${FAKE_META_VIOLATE:-}" ]] && echo "corrupted" >> PROMPT_CORE.md
fi
echo '{"type":"result","is_error":false,"usage":{"input_tokens":1000,"output_tokens":100},"num_turns":1,"duration_api_ms":10,"result":"fake iteration done"}'
FAKE
chmod +x "$WORK/bin/claude"
export PATH="$WORK/bin:$PATH"

# ---------- scenario 1: fresh scaffold runs in split (CORE + POLICY) mode
EXP1="$WORK/exp1"
python3 "$SCRIPTS/loop.py" init "$EXP1" >/dev/null
[[ -f "$EXP1/PROMPT_CORE.md" ]] && pass "init scaffolds PROMPT_CORE.md" \
                                || fail "init should scaffold PROMPT_CORE.md"
[[ -f "$EXP1/POLICY.md" ]]      && pass "init scaffolds POLICY.md" \
                                || fail "init should scaffold POLICY.md"
[[ ! -f "$EXP1/PROMPT.md" ]]    && pass "init no longer scaffolds PROMPT.md" \
                                || fail "init should not scaffold monolithic PROMPT.md"
[[ -f "$EXP1/_dashboard.py" ]]  && pass "init copies the _dashboard.py sibling" \
                                || fail "init should copy _dashboard.py (display travels beside the referee)"

(cd "$EXP1" && DASH_OPEN=never ./runner.sh -n 1 -p 5 -s 0 >/dev/null 2>&1) \
  || fail "runner exited nonzero in split mode"
if [[ -f "$EXP1/prompt_capture.txt" ]]; then
  grep -q "frozen protocol layer" "$EXP1/prompt_capture.txt" \
    && pass "prompt contains CORE content" || fail "prompt missing CORE content"
  grep -q "mutable strategy layer" "$EXP1/prompt_capture.txt" \
    && pass "prompt contains POLICY content" || fail "prompt missing POLICY content"
else
  fail "fake claude never received a prompt in split mode"
fi
expected="$(sha12 "$EXP1/POLICY.md" 2>/dev/null)"
got="$(audit_sha "$EXP1/loop_audit.jsonl")"
[[ -n "$expected" && "$got" == "$expected" ]] \
  && pass "audit policy_sha = sha256(POLICY.md)[:12]" \
  || fail "audit policy_sha '$got' != sha256(POLICY.md)[:12] '$expected'"
nums="$(python3 -c "
import json,sys
r=json.loads(open(sys.argv[1]).readline())
print(r.get('in_tokens'), r.get('out_tokens'), r.get('turns'), r.get('api_ms'))" \
  "$EXP1/loop_audit.jsonl" 2>/dev/null)"
[[ "$nums" == "1000 100 1 10" ]] \
  && pass "audit numeric fields parsed by the harness" \
  || fail "audit numeric fields wrong: '$nums' (expected '1000 100 1 10')"

# ---------- scenario 2: legacy monolithic PROMPT.md still runs
EXP2="$WORK/exp2"
mkdir -p "$EXP2/candidates" "$EXP2/checkpoints"
cp "$SCRIPTS/loop.py" "$SCRIPTS/runner.sh" "$SCRIPTS/_stream_view.py" "$EXP2/"
chmod +x "$EXP2/runner.sh"
cat > "$EXP2/experiment.json" <<'EOF'
{"name":"legacy","eval_cmd":"python3 evaluate.py","primary_metric":"score","direction":"maximize"}
EOF
printf '# legacy prompt\nlegacy marker text\n' > "$EXP2/PROMPT.md"
(cd "$EXP2" && DASH_OPEN=never ./runner.sh -n 1 -p 5 -s 0 >/dev/null 2>&1) \
  || fail "runner exited nonzero in legacy mode"
grep -q "legacy marker text" "$EXP2/prompt_capture.txt" 2>/dev/null \
  && pass "legacy PROMPT.md used as the prompt" || fail "legacy PROMPT.md not used"
expected="$(sha12 "$EXP2/PROMPT.md")"
got="$(audit_sha "$EXP2/loop_audit.jsonl")"
[[ "$got" == "$expected" ]] && pass "legacy policy_sha = sha256(PROMPT.md)[:12]" \
  || fail "legacy policy_sha '$got' != sha256(PROMPT.md)[:12] '$expected'"

# ---------- scenario 3: meta-pass arms a trial, blind window fails, revert
EXP3="$WORK/exp3"
python3 "$SCRIPTS/loop.py" init "$EXP3" >/dev/null
[[ -f "$EXP3/META_PROMPT.md" ]] && pass "init scaffolds META_PROMPT.md" \
                               || fail "init should scaffold META_PROMPT.md"
# -M 2: meta point at i=2 (baseline + arm trial 1) and i=4 (fitness 0 < eps
# 0.5 -> revert trial 1, then arm trial 2). Exactly ONE trial marker survives.
(cd "$EXP3" && DASH_OPEN=never FAKE_CAPTURE="$WORK/exp3_prompt.txt" \
   ./runner.sh -n 4 -p 99 -s 0 -M 2 --meta-eps 0.5 >"$WORK/exp3_runner.log" 2>&1) \
  || fail "runner exited nonzero in meta mode"
markers=$(grep -c "TRIAL-EDIT" "$EXP3/POLICY.md" 2>/dev/null || true)
[[ "$markers" == "1" ]] && pass "failed trial reverted, new trial armed (1 marker)" \
  || fail "expected exactly 1 TRIAL-EDIT marker in POLICY.md, got '$markers'"
meta_rows=$(python3 -c "
import json
rows=[json.loads(l) for l in open('$EXP3/loop_audit.jsonl') if l.strip()]
print(sum(1 for r in rows if r.get('phase')=='meta'))" 2>/dev/null)
[[ "$meta_rows" == "2" ]] && pass "2 meta-pass audit rows recorded" \
  || fail "expected 2 phase=meta audit rows, got '$meta_rows'"
pending=$(python3 -c "
import json;print(json.load(open('$EXP3/meta_state.json'))['pending'])" 2>/dev/null)
[[ "$pending" == "True" ]] && pass "meta_state pending after re-arm" \
  || fail "meta_state.json pending expected True, got '$pending'"
[[ -d "$EXP3/.git" ]] && pass "experiment brought under git for revert" \
  || fail "runner should git-init the experiment for the meta-ratchet"

# ---------- scenario 4: meta boundary violation is fully rolled back
EXP4="$WORK/exp4"
python3 "$SCRIPTS/loop.py" init "$EXP4" >/dev/null
CORE_BEFORE="$(cat "$EXP4/PROMPT_CORE.md")"
(cd "$EXP4" && DASH_OPEN=never FAKE_META_VIOLATE=1 FAKE_CAPTURE="$WORK/exp4_prompt.txt" \
   ./runner.sh -n 2 -p 99 -s 0 -M 2 >"$WORK/exp4_runner.log" 2>&1) \
  || fail "runner exited nonzero in violation scenario"
grep -q "corrupted" "$EXP4/PROMPT_CORE.md" \
  && fail "PROMPT_CORE.md corruption was NOT rolled back" \
  || pass "PROMPT_CORE.md violation rolled back"
grep -q "TRIAL-EDIT" "$EXP4/POLICY.md" \
  && fail "violating meta-pass POLICY.md edit was NOT rolled back" \
  || pass "violating POLICY.md edit rolled back"
pending=$(python3 -c "
import json;print(json.load(open('$EXP4/meta_state.json'))['pending'])" 2>/dev/null)
[[ "$pending" == "False" ]] && pass "violating trial not armed" \
  || fail "meta_state pending expected False after violation, got '$pending'"

echo
if ((FAILURES)); then echo "$FAILURES assertion(s) FAILED"; exit 1; fi
echo "all runner integration assertions passed"
