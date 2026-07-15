#!/bin/bash
# runner.sh — headless driver: one FRESH `claude -p` process per iteration.
#
# Context never accumulates across iterations; all state lives on disk
# (LAB_NOTEBOOK.md, results.jsonl, leaderboard.json). The harness owns the
# only stop decision (statistical plateau); --max-iters is runaway insurance.
#
# The per-iteration prompt is PROMPT_CORE.md (frozen protocol + trust rules)
# + POLICY.md (mutable strategy — the meta-loop's surface), concatenated
# fresh each iteration so policy edits hot-swap without touching the runner.
# Each audit record carries policy_sha = sha256 of the mutable file, so
# meta-fitness can be attributed to the policy that produced it. A legacy
# monolithic PROMPT.md still works (attributed as its own policy).
#
# Usage:
#   ./runner.sh [options]
#     -n, --max-iters N     hard cap on iterations            (default 25)
#     -p, --plateau N       stop after N evals w/o promotion  (default 10)
#     -m, --model M         model for routine iterations      (default sonnet)
#     -d, --deep-model M    model for the periodic deep pass  (default opus)
#     -D, --deep-every N    deep pass every Nth iteration     (default 10; 0=off)
#     -e, --effort LEVEL    low|medium|high|xhigh|max, or "ultracode" =
#                           xhigh + multi-agent orchestration opt-in
#                           (Workflow keyword injected into the prompt)
#     -t, --max-turns N     per-iteration turn cap            (default 50)
#     -s, --sleep SECONDS   pause between iterations          (default 5)
#     -M, --meta-every N    meta-pass every Nth iteration     (default 0=off):
#                           a bounded agent may propose ONE edit to POLICY.md,
#                           trialled BLIND for the next N iterations and
#                           auto-reverted (git) unless its window beats the
#                           incumbent's by --meta-eps (loop.py meta-ratchet)
#         --meta-eps E      meta-promotion epsilon             (default 0.0;
#                           set >= 2x your window-to-window noise)
#         --meta-max-diff N max changed lines per policy edit  (default 30)
#         --meta-model M    model for the meta-pass            (default: deep model)
#         --dry-run         print the claude command and exit
#   Legacy positional form still works: ./runner.sh [max_iters] [plateau_n]
#
# Env: DASH_OPEN=every  → pop each checkpoint PNG as it lands.
#
# Runs on your logged-in Claude subscription (no API key needed).
# Uses --dangerously-skip-permissions: the agent edits candidates/ and the
# notebook, runs the harness. Trust boundaries are enforced by CLAUDE.md
# rules + the harness's static checks + champion regression guard.

set -uo pipefail
cd "$(dirname "$0")" || exit 1

MAX_ITERS=25
PLATEAU_N=10
MODEL="${MODEL:-sonnet}"
DEEP_MODEL="${DEEP_MODEL:-opus}"
DEEP_EVERY=10
EFFORT=""
MAX_TURNS=50
SLEEP_S=5
META_EVERY=0
META_EPS=0.0
META_MAX_DIFF=30
META_MODEL=""
DRY_RUN=0
AUDIT=loop_audit.jsonl

# legacy positionals
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then MAX_ITERS="$1"; shift; fi
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then PLATEAU_N="$1"; shift; fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--max-iters)  MAX_ITERS="$2"; shift 2 ;;
    -p|--plateau)    PLATEAU_N="$2"; shift 2 ;;
    -m|--model)      MODEL="$2"; shift 2 ;;
    -d|--deep-model) DEEP_MODEL="$2"; shift 2 ;;
    -D|--deep-every) DEEP_EVERY="$2"; shift 2 ;;
    -e|--effort)     EFFORT="$2"; shift 2 ;;
    -t|--max-turns)  MAX_TURNS="$2"; shift 2 ;;
    -s|--sleep)      SLEEP_S="$2"; shift 2 ;;
    -M|--meta-every) META_EVERY="$2"; shift 2 ;;
    --meta-eps)      META_EPS="$2"; shift 2 ;;
    --meta-max-diff) META_MAX_DIFF="$2"; shift 2 ;;
    --meta-model)    META_MODEL="$2"; shift 2 ;;
    --dry-run)       DRY_RUN=1; shift ;;
    -h|--help)       sed -n '2,44p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (try --help)"; exit 2 ;;
  esac
done

# "ultracode" = xhigh effort + the documented keyword opt-in that grants the
# headless agent multi-agent orchestration (Workflow tool) for its iteration.
ULTRA=0
if [[ "$EFFORT" == "ultracode" ]]; then EFFORT="xhigh"; ULTRA=1; fi

sha12() {  # short content hash for policy attribution in the audit log
  python3 -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()[:12])" "$1"
}

build_cmd() {  # $1 = model
  local prompt
  # the policy surface is hot-swappable: re-read and re-hash it every iteration
  if [[ -f PROMPT_CORE.md && -f POLICY.md ]]; then
    prompt="$(cat PROMPT_CORE.md)

$(cat POLICY.md)"
    POLICY_SHA="$(sha12 POLICY.md)"
  elif [[ -f PROMPT.md ]]; then     # legacy monolithic prompt
    prompt="$(cat PROMPT.md)"
    POLICY_SHA="$(sha12 PROMPT.md)"
  else
    echo "FATAL: no PROMPT_CORE.md + POLICY.md (or legacy PROMPT.md) in $PWD" >&2
    exit 1
  fi
  # harness-computed lineage scoreboard: island-rule data, not prose
  local scoreboard
  scoreboard="$(python3 loop.py lineage-scoreboard 2>/dev/null || true)"
  [[ -n "$scoreboard" ]] && prompt="$prompt

$scoreboard"
  if ((ULTRA)); then
    prompt="ultracode

$prompt

(ultracode note: you may fan out subagent workflows for parallel hypothesis
variants or adversarial pre-review, but the protocol is unchanged: ONE
candidate submitted to the harness, ONE notebook entry, then stop.
PROBE ACCOUNTING: offline probes select on TRAIN/dev data only — never
touch the holdout outside the harness. Every probed variant must appear in
your ACTIVE LOG block as one 'probe:' line (variant, split, self-measured
number marked [self-reported], verdict); probe-killed ideas get a GRAVEYARD
row. Probe numbers are telemetry, never evidence — only harness verdicts
are truth.)"
  fi
  CMD=(claude -p "$prompt"
       --model "$1"
       --max-turns "$MAX_TURNS"
       --dangerously-skip-permissions
       --output-format stream-json --verbose)
  [[ -n "$EFFORT" ]] && CMD+=(--effort "$EFFORT")
}

gitc() {  # commit with a stable identity so the meta-ratchet works anywhere
  git -c user.name=meta-loop -c user.email=meta-loop@local "$@"
}

# ---------------------------------------------------------------- meta-pass
# P1 of the meta-loop (docs/meta-loop-design.md): every META_EVERY successful
# iterations, (1) verdict the trial policy that just ran blind — keep it
# (commit) or revert it (git checkout) via loop.py meta-ratchet — then
# (2) spawn a bounded meta-agent that may propose ONE small edit to
# POLICY.md for the next blind window. Trial edits stay UNCOMMITTED during
# their window; git is the revert mechanism. The runner enforces the write
# boundary by diff inspection — trust is not delegated to the meta-agent.

run_meta_pass() {
  echo
  echo "=== meta-pass after iteration $iter $(date '+%H:%M:%S') ==="
  if [[ ! -f META_PROMPT.md ]]; then
    echo "meta: no META_PROMPT.md here — skipping (re-scaffold or copy the template)"
    return
  fi
  if [[ ! -d .git ]]; then
    git init -q && git add -A >/dev/null 2>&1 \
      && gitc commit -qm "meta-loop: initial experiment state" \
      || { echo "meta: git unavailable — skipping (no revert mechanism)"; return; }
  fi

  # 1. verdict on the window that just finished
  local verdict_json verdict
  verdict_json=$(python3 loop.py meta-ratchet check --window "$META_EVERY" --eps "$META_EPS") \
    || { echo "meta: ratchet check failed — skipping"; return; }
  verdict=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['verdict'])" "$verdict_json")
  echo "meta: ratchet $verdict_json"
  if [[ "$verdict" == "revert" ]]; then
    git checkout -q -- POLICY.md
    echo "meta: trial policy reverted — incumbent restored ($(sha12 POLICY.md))"
  elif [[ "$verdict" == "keep" ]]; then
    git add POLICY.md && gitc commit -qm "meta: keep policy $(sha12 POLICY.md)"
    echo "meta: trial policy kept as the new incumbent"
  fi

  # 2. checkpoint iteration debris so the next trial's revert stays isolated
  git add -A >/dev/null 2>&1
  gitc commit -qm "state after iteration $iter" >/dev/null 2>&1 || true

  # 3. spawn the bounded meta-agent
  local stats mprompt mmodel ts_meta t0m rcm
  stats=$(python3 loop.py meta-stats --window "$META_EVERY" --json)
  mprompt="$(cat META_PROMPT.md)

## Current window meta-stats (harness-computed)
\`\`\`json
$stats
\`\`\`"
  mmodel="${META_MODEL:-$DEEP_MODEL}"
  ts_meta=$(date '+%Y-%m-%dT%H:%M:%S'); t0m=$(date +%s)
  rm -f .last_result.json
  claude -p "$mprompt" --model "$mmodel" --max-turns 15 \
    --dangerously-skip-permissions --output-format stream-json --verbose \
    2>>runner_last_stderr.log | python3 -u _stream_view.py .last_result.json
  rcm="${PIPESTATUS[0]}"

  # 4. audit the meta-pass spend (phase=meta rows are excluded from window
  #    fitness by meta-stats — both compared windows carry one meta-pass)
  python3 -c "
import json
try:
    d = json.load(open('.last_result.json'))
except Exception:
    d = {}
u = d.get('usage') or {}
rec = {'iter': $iter, 'phase': 'meta', 'model': '$mmodel', 'ts_start': '$ts_meta',
       'wall_s': $(( $(date +%s) - t0m )), 'exit': $rcm,
       'api_ms': d.get('duration_api_ms'), 'turns': d.get('num_turns'),
       'in_tokens': u.get('input_tokens'), 'out_tokens': u.get('output_tokens'),
       'result_tail': (d.get('result') or '')[-200:]}
print(json.dumps(rec))
" >> "$AUDIT"

  # 5. enforce the write boundary: only POLICY.md (+ META_LOG.md) may change.
  #    Runner-owned scratch (audit, stream files, dashboard) is exempt.
  local changed illegal untracked
  changed=$(git diff --name-only)
  illegal=$(echo "$changed" | grep -vE \
    '^(POLICY\.md|META_LOG\.md|\.last_result\.json|runner_last_stderr\.log|loop_audit\.jsonl|dashboard\.html|\.dash_opened)$|^checkpoints/' \
    | grep -v '^$' || true)
  if [[ -n "$illegal" ]]; then
    echo "meta: BOUNDARY VIOLATION — meta-pass touched: $illegal — restoring; trial NOT armed"
    # shellcheck disable=SC2086
    git checkout -q -- $illegal POLICY.md 2>/dev/null || true
    return
  fi
  untracked=$(git status --porcelain | grep '^??' || true)
  [[ -n "$untracked" ]] && echo "meta: note — meta-pass created untracked files:"$'\n'"$untracked"

  if echo "$changed" | grep -q '^POLICY\.md$'; then
    local diff_lines
    diff_lines=$(git diff --numstat POLICY.md | awk '{print $1 + $2}')
    if (( diff_lines > META_MAX_DIFF )); then
      echo "meta: proposed diff ($diff_lines lines) exceeds cap $META_MAX_DIFF — rejected"
      git checkout -q -- POLICY.md
    else
      python3 loop.py meta-ratchet arm --policy-sha "$(sha12 POLICY.md)" >/dev/null
      echo "meta: trial policy $(sha12 POLICY.md) armed — runs blind for the next $META_EVERY iterations"
    fi
  else
    echo "meta: no policy edit proposed (null pass)"
  fi
}

if ((DRY_RUN)); then
  build_cmd "$MODEL"
  printf 'would run, per iteration:\n  %q' "${CMD[0]}"
  printf ' %q' "${CMD[@]:1:1}" "PROMPT_CORE.md+POLICY.md(contents)" "${CMD[@]:3}"
  echo
  exit 0
fi

# preflight: corpus frozen, harness healthy, champion intact
python3 loop.py champion || exit 1

echo "loop starting: max_iters=$MAX_ITERS plateau_n=$PLATEAU_N model=$MODEL" \
     "deep=$DEEP_MODEL/every-$DEEP_EVERY effort=${EFFORT:-default} max_turns=$MAX_TURNS" \
     "meta=${META_EVERY:-0}$( ((META_EVERY > 0)) && echo "/eps-$META_EPS" )"
trap 'echo; echo "interrupted — state is safe in results.jsonl/notebook"; exit 130' INT

# continue numbering from the audit log (successful iterations only) — not
# from candidate-file count, which drifts with never-evaluated files
ITER_START=$(python3 -c "
import json
try:
    print(max((json.loads(l)['iter'] for l in open('$AUDIT')
               if l.strip() and json.loads(l).get('exit') == 0), default=0))
except Exception:
    print(0)")

CONSEC_FAILS=0
MAX_CONSEC_FAILS=3
BACKOFF_S=300

i=1
while ((i <= MAX_ITERS)); do
  iter=$((ITER_START + i))
  model="$MODEL"
  if ((DEEP_EVERY > 0 && i % DEEP_EVERY == 0)); then model="$DEEP_MODEL"; fi

  echo
  echo "=== iteration $iter (model: $model) $(date '+%H:%M:%S') ==="
  t0=$(date +%s)
  ts_start=$(date '+%Y-%m-%dT%H:%M:%S')   # portable (no BSD-only date -r)

  build_cmd "$model"
  rm -f .last_result.json
  # stream the agent's thinking/replies/tool calls live to this terminal;
  # the final result event lands in .last_result.json for the audit record
  "${CMD[@]}" 2>runner_last_stderr.log | python3 -u _stream_view.py .last_result.json
  rc="${PIPESTATUS[0]}"
  # a turn that ended in an API error (rate limit etc.) exits 0 but flags is_error
  if [[ $rc -eq 0 ]] && python3 -c "
import json,sys
try: sys.exit(0 if json.load(open('.last_result.json')).get('is_error') else 1)
except Exception: sys.exit(1)"; then rc=1; fi
  t1=$(date +%s)
  wall=$((t1 - t0))

  # audit record: wall time, api duration, turns, tokens (subscription => $0)
  python3 -c "
import json
try:
    d = json.load(open('.last_result.json'))
except Exception:
    d = {}
u = d.get('usage') or {}
rec = {'iter': $iter, 'model': '$model', 'ts_start': '$ts_start',
       'wall_s': $wall, 'exit': $rc, 'policy_sha': '$POLICY_SHA',
       'api_ms': d.get('duration_api_ms'), 'turns': d.get('num_turns'),
       'in_tokens': u.get('input_tokens'), 'out_tokens': u.get('output_tokens'),
       'result_tail': (d.get('result') or '')[-200:]}
print(json.dumps(rec))
" >> "$AUDIT"

  echo "iteration $iter done in ${wall}s (exit $rc)"

  if ((rc != 0)); then
    # FAILED iteration: do NOT consume the budget or advance numbering.
    CONSEC_FAILS=$((CONSEC_FAILS + 1))
    if grep -qiE "session limit|rate limit|usage limit|overloaded" \
         .last_result.json runner_last_stderr.log 2>/dev/null; then
      echo "rate-limited — backing off ${BACKOFF_S}s (failures don't count against max-iters)"
      sleep "$BACKOFF_S"
      ((BACKOFF_S < 3600)) && BACKOFF_S=$((BACKOFF_S * 2))
    elif ((CONSEC_FAILS >= MAX_CONSEC_FAILS)); then
      echo "aborting: $CONSEC_FAILS consecutive non-rate-limit failures (see runner_last_stderr.log)"
      break
    else
      sleep 30
    fi
    continue
  fi
  CONSEC_FAILS=0
  BACKOFF_S=300

  # refresh dashboard with the new audit data
  python3 loop.py dashboard >/dev/null 2>&1 || true

  # the harness owns the stop decision
  if ! python3 loop.py plateau "$PLATEAU_N"; then
    echo "plateau reached after iteration $iter — stopping loop"
    break
  fi

  # meta-loop: verdict the finished window, maybe arm a new trial policy
  if ((META_EVERY > 0 && i % META_EVERY == 0)); then
    run_meta_pass
  fi

  i=$((i + 1))
  # gentle pacing for subscription rate limits
  sleep "$SLEEP_S"
done

echo
echo "loop finished. Audit: $AUDIT · Dashboard: dashboard.html"
python3 loop.py dashboard >/dev/null 2>&1 || true
