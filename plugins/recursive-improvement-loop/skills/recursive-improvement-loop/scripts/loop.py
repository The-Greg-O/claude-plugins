#!/usr/bin/env python3
"""
loop.py — domain-agnostic harness core for recursive-improvement experiments.

The harness is the TRUSTED REFEREE of an experiment: the sole writer of
results.jsonl and leaderboard.json, owner of the champion ratchet, the
plateau stop, notebook compaction, and the per-checkpoint dashboard.
Everything domain-specific lives in ONE user-supplied evaluator script.

Evaluator contract (the only thing an experiment must provide):
    <eval_cmd> <candidate_path>
prints exactly one JSON object on its final stdout line:
    {
      "gate_passed": true|false,     # hard pass/fail (correctness/validity)
      "gate_error":  "...",          # why the gate failed (when false)
      "metrics":     {"name": num},  # everything measurable, incl. primary
    }
The harness never trusts the iterating agent's numbers — only this output.

Commands:
    loop.py init [dir]            scaffold a new experiment (interactive-ish)
    loop.py eval --candidate F [--meta JSON] [--baseline]
    loop.py champion              re-verify stored champion (drift guard)
    loop.py plateau N             exit 1 if no promotion in last N evals
    loop.py status                one-screen experiment state
    loop.py meta-stats [--window K] [--json]
                                  meta-fitness readout over the last K
                                  runner iterations (loop-improvement data)
    loop.py meta-ratchet check --window K [--eps E]
                                  keep/revert verdict on the trial POLICY.md
    loop.py meta-ratchet arm [--policy-file F]
                                  mark a trial policy as running blind
    loop.py lineage-scoreboard    per-lineage stats block (runner injects
                                  this into each iteration prompt)
    loop.py audit-append --iter N --model M --ts-start T --wall S --exit RC
                         --policy-file F [--result-file F] [--phase meta]
                                  write one loop_audit.jsonl row (the
                                  harness is the sole writer of trusted
                                  records; the runner only measures)
    loop.py dashboard             regenerate dashboard only
    loop.py compact [--keep N]    roll aged notebook blocks into GRAVEYARD

Config: experiment.json in the experiment root (see templates/).
Stdlib only; matplotlib is optional (dashboard falls back to HTML tables).
"""

import argparse
import datetime as dt
import hashlib
import html
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

ROOT = os.getcwd()
CONFIG = os.path.join(ROOT, "experiment.json")
RESULTS = os.path.join(ROOT, "results.jsonl")
LEADERBOARD = os.path.join(ROOT, "leaderboard.json")
NOTEBOOK = os.path.join(ROOT, "LAB_NOTEBOOK.md")
AUDIT = os.path.join(ROOT, "loop_audit.jsonl")
META_STATE = os.path.join(ROOT, "meta_state.json")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
DASHBOARD = os.path.join(ROOT, "dashboard.html")
DASH_MARKER = os.path.join(ROOT, ".dash_opened")
TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def load_config():
    if not os.path.exists(CONFIG):
        sys.exit("FATAL: no experiment.json here. Run `loop.py init` first "
                 "(or cd into the experiment directory).")
    cfg = json.load(open(CONFIG))
    for key in ("name", "eval_cmd", "primary_metric", "direction"):
        if key not in cfg:
            sys.exit(f"FATAL: experiment.json missing required key: {key}")
    if cfg["direction"] not in ("maximize", "minimize"):
        sys.exit("FATAL: direction must be 'maximize' or 'minimize'")
    cfg.setdefault("promote_eps", 1e-9)
    cfg.setdefault("promote_eps_rel", 0.0)      # fraction of champion primary
    cfg.setdefault("eval_timeout_s", 600)
    cfg.setdefault("reverify_champion", True)   # true | false | "gate-only"
    cfg.setdefault("reverify_every", 1)         # re-verify every Nth eval
    cfg.setdefault("reverify_rel_tol", 0.001)   # drift tolerance; >= 2x noise band
    cfg.setdefault("secondary_metrics", [])
    cfg.setdefault("max_candidate_bytes", 262144)   # anti embed-the-answer
    cfg.setdefault("primary_axis_range", None)      # [lo, hi] anchored dashboard scale
    return cfg


def better(cfg, a, b):
    """Is primary value `a` strictly better than `b` (with epsilon)?"""
    if b is None:
        return a is not None
    if a is None:
        return False
    eps = max(cfg["promote_eps"], cfg["promote_eps_rel"] * abs(b))
    return a > b + eps if cfg["direction"] == "maximize" else a < b - eps


# ---------------------------------------------------------------- records

def read_results():
    if not os.path.exists(RESULTS):
        return []
    return [json.loads(line) for line in open(RESULTS) if line.strip()]


def read_leaderboard():
    if os.path.exists(LEADERBOARD):
        return json.load(open(LEADERBOARD))
    return {"champion": None, "history": [], "baselines": {}}


def append_result(rec):
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")


def write_leaderboard(lb):
    # atomic: write to a temp file then replace, so an interrupt can never
    # leave the trusted leaderboard.json half-written.
    tmp = LEADERBOARD + ".tmp"
    with open(tmp, "w") as f:
        json.dump(lb, f, indent=2)
    os.replace(tmp, LEADERBOARD)


def _sha12_file(path):
    """Short content hash for policy attribution; None if unreadable."""
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]
    except OSError:
        return None


def build_audit_record(iter_n, model, ts_start, wall_s, exit_code,
                       policy_path, result_path, phase=None):
    """Assemble one loop_audit.jsonl row: the runner passes only what it
    uniquely measures (iteration, model, timestamps, wall clock, exit);
    the harness parses the claude CLI result event and hashes the active
    policy itself, so every trusted field is built by tested python."""
    try:
        d = json.load(open(result_path))
    except Exception:
        d = {}
    u = d.get("usage") or {}
    rec = {"iter": iter_n, "model": model, "ts_start": ts_start,
           "wall_s": wall_s, "exit": exit_code}
    if phase:
        rec["phase"] = phase
    rec.update({"policy_sha": _sha12_file(policy_path),
                "api_ms": d.get("duration_api_ms"),
                "turns": d.get("num_turns"),
                "in_tokens": u.get("input_tokens"),
                "out_tokens": u.get("output_tokens"),
                "result_tail": (d.get("result") or "")[-200:]})
    return rec


def audit_append(iter_n, model, ts_start, wall_s, exit_code,
                 policy_path, result_path, phase=None):
    rec = build_audit_record(iter_n, model, ts_start, wall_s, exit_code,
                             policy_path, result_path, phase=phase)
    # one write() call per row: an interrupt can't tear the trusted record
    with open(AUDIT, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------- evaluator

def run_evaluator(cfg, candidate_path):
    cmd = shlex.split(cfg["eval_cmd"]) + [candidate_path]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=cfg["eval_timeout_s"], cwd=ROOT)
    except subprocess.TimeoutExpired:
        return {"gate_passed": False,
                "gate_error": f"evaluator exceeded {cfg['eval_timeout_s']}s",
                "metrics": {}, "wall_s_eval": cfg["eval_timeout_s"]}
    wall = round(time.perf_counter() - t0, 2)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        return {"gate_passed": False,
                "gate_error": "evaluator crashed: " + " | ".join(tail),
                "metrics": {}, "wall_s_eval": wall}
    try:
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        assert isinstance(out.get("gate_passed"), bool)
        assert isinstance(out.get("metrics", {}), dict)
    except Exception as e:
        return {"gate_passed": False,
                "gate_error": f"evaluator output not valid contract JSON: {e}",
                "metrics": {}, "wall_s_eval": wall}
    out.setdefault("gate_error", "")
    out.setdefault("metrics", {})
    out["wall_s_eval"] = wall
    return out


def verify_champion(cfg, iter_n=None):
    lb = read_leaderboard()
    ch = lb.get("champion")
    if not ch:
        return None
    mode = cfg["reverify_champion"]
    if mode is False:
        return ch
    every = max(1, int(cfg["reverify_every"]))
    if iter_n is not None and every > 1 and iter_n % every != 0:
        return ch                      # skip this round (expensive evaluators)
    path = os.path.join(ROOT, ch["candidate"])
    if not os.path.exists(path):
        sys.exit(f"FATAL: champion file {ch['candidate']} missing — refusing to run")
    out = run_evaluator(cfg, path)
    if not out["gate_passed"]:
        sys.exit(f"FATAL: champion {ch['id']} no longer passes the gate "
                 f"({out['gate_error']}) — environment/data drift. Investigate "
                 f"before iterating further.")
    if mode == "gate-only":
        return ch                      # noisy primaries: gate yes, drift no
    new_p = out["metrics"].get(cfg["primary_metric"])
    tol = max(cfg["promote_eps"], abs(ch["primary"]) * cfg["reverify_rel_tol"])
    if new_p is None or abs(new_p - ch["primary"]) > tol:
        sys.exit(f"FATAL: champion primary drift {ch['primary']} -> {new_p} "
                 f"(tolerance {tol}) — the evaluator or data changed under the "
                 f"loop. If the evaluator is just noisy, raise reverify_rel_tol "
                 f"(>= 2x your noise band), aggregate best-of-K inside "
                 f"evaluate.py, or set reverify_champion to \"gate-only\".")
    return ch


# ---------------------------------------------------------------- eval

def eval_candidate(cfg, path, meta, baseline=False):
    if not os.path.isfile(path):
        sys.exit(f"FATAL: candidate {path} does not exist or is not a file "
                 f"(nothing recorded — create it and re-run)")
    relpath = os.path.relpath(os.path.abspath(path), ROOT)
    size = os.path.getsize(path)
    results = read_results()
    iter_n = 1 + sum(1 for r in results if r.get("lineage") != "baseline")

    rec = {
        "id": meta.get("id") or os.path.splitext(os.path.basename(path))[0],
        "iter": 0 if baseline else iter_n,
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "candidate": relpath,
        "lineage": "baseline" if baseline else meta.get("lineage", "unspecified"),
        "parent_id": meta.get("parent"),
        "hypothesis": (meta.get("hypothesis") or "")[:300],
        "candidate_bytes": size,
    }

    # embed-the-answer guard runs FIRST — reject an oversized candidate before
    # paying for the (potentially expensive) champion re-verify or evaluator run.
    if size > cfg["max_candidate_bytes"]:
        rec.update(gate_passed=False, promoted=False, primary=None, metrics={},
                   gate_error=f"candidate is {size} bytes > max_candidate_bytes "
                              f"{cfg['max_candidate_bytes']} (embed-the-answer guard)")
        append_result(rec)
        print_verdict(cfg, rec, None)
        make_dashboard(cfg)
        return rec

    champ = None if baseline else verify_champion(cfg, iter_n=iter_n)
    out = run_evaluator(cfg, path)
    rec.update(out)
    rec["primary"] = out["metrics"].get(cfg["primary_metric"])
    if rec["gate_passed"] and rec["primary"] is None:
        rec["gate_passed"] = False
        rec["gate_error"] = (f"metrics missing primary_metric "
                             f"'{cfg['primary_metric']}'")

    promoted = (not baseline and rec["gate_passed"]
                and better(cfg, rec["primary"], champ["primary"] if champ else None))
    rec["promoted"] = bool(promoted)

    if champ and rec["gate_passed"]:
        rec["delta_vs_champion"] = {
            k: round(v - champ.get("metrics", {}).get(k, 0), 6)
            for k, v in rec["metrics"].items()
            if isinstance(v, (int, float))
            and isinstance(champ.get("metrics", {}).get(k), (int, float))
        }

    append_result(rec)

    lb = read_leaderboard()
    if baseline:
        lb["baselines"][rec["id"]] = rec["primary"]
        write_leaderboard(lb)
    elif promoted:
        lb["champion"] = {k: rec[k] for k in
                          ("id", "iter", "ts", "candidate", "lineage",
                           "primary", "metrics") if k in rec}
        lb["history"].append(lb["champion"])
        lb["updated"] = rec["ts"]
        write_leaderboard(lb)

    print_verdict(cfg, rec, champ)
    make_dashboard(cfg)
    return rec


def print_verdict(cfg, rec, champ):
    print("\n========== HARNESS VERDICT (copy this into the notebook) ==========")
    print(f"id={rec['id']}  iter={rec['iter']}  gate_passed={rec['gate_passed']}")
    if rec["gate_passed"]:
        print(f"primary {cfg['primary_metric']}={rec['primary']}  "
              f"({cfg['direction']})")
        for k, v in sorted(rec["metrics"].items()):
            if k != cfg["primary_metric"]:
                print(f"  {k}={v}")
        if champ and rec.get("delta_vs_champion"):
            d = rec["delta_vs_champion"].get(cfg["primary_metric"])
            if d is not None:
                print(f"delta vs champion {champ['id']}: {d:+}")
        print("PROMOTED TO CHAMPION" if rec.get("promoted")
              else "not promoted (champion stands)")
    else:
        print(f"gate_error: {rec.get('gate_error')}")
    print("====================================================================\n")


# ---------------------------------------------------------------- plateau / status

def plateau(n):
    results = [r for r in read_results() if r.get("lineage") != "baseline"]
    since = 0
    for r in reversed(results):
        if r.get("promoted"):
            break
        since += 1
    ch = read_leaderboard().get("champion") or {}
    print(f"evals since last promotion: {since}  (threshold {n})  "
          f"champion primary: {ch.get('primary')}")
    if since >= n:
        print("PLATEAU REACHED — signalling the loop to stop")
        return 1
    return 0


def status(cfg):
    results = read_results()
    cands = [r for r in results if r.get("lineage") != "baseline"]
    lb = read_leaderboard()
    ch = lb.get("champion") or {}
    print(f"experiment: {cfg['name']}")
    print(f"objective:  {cfg['direction']} {cfg['primary_metric']}")
    print(f"evals:      {len(cands)} candidates, {len(results) - len(cands)} baselines")
    print(f"champion:   {ch.get('id', '—')} primary={ch.get('primary', '—')} "
          f"(iter {ch.get('iter', '—')})")
    print(f"promotions: {len(lb.get('history') or [])}")
    if lb.get("baselines"):
        print("baselines: " + ", ".join(f"{k}={v}" for k, v in lb["baselines"].items()))
    fails = sum(1 for r in cands if not r.get("gate_passed"))
    print(f"gate fails: {fails}/{len(cands)}")


# ---------------------------------------------------------------- meta-stats
#
# Meta-fitness readout for the meta-loop (docs/meta-loop-design.md): how well
# is the improvement PROCESS working, per token spent, over a recent window?
# Computed exclusively from harness/runner-written files (results.jsonl +
# loop_audit.jsonl) — never from agent self-reports. The window is defined by
# the last K SUCCESSFUL runner iterations; failed iterations inside that span
# still count toward token cost (they were paid for).

def _entropy_bits(counts):
    total = sum(counts)
    if not total:
        return None
    # + 0.0 normalizes the -0.0 float artifact of a single-lineage window
    return -sum((c / total) * math.log2(c / total) for c in counts if c) + 0.0


def _hyp_tokens(text):
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _is_fuzzy_repeat(hyp, prior_hyps, threshold=0.6):
    """Token-set Jaccard >= threshold against any earlier hypothesis."""
    toks = _hyp_tokens(hyp)
    if not toks:
        return False
    for prev in prior_hyps:
        ptoks = _hyp_tokens(prev)
        if ptoks and len(toks & ptoks) / len(toks | ptoks) >= threshold:
            return True
    return False


def compute_meta_stats(cfg, results, audits, window):
    """Pure function of the two record streams; returns the stats dict."""
    pm = cfg["primary_metric"]
    sign = 1.0 if cfg["direction"] == "maximize" else -1.0
    cands = [r for r in results if r.get("lineage") != "baseline"]
    # meta-pass rows (phase == "meta") are meta-loop overhead: they never
    # count as inner iterations or window fitness cost — both compared
    # windows carry one meta-pass each — but their spend is reported.
    ok_audits = sorted((a for a in audits
                        if a.get("exit") == 0 and a.get("phase") != "meta"),
                       key=lambda a: a.get("ts_start") or "")

    if ok_audits:
        t_lo = ok_audits[-window:][0].get("ts_start") or ""
        span = sorted((a for a in audits
                       if (a.get("ts_start") or "") >= t_lo),
                      key=lambda a: a.get("ts_start") or "")
        win_audits = [a for a in span if a.get("phase") != "meta"]
        meta_rows = [a for a in span if a.get("phase") == "meta"]
        evals = [r for r in cands if (r.get("ts") or "") >= t_lo]
        window_iters = len(ok_audits[-window:])
        tokens = {"in": sum(a.get("in_tokens") or 0 for a in win_audits),
                  "out": sum(a.get("out_tokens") or 0 for a in win_audits)}
        tokens["total"] = tokens["in"] + tokens["out"]
        meta_tokens = sum((a.get("in_tokens") or 0) + (a.get("out_tokens") or 0)
                          for a in meta_rows)
        wall_s = round(sum(a.get("wall_s") or 0 for a in win_audits), 1)
    else:                       # evals run outside the runner: window by eval
        t_lo = None
        win_audits = []
        evals = cands[-window:]
        window_iters = None
        tokens = None
        meta_tokens = None
        wall_s = None

    n = len(evals)
    n_promoted = sum(1 for r in evals if r.get("promoted"))
    gate_fails = sum(1 for r in evals if not r.get("gate_passed"))
    deltas = [r["delta_vs_champion"][pm] for r in evals
              if r.get("promoted") and isinstance(
                  r.get("delta_vs_champion", {}).get(pm), (int, float))]

    # champion improvement across the window, signed so positive = better.
    # Base = champion entering the window; a first-ever promotion measures
    # from the first champion (improvement over "nothing" is undefined).
    promoted = [r for r in evals if r.get("promoted")
                and isinstance(r.get("primary"), (int, float))]
    champ_before = None
    pre = cands[:len(cands) - n]
    for r in pre:
        if r.get("promoted") and isinstance(r.get("primary"), (int, float)):
            champ_before = r["primary"]
    if promoted:
        base = champ_before if champ_before is not None else promoted[0]["primary"]
        improvement = sign * (promoted[-1]["primary"] - base)
    else:
        improvement = 0.0
    imp_per_mtoken = (round(improvement / (tokens["total"] / 1e6), 6)
                      if tokens and tokens["total"] else None)

    lineage_counts = {}
    for r in evals:
        lin = r.get("lineage") or "unspecified"
        lineage_counts[lin] = lineage_counts.get(lin, 0) + 1

    # hypothesis novelty: window hypotheses fuzzy-matched against ALL prior
    # hypotheses (evals is a chronological suffix of cands)
    hyp_all = [r.get("hypothesis") or "" for r in cands]
    start = len(cands) - n
    repeats = considered = 0
    for k, r in enumerate(evals):
        if not _hyp_tokens(r.get("hypothesis") or ""):
            continue
        considered += 1
        if _is_fuzzy_repeat(r["hypothesis"], hyp_all[:start + k]):
            repeats += 1

    gaps = {}
    for r in evals:
        if not r.get("gate_passed"):
            continue
        m = r.get("metrics") or {}
        for key, v in m.items():
            if key.endswith("_train") and isinstance(v, (int, float)):
                h = m.get(key[:-len("_train")] + "_holdout")
                if isinstance(h, (int, float)):
                    gaps.setdefault(key[:-len("_train")], []).append(v - h)

    policies = []
    if win_audits:
        groups = {}
        for a in win_audits:
            sha = a.get("policy_sha") or "unattributed"
            g = groups.setdefault(sha, {"policy_sha": sha, "iters": 0,
                                        "evals": 0, "promotions": 0,
                                        "gate_fails": 0, "tokens_total": 0})
            g["iters"] += 1 if a.get("exit") == 0 else 0
            g["tokens_total"] += (a.get("in_tokens") or 0) + (a.get("out_tokens") or 0)
        for r in evals:
            rts = r.get("ts") or ""
            sha = win_audits[0].get("policy_sha") or "unattributed"
            for a in win_audits:
                if (a.get("ts_start") or "") <= rts:
                    sha = a.get("policy_sha") or "unattributed"
                else:
                    break
            g = groups[sha]
            g["evals"] += 1
            g["promotions"] += 1 if r.get("promoted") else 0
            g["gate_fails"] += 0 if r.get("gate_passed") else 1
        policies = list(groups.values())

    return {
        "window": window,
        "window_iters": window_iters,
        "since_ts": t_lo or (evals[0].get("ts") if evals else None),
        "evals": n,
        "promotions": n_promoted,
        "promotions_per_eval": round(n_promoted / n, 4) if n else None,
        "mean_promoted_delta": (round(sum(deltas) / len(deltas), 6)
                                if deltas else None),
        "gate_fail_rate": round(gate_fails / n, 4) if n else None,
        "champion_improvement": round(improvement, 6),
        "champion_improvement_per_mtoken": imp_per_mtoken,
        "tokens": tokens,
        "meta_tokens": meta_tokens,
        "wall_s_total": wall_s,
        "lineage_entropy_bits": (round(_entropy_bits(lineage_counts.values()), 4)
                                 if lineage_counts else None),
        "lineage_counts": lineage_counts,
        "hypothesis_repeat_rate": (round(repeats / considered, 4)
                                   if considered else None),
        "train_holdout_gap": {k: round(sum(v) / len(v), 6)
                              for k, v in gaps.items()},
        "policies": policies,
    }


def meta_stats(cfg, window, as_json=False):
    results = read_results()
    audits = ([json.loads(line) for line in open(AUDIT) if line.strip()]
              if os.path.exists(AUDIT) else [])
    s = compute_meta_stats(cfg, results, audits, window)
    if as_json:
        print(json.dumps(s, indent=2))
        return
    span = (f"last {s['window_iters']} runner iterations"
            if s["window_iters"] is not None
            else f"last {s['evals']} evals (no audit data)")
    print(f"meta-stats: {span}, since {s['since_ts'] or '—'}")
    print(f"evals:      {s['evals']}  promotions: {s['promotions']} "
          f"(rate {s['promotions_per_eval']})  "
          f"gate-fail rate: {s['gate_fail_rate']}")
    print(f"champion:   {s['champion_improvement']:+} {cfg['primary_metric']} "
          f"over window  ({s['champion_improvement_per_mtoken']} per Mtoken)")
    print(f"mean promoted delta: {s['mean_promoted_delta']}")
    if s["tokens"]:
        print(f"cost:       {s['tokens']['total']:,} tokens "
              f"(in {s['tokens']['in']:,} / out {s['tokens']['out']:,}), "
              f"{s['wall_s_total']}s wall")
    print(f"diversity:  lineage entropy {s['lineage_entropy_bits']} bits "
          f"{s['lineage_counts']}")
    print(f"novelty:    hypothesis repeat rate {s['hypothesis_repeat_rate']}")
    if s["train_holdout_gap"]:
        gap = ", ".join(f"{k}={v:+}" for k, v in s["train_holdout_gap"].items())
        print(f"overfit:    train-holdout gap {gap}")
    for p in s["policies"]:
        print(f"policy {p['policy_sha']}: {p['iters']} iters, "
              f"{p['evals']} evals, {p['promotions']} promotions, "
              f"{p['gate_fails']} gate fails, {p['tokens_total']:,} tokens")
    print("(interpretation rubric: SKILL.md failure-mode table)")


# ---------------------------------------------------------------- lineage scoreboard
#
# FunSearch-bandit data for the island rule: the runner injects this into
# every iteration prompt so "advance a dormant lineage" is driven by
# harness-written numbers instead of the agent's reading of prose.

def compute_lineage_scoreboard(cfg, results):
    cands = [r for r in results if r.get("lineage") != "baseline"]
    rows = {}
    for r in cands:
        lin = r.get("lineage") or "unspecified"
        d = rows.setdefault(lin, {"lineage": lin, "attempts": 0,
                                  "gate_fails": 0, "promotions": 0,
                                  "best": None, "last_iter": None})
        d["attempts"] += 1
        d["gate_fails"] += 0 if r.get("gate_passed") else 1
        d["promotions"] += 1 if r.get("promoted") else 0
        p = r.get("primary")
        if isinstance(p, (int, float)):
            if d["best"] is None or (p > d["best"] if cfg["direction"] == "maximize"
                                     else p < d["best"]):
                d["best"] = p
        if isinstance(r.get("iter"), int):
            d["last_iter"] = max(d["last_iter"] or 0, r["iter"])
    none_key = float("-inf") if cfg["direction"] == "maximize" else float("inf")
    return sorted(rows.values(),
                  key=lambda d: none_key if d["best"] is None else d["best"],
                  reverse=(cfg["direction"] == "maximize"))


def island_alert(results):
    """The lineage that produced the last 3 promotions — if a single one."""
    promos = [r.get("lineage") or "unspecified" for r in results
              if r.get("lineage") != "baseline" and r.get("promoted")]
    if len(promos) >= 3 and len(set(promos[-3:])) == 1:
        return promos[-3]
    return None


def lineage_scoreboard(cfg):
    results = read_results()
    rows = compute_lineage_scoreboard(cfg, results)
    if not rows:
        return                       # nothing yet — the runner injects nothing
    print("## Lineage scoreboard (harness-computed)")
    for d in rows:
        print(f"- {d['lineage']}: {d['attempts']} attempts, "
              f"{d['promotions']} promotions, "
              f"best {cfg['primary_metric']}={d['best']}, "
              f"{d['gate_fails']} gate-fails, last iter {d['last_iter']}")
    hot = island_alert(results)
    if hot:
        print(f"- ISLAND RULE TRIGGERED: last 3 promotions all from "
              f"'{hot}' — advance a DIFFERENT or dormant lineage now.")


# ---------------------------------------------------------------- meta-ratchet
#
# P1 of the meta-loop: a trial POLICY.md runs BLIND for one window of inner
# iterations; it is kept only if its window's meta-fitness beats the
# incumbent's window by >= eps, else the runner reverts it (git). This file
# owns only the decision + state; the runner owns the git mechanics.

def decide_meta_ratchet(state, fitness, eps):
    """State machine for the meta-ratchet. Returns (verdict, new_state).

    verdicts: baseline (no trial ran; incumbent fitness refreshed),
              keep     (trial beat incumbent by >= eps; trial is the new
                        incumbent),
              revert   (trial failed to beat; runner must restore POLICY.md).
    """
    if not state:
        return "baseline", {"incumbent_fitness": fitness, "pending": False}
    if state.get("pending"):
        if fitness >= state["incumbent_fitness"] + eps:
            return "keep", {"incumbent_fitness": fitness, "pending": False}
        return "revert", {"incumbent_fitness": state["incumbent_fitness"],
                          "pending": False}
    return "baseline", {"incumbent_fitness": fitness, "pending": False}


def _window_fitness(stats):
    """Scalar meta-fitness of a window: champion improvement per Mtoken
    (cost-normalized by design; see docs/meta-loop-design.md)."""
    f = stats.get("champion_improvement_per_mtoken")
    return f if isinstance(f, (int, float)) else 0.0


def meta_ratchet(cfg, op, window, eps, policy_file=None):
    state = json.load(open(META_STATE)) if os.path.exists(META_STATE) else None
    if op == "arm":
        if state is None:
            sys.exit("FATAL: meta-ratchet arm before any check — run "
                     "`loop.py meta-ratchet check` first")
        sha = _sha12_file(policy_file) if policy_file else None
        state["pending"] = True
        state["trial_policy_sha"] = sha
        state["armed_ts"] = dt.datetime.now().isoformat(timespec="seconds")
        json.dump(state, open(META_STATE, "w"), indent=2)
        print(json.dumps({"verdict": "armed", "trial_policy_sha": sha}))
        return
    results = read_results()
    audits = ([json.loads(line) for line in open(AUDIT) if line.strip()]
              if os.path.exists(AUDIT) else [])
    stats = compute_meta_stats(cfg, results, audits, window)
    fitness = _window_fitness(stats)
    verdict, new_state = decide_meta_ratchet(state, fitness, eps)
    new_state["checked_ts"] = dt.datetime.now().isoformat(timespec="seconds")
    new_state["window"] = window
    new_state["eps"] = eps
    json.dump(new_state, open(META_STATE, "w"), indent=2)
    print(json.dumps({"verdict": verdict, "fitness": fitness,
                      "incumbent_fitness": new_state["incumbent_fitness"],
                      "window_evals": stats["evals"],
                      "window_promotions": stats["promotions"]}))


# ---------------------------------------------------------------- dashboard

def make_dashboard(cfg):
    results = read_results()
    cands = [r for r in results if r.get("lineage") != "baseline"]
    lb = read_leaderboard()
    audits = ([json.loads(line) for line in open(AUDIT) if line.strip()]
              if os.path.exists(AUDIT) else [])
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    cp_n = len(cands)
    png_rel = None
    try:
        png_rel = _plot(cfg, cands, lb, audits, cp_n)
    except ImportError:
        pass  # matplotlib absent — HTML-only dashboard
    except Exception as e:
        # the result is already recorded; a plotting glitch must never fail the
        # eval that produced it. Fall back to the HTML-only dashboard.
        print(f"warning: dashboard plot failed ({e}); rendering tables only",
              file=sys.stderr)
    _write_html(cfg, cands, lb, audits, cp_n, png_rel)
    dash_open = os.environ.get("DASH_OPEN", "")
    if sys.platform == "darwin" and dash_open != "never":
        if not os.path.exists(DASH_MARKER):
            subprocess.Popen(["open", DASHBOARD])
            open(DASH_MARKER, "w").write(dt.datetime.now().isoformat())
        elif dash_open == "every" and png_rel:
            subprocess.Popen(["open", "-g", os.path.join(ROOT, png_rel)])


def _plot(cfg, cands, lb, audits, cp_n):
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    # light theme
    BG, TEXT, MUTED = "#ffffff", "#202124", "#5f6368"
    C_ATTEMPT, C_CHAMP, C_FAIL = "#1f77b4", "#2ca02c", "#d62728"
    C_WALL, C_TIN, C_TOUT, C_SEC = "#17a2b8", "#1f77b4", "#ff7f0e", "#7b3fb0"

    pm, direction = cfg["primary_metric"], cfg["direction"]
    secondary = [m for m in cfg.get("secondary_metrics", []) if m != pm]
    nsec = len(secondary)
    sec_cols = min(3, nsec) or 1
    sec_rows = math.ceil(nsec / sec_cols) if nsec else 0

    sec_row_h = 2.2
    # top -> bottom: primary (compact, centered) · secondary small-multiples ·
    # agent economics (least important -> bottom).
    heights = [3.0] + ([sec_row_h * sec_rows] if sec_rows else []) + [sec_row_h]
    fig = plt.figure(figsize=(13, sum(heights) + 0.6), layout="constrained")
    fig.patch.set_facecolor(BG)
    subfigs = fig.subfigures(len(heights), 1, height_ratios=heights)
    for sf in subfigs:
        sf.patch.set_facecolor(BG)
    sf_primary = subfigs[0]
    sf_secondary = subfigs[1] if sec_rows else None
    sf_econ = subfigs[-1]
    fig.suptitle(f"{cfg['name']} — checkpoint {cp_n} — "
                 f"{dt.datetime.now():%Y-%m-%d %H:%M}", fontsize=13, color=TEXT)

    ok = [r for r in cands if r.get("gate_passed")]
    bad = [r for r in cands if not r.get("gate_passed")]
    xlim = (0.5, max(cp_n, 1) + 0.5)   # avoid a degenerate axis on the first eval

    def style_x(a):
        a.set_xlim(*xlim)
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        a.grid(alpha=0.3)
        a.set_facecolor(BG)

    # --- primary metric: compact width, centered in its row; direct champion
    #     line through promoted points; legend ABOVE the axes (off the data) ---
    gp = sf_primary.add_gridspec(1, 3, width_ratios=[1, 3, 1])
    axp = sf_primary.add_subplot(gp[0, 1])
    axp.set_facecolor(BG)
    axp.scatter([r["iter"] for r in ok], [r["primary"] for r in ok],
                s=28, c=C_ATTEMPT, label="attempt", zorder=3)
    champs = [(r["iter"], r["primary"]) for r in cands
              if r.get("promoted") and r.get("primary") is not None]
    if champs:
        axp.plot([c[0] for c in champs], [c[1] for c in champs], "-o",
                 c=C_CHAMP, lw=2, ms=4, label="champion", zorder=2)
    if bad:
        # gate-fails as a bottom-margin rug (axis-fraction y) so position never
        # reads as a real score.
        axp.scatter([r["iter"] for r in bad], [0.025] * len(bad), marker="x",
                    c=C_FAIL, s=34, transform=axp.get_xaxis_transform(),
                    clip_on=False, label=f"gate fail ({len(bad)})", zorder=4)
    for name, val in (lb.get("baselines") or {}).items():
        if val is not None:
            axp.axhline(val, ls="--", lw=1, c=MUTED, alpha=0.8)
            axp.text(0.995, val, f" {name}", fontsize=7, va="bottom", ha="right",
                     c=MUTED, transform=axp.get_yaxis_transform())
    if cfg.get("primary_axis_range"):
        axp.set_ylim(*cfg["primary_axis_range"])
    axp.set_title(f"primary: {pm}  ({direction})", loc="left", fontsize=11, pad=12, color=TEXT)
    axp.set_xlabel("iteration")
    axp.set_ylabel(pm, fontsize=9)
    style_x(axp)
    axp.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=3, fontsize=8, frameon=False)

    # --- secondary metrics: one auto-scaled small-multiple per metric (ALL of
    #     them), titled on the y-axis by the metric, with x-tick values shown ---
    if sec_rows:
        axes = sf_secondary.subplots(sec_rows, sec_cols, squeeze=False)
        flat = [a for row in axes for a in row]
        for idx, m in enumerate(secondary):
            a = flat[idx]
            a.set_facecolor(BG)
            pts = [(r["iter"], r["metrics"].get(m)) for r in ok
                   if isinstance(r.get("metrics", {}).get(m), (int, float))]
            if pts:
                a.plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=3, lw=1.2, c=C_SEC)
            else:
                a.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=7, transform=a.transAxes)
            a.set_xlabel("iteration", fontsize=7)
            a.set_ylabel(m, fontsize=7.5)
            a.tick_params(labelsize=7)
            style_x(a)
        for a in flat[nsec:]:
            a.axis("off")

    # --- agent economics: wall time (left) + stacked in/out tokens (right twin)
    #     per RUNNER iteration. Tokens exist only per runner iteration (audit),
    #     not per candidate, so this is honestly indexed by runner iteration.
    #     Rows summed per iter so a retried/rate-limited iteration's full cost is
    #     visible (also dedups the duplicate-iter retry). Legend above the axes. ---
    axe = sf_econ.subplots()
    axe.set_facecolor(BG)
    agg = {}
    for a in audits:
        it = a.get("iter")
        if it is None:
            continue
        d = agg.setdefault(it, {"wall": 0.0, "tin": 0, "tout": 0})
        d["wall"] += a.get("wall_s") or 0
        d["tin"] += a.get("in_tokens") or 0
        d["tout"] += a.get("out_tokens") or 0
    its = sorted(agg)
    if its:
        w = 0.38
        axe.bar([i - 0.20 for i in its], [agg[i]["wall"] for i in its], width=w,
                color=C_WALL, label="agent wall (s)")
        axt = axe.twinx()
        tin = [agg[i]["tin"] for i in its]
        tout = [agg[i]["tout"] for i in its]
        axt.bar([i + 0.20 for i in its], tin, width=w, color=C_TIN, label="in tokens")
        axt.bar([i + 0.20 for i in its], tout, width=w, bottom=tin, color=C_TOUT, label="out tokens")
        axt.set_ylabel("tokens", fontsize=8, color=MUTED)
        axt.tick_params(labelsize=7)
        axt.set_facecolor("none")
        axe.set_ylabel("agent wall (s)", fontsize=8, color=MUTED)
        h1, l1 = axe.get_legend_handles_labels()
        h2, l2 = axt.get_legend_handles_labels()
        axe.legend(h1 + h2, l1 + l2, loc="lower right", bbox_to_anchor=(1.0, 1.01),
                   fontsize=7, ncol=3, frameon=False)
        axe.set_xlim(min(its) - 0.6, max(its) + 0.6)
        axe.xaxis.set_major_locator(MaxNLocator(integer=True))
        axe.grid(alpha=0.3)
    else:
        axe.text(0.5, 0.5, "no audit data", ha="center", va="center", transform=axe.transAxes)
    axe.set_title("agent time + tokens per runner iteration", loc="left", fontsize=10, pad=6, color=TEXT)
    axe.set_xlabel("runner iteration")

    png = os.path.join(CHECKPOINT_DIR, f"cp_{cp_n:04d}.png")
    fig.savefig(png, dpi=120, facecolor=BG)
    fig.savefig(os.path.join(CHECKPOINT_DIR, "cp_latest.png"), dpi=120, facecolor=BG)
    plt.close(fig)
    return "checkpoints/cp_latest.png"


def _write_html(cfg, cands, lb, audits, cp_n, png_rel):
    # escape everything agent- or candidate-derived (ids, lineages, hypothesis
    # and gate-error text, experiment name) before it lands in the page.
    e = html.escape
    name, pm = e(cfg["name"]), e(cfg["primary_metric"])
    ch = lb.get("champion") or {}
    rows = ""
    for r in cands[-12:][::-1]:
        cls = "ok" if r.get("gate_passed") else "bad"
        star = " ⭐" if r.get("promoted") else ""
        txt = (r.get("hypothesis") or r.get("gate_error") or "")
        disp = (txt[:90] + "…") if len(txt) > 90 else txt
        rows += (f"<tr class='{cls}'><td>{r['iter']}</td>"
                 f"<td>{e(str(r['id']))}{star}</td>"
                 f"<td>{e(str(r.get('lineage', '')))}</td>"
                 f"<td>{'pass' if r.get('gate_passed') else 'FAIL'}</td>"
                 f"<td>{e(str(r.get('primary', '—')))}</td>"
                 f"<td title=\"{e(txt)}\">{e(disp)}</td></tr>")
    base_rows = "".join(f"<tr><td>{e(str(k))}</td><td>{e(str(v))}</td></tr>"
                        for k, v in (lb.get("baselines") or {}).items())
    img = (f'<div><img src="{png_rel}?v={cp_n}"></div>' if png_rel else
           "<p><i>matplotlib not installed — charts disabled, tables only.</i></p>")
    total_wall = sum(a.get("wall_s", 0) for a in audits)
    html_doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="30"><title>{name}</title>
<style>
 body{{font-family:-apple-system,Helvetica,sans-serif;margin:18px;background:#ffffff;color:#202124}}
 h1{{font-size:20px}} h2{{font-size:14px;color:#5f6368}}
 table{{border-collapse:collapse;font-size:12px;margin-bottom:14px}}
 td,th{{border:1px solid #ddd;padding:3px 8px}} td{{max-width:520px}}
 tr.ok td{{color:#1b5e20}} tr.bad td{{color:#b00020}}
 .champ{{background:#e8f5e9;border:1px solid #2e7d32;padding:10px;border-radius:8px;
        display:inline-block;margin-bottom:12px}}
 img{{max-width:100%;border:1px solid #ddd;border-radius:6px}}
</style></head><body>
<h1>{name} <span style="color:#5f6368">— checkpoint {cp_n}, auto-refreshes every 30 s</span></h1>
<div class="champ"><b>Champion:</b> {e(str(ch.get('id', '—')))} ({e(str(ch.get('lineage', '—')))}) —
 <b>{pm} = {e(str(ch.get('primary', '—')))}</b> ({e(cfg['direction'])})</div>
{img}
<h2>Last {min(12, len(cands))} attempts (newest first) — ⭐ = promoted (hover for full text)</h2>
<table><tr><th>iter</th><th>id</th><th>lineage</th><th>gate</th>
<th>{pm}</th><th>hypothesis / error</th></tr>{rows}</table>
<h2>Baselines</h2><table><tr><th>id</th><th>{pm}</th></tr>{base_rows}</table>
<h2>Audit</h2><p>{len(audits)} runner iterations, {total_wall / 60:.1f} min total wall time.
Machine truth: results.jsonl · leaderboard.json · loop_audit.jsonl.</p>
</body></html>"""
    with open(DASHBOARD, "w") as f:
        f.write(html_doc)


# ---------------------------------------------------------------- compact

def compact_notebook(keep):
    if not os.path.exists(NOTEBOOK):
        print("no LAB_NOTEBOOK.md here")
        return
    text = open(NOTEBOOK).read()
    m = re.search(r"(## ACTIVE LOG\n)(.*)$", text, re.S)
    if not m:
        print("no ACTIVE LOG section found")
        return
    head, body = m.group(1), m.group(2)
    blocks = re.split(r"(?=### \[)", body)
    intro, blocks = blocks[0], [b for b in blocks[1:] if b.strip()]
    if len(blocks) <= keep:
        print(f"ACTIVE LOG has {len(blocks)} blocks (≤ {keep}); nothing to compact")
        return
    aged, kept = blocks[:-keep], blocks[-keep:]
    rows = []
    for b in aged:
        idm = re.match(r"### \[([^\]]+)\]", b)
        cid = idm.group(1) if idm else "?"
        lineage = re.search(r"lineage:\s*([\w-]+)", b)
        # matches both "primary score_holdout=0.7" (verdict format) and "primary=0.7"
        prim = re.search(r"primary(?:\s+\S+?)?\s*[=:]\s*([\d.eE+-]+)", b)
        verdict = "promoted" if "PROMOTED" in b else "not promoted"
        why = re.search(r"(?:WHY|Why):\s*([^\n]+)", b)
        rows.append(f"| {cid} | {lineage.group(1) if lineage else '?'} | "
                    f"{prim.group(1) if prim else '—'} | {verdict} | "
                    f"{(why.group(1) if why else '')[:80]} |")
    new_text = text.replace(m.group(0), head + intro + "".join(kept))
    gm = re.search(r"## GRAVEYARD.*?\n(\|[^\n]*\n)+", new_text, re.S)
    if gm:
        new_text = new_text[:gm.end()] + "\n".join(rows) + "\n" + new_text[gm.end():]
    else:
        hm = re.search(r"## GRAVEYARD[^\n]*\n", new_text)
        if hm:
            header = ("| id | lineage | primary | verdict | why it failed |\n"
                      "|---|---|---|---|---|\n")
            new_text = (new_text[:hm.end()] + header + "\n".join(rows) + "\n"
                        + new_text[hm.end():])
        else:
            print("ERROR: no GRAVEYARD section found — aborting WITHOUT writing "
                  "(aged blocks preserved). Add a '## GRAVEYARD' heading first.")
            return
    open(NOTEBOOK, "w").write(new_text)
    print(f"compacted {len(aged)} aged blocks into GRAVEYARD; kept {len(kept)}")


# ---------------------------------------------------------------- init

def init_experiment(target):
    target = os.path.abspath(target)
    os.makedirs(target, exist_ok=True)
    if os.path.exists(os.path.join(target, "experiment.json")):
        sys.exit(f"{target} already has an experiment.json — refusing to overwrite")
    name = os.path.basename(target)
    for fname in sorted(os.listdir(TEMPLATES)):
        src = os.path.join(TEMPLATES, fname)
        if not os.path.isfile(src) or ".template" not in fname:
            continue   # skip __pycache__/, .DS_Store, and other strays
        dst = os.path.join(target, fname.replace(".template", ""))
        body = open(src).read().replace("{{NAME}}", name)
        open(dst, "w").write(body)
    os.makedirs(os.path.join(target, "candidates"), exist_ok=True)
    os.makedirs(os.path.join(target, "checkpoints"), exist_ok=True)
    # the harness travels with the experiment so it survives plugin updates
    me = os.path.abspath(__file__)
    shutil.copy(me, os.path.join(target, "loop.py"))
    for extra in ("runner.sh", "_stream_view.py"):
        src = os.path.join(os.path.dirname(me), extra)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(target, extra))
            os.chmod(os.path.join(target, extra), 0o755)
    print(f"scaffolded experiment '{name}' in {target}")
    print("next steps:")
    print("  1. fill in experiment.json (eval_cmd, primary_metric, direction)")
    print("  2. write evaluate.py to the contract (gate + metrics)")
    print("  3. edit POLICY.md objectives + LAB_NOTEBOOK.md seed INSIGHTS")
    print("  4. add a seed candidate, then: python3 loop.py eval --candidate <f>")
    print("  5. record baselines: python3 loop.py eval --candidate <f> --baseline")
    print("  6. run: ./runner.sh -n 25 -p 10")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p_init = sub.add_parser("init")
    p_init.add_argument("dir", nargs="?", default=".")
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--candidate", required=True)
    p_eval.add_argument("--meta", default="{}")
    p_eval.add_argument("--baseline", action="store_true")
    sub.add_parser("champion")
    p_plat = sub.add_parser("plateau")
    p_plat.add_argument("n", type=int)
    sub.add_parser("status")
    p_meta = sub.add_parser("meta-stats")
    p_meta.add_argument("--window", type=int, default=15)
    p_meta.add_argument("--json", action="store_true")
    sub.add_parser("lineage-scoreboard")
    p_audit = sub.add_parser("audit-append")
    p_audit.add_argument("--iter", type=int, required=True, dest="iter_n")
    p_audit.add_argument("--model", required=True)
    p_audit.add_argument("--ts-start", required=True)
    p_audit.add_argument("--wall", type=int, required=True)
    p_audit.add_argument("--exit", type=int, required=True, dest="exit_code")
    p_audit.add_argument("--phase", default=None)
    p_audit.add_argument("--policy-file", required=True)
    p_audit.add_argument("--result-file", default=".last_result.json")
    p_ratchet = sub.add_parser("meta-ratchet")
    p_ratchet.add_argument("op", choices=("check", "arm"))
    p_ratchet.add_argument("--window", type=int, default=10)
    p_ratchet.add_argument("--eps", type=float, default=0.0)
    p_ratchet.add_argument("--policy-file", default=None)
    sub.add_parser("dashboard")
    p_comp = sub.add_parser("compact")
    p_comp.add_argument("--keep", type=int, default=15)
    args = ap.parse_args()

    if args.cmd == "init":
        init_experiment(args.dir)
        return
    if args.cmd is None:
        ap.print_help()
        return
    cfg = load_config()
    if args.cmd == "eval":
        try:
            meta = json.loads(args.meta)
        except json.JSONDecodeError:
            meta = {"hypothesis": args.meta}
        eval_candidate(cfg, args.candidate, meta, baseline=args.baseline)
    elif args.cmd == "champion":
        ch = verify_champion(cfg)
        print(f"champion OK: {ch}" if ch else "no champion yet")
    elif args.cmd == "plateau":
        sys.exit(plateau(args.n))
    elif args.cmd == "status":
        status(cfg)
    elif args.cmd == "meta-stats":
        meta_stats(cfg, args.window, as_json=args.json)
    elif args.cmd == "meta-ratchet":
        meta_ratchet(cfg, args.op, args.window, args.eps,
                     policy_file=args.policy_file)
    elif args.cmd == "lineage-scoreboard":
        lineage_scoreboard(cfg)
    elif args.cmd == "audit-append":
        audit_append(args.iter_n, args.model, args.ts_start, args.wall,
                     args.exit_code, args.policy_file, args.result_file,
                     phase=args.phase)
    elif args.cmd == "dashboard":
        make_dashboard(cfg)
        print(f"dashboard: {DASHBOARD}")
    elif args.cmd == "compact":
        compact_notebook(args.keep)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
