#!/usr/bin/env python3
"""
_dashboard.py — display sibling of the trusted harness (loop.py).

Renders dashboard.html plus per-checkpoint PNGs from the harness-written
records (results.jsonl, leaderboard.json, loop_audit.jsonl). Copied into
each experiment by `loop.py init`, like _stream_view.py. Deliberately NOT
part of the referee: display code carries no trust responsibility, and
loop.py invokes it through a guard that can never fail an eval.
matplotlib is optional — without it the dashboard falls back to HTML
tables only.
"""

import datetime as dt
import html
import json
import os
import subprocess
import sys

import loop

def make_dashboard(cfg):
    results = loop.read_results()
    cands = [r for r in results if r.get("lineage") != "baseline"]
    lb = loop.read_leaderboard()
    audits = ([json.loads(line) for line in open(loop.AUDIT) if line.strip()]
              if os.path.exists(loop.AUDIT) else [])
    os.makedirs(loop.CHECKPOINT_DIR, exist_ok=True)
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
        if not os.path.exists(loop.DASH_MARKER):
            subprocess.Popen(["open", loop.DASHBOARD])
            open(loop.DASH_MARKER, "w").write(dt.datetime.now().isoformat())
        elif dash_open == "every" and png_rel:
            subprocess.Popen(["open", "-g", os.path.join(loop.ROOT, png_rel)])


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

    png = os.path.join(loop.CHECKPOINT_DIR, f"cp_{cp_n:04d}.png")
    fig.savefig(png, dpi=120, facecolor=BG)
    fig.savefig(os.path.join(loop.CHECKPOINT_DIR, "cp_latest.png"), dpi=120, facecolor=BG)
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
    with open(loop.DASHBOARD, "w") as f:
        f.write(html_doc)

