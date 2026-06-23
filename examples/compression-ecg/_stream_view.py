#!/usr/bin/env python3
"""
_stream_view.py — live terminal renderer for `claude -p --output-format stream-json`.

Reads JSONL events on stdin, pretty-prints the agent's thinking, replies, and
tool activity as they happen, and writes the final `result` event JSON to the
file given as argv[1] so runner.sh can build its audit record.

Runner-owned (trusted). The iterating agent never touches this file.
"""

import json
import sys
import textwrap

DIM = "\033[2m"
GRAY = "\033[90m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"
USE_COLOR = sys.stdout.isatty()


def c(code, s):
    return f"{code}{s}{RESET}" if USE_COLOR else s


def say(s):
    print(s, flush=True)


def wrap(s, prefix, width=100):
    out = []
    for para in s.splitlines():
        if not para.strip():
            continue
        out.extend(textwrap.wrap(para, width=width,
                                 initial_indent=prefix, subsequent_indent=prefix))
    return "\n".join(out)


def tool_summary(name, inp):
    if not isinstance(inp, dict):
        return name
    if name == "Bash":
        return f"Bash$ {str(inp.get('command', ''))[:160]}"
    if name in ("Read", "Write", "Edit"):
        return f"{name}: {inp.get('file_path', '?')}"
    if name in ("Glob", "Grep"):
        return f"{name}: {inp.get('pattern', '?')}"
    if name in ("Task", "Agent"):
        return f"{name}: {str(inp.get('description') or inp.get('prompt', ''))[:120]}"
    if name == "Workflow":
        return f"Workflow: {str(inp.get('name') or 'inline script')[:120]}"
    keys = ", ".join(list(inp)[:4])
    return f"{name}({keys})"


def result_text(content):
    """tool_result content can be a string or a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text")
    return ""


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        t = ev.get("type")

        if t == "system" and ev.get("subtype") == "init":
            say(c(DIM, f"· session {ev.get('session_id', '?')[:8]} started "
                       f"(cwd {ev.get('cwd', '?')})"))

        elif t == "assistant":
            for block in (ev.get("message") or {}).get("content") or []:
                if not isinstance(block, dict):
                    continue
                bt = block.get("type")
                if bt == "thinking":
                    say(c(GRAY, wrap(block.get("thinking", ""), "  🧠 ")))
                elif bt == "text":
                    say(wrap(block.get("text", ""), "  💬 "))
                elif bt == "tool_use":
                    say(c(CYAN, f"  ⚙  {tool_summary(block.get('name', '?'), block.get('input'))}"))

        elif t == "user":
            for block in (ev.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    txt = result_text(block.get("content")).strip()
                    if not txt:
                        continue
                    # show harness verdicts in full; trim everything else
                    if "HARNESS VERDICT" in txt:
                        say(c(GREEN, wrap(txt, "  ✅ ")))
                    else:
                        trimmed = txt[:400] + ("…" if len(txt) > 400 else "")
                        say(c(DIM, wrap(trimmed, "  ↳ ", width=110)))

        elif t == "result":
            if out_path:
                with open(out_path, "w") as f:
                    json.dump(ev, f)
            u = ev.get("usage") or {}
            say(c(YELLOW,
                  f"■ turn complete: {ev.get('num_turns', '?')} turns, "
                  f"{(ev.get('duration_api_ms') or 0) / 1000:.0f}s api, "
                  f"{u.get('output_tokens', '?')} out-tokens"
                  f"{', ERROR' if ev.get('is_error') else ''}"))


if __name__ == "__main__":
    main()
