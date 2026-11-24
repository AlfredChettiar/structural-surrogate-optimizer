"""UserPromptSubmit hook: log every raw prompt and route substantive ones through `prompt-refiner`.

Claude Code hooks cannot launch a subagent directly, so this hook (1) appends the verbatim prompt
to prompts/prompt_history.jsonl and (2) injects a mandatory instruction telling the main agent to
delegate to the `prompt-refiner` subagent before acting.

Skipped (logged, not refined): prompts of <= SHORT_PROMPT_WORDS words (e.g. "yes, approve"),
prompts starting with "raw:" and slash commands. The hook never blocks a prompt: any internal
error is swallowed and the prompt proceeds unchanged.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path

SHORT_PROMPT_WORDS = 4


def _project_dir(payload: dict) -> Path:
    return Path(os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or Path.cwd())


def _next_entry_id(log_path: Path) -> str:
    n = 0
    if log_path.exists():
        with log_path.open(encoding="utf-8") as fh:
            n = sum(1 for line in fh if line.strip())
    return f"P-{n + 1:04d}"


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        return 0
    prompt = str(payload.get("prompt", ""))
    stripped = prompt.strip()
    try:
        root = _project_dir(payload)
        log_dir = root / "prompts"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "prompt_history.jsonl"
        entry_id = _next_entry_id(log_path)
        refine = not (
            len(stripped.split()) <= SHORT_PROMPT_WORDS
            or stripped.lower().startswith("raw:")
            or stripped.startswith("/")
        )
        record = {
            "id": entry_id,
            "timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "session_id": payload.get("session_id"),
            "sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12],
            "refine": refine,
            "prompt": prompt,
        }
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # never block the user's prompt
        return 0

    if not refine:
        return 0
    context = (
        f"PROMPT-REFINER PROTOCOL (entry {entry_id}): before doing anything else, invoke the "
        f"`prompt-refiner` subagent with ENTRY_ID={entry_id} and the user's prompt verbatim. "
        "Use the refined brief it returns as your working specification and surface any blocking "
        "open questions it lists. The user's original wording stays authoritative: if the refined "
        "brief and the original conflict, follow the original and ask."
    )
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                              "additionalContext": context}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
