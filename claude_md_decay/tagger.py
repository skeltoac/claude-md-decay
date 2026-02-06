"""Interactive CLI for human tagging of compliance violations."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from .config import DATA_DIR
from .retrospective import ActivationOpportunity


TAGS_FILE = DATA_DIR / "retrospective_tags.jsonl"


def _load_existing_tags() -> set[tuple[str, int, str]]:
    """Load already-tagged (session_id, turn_index, rule_id) triples."""
    tagged = set()
    if TAGS_FILE.exists():
        with open(TAGS_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                tagged.add((rec["session_id"], rec["turn_index"], rec["rule_id"]))
    return tagged


def _save_tag(opp: ActivationOpportunity, compliance: float) -> None:
    """Append a tag record to the tags file."""
    TAGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "session_id": opp.session_id,
        "turn_index": opp.turn_index,
        "cumulative_tokens": opp.cumulative_input_tokens,
        "rule_id": opp.rule_id,
        "compliance": compliance,
        "compactions_before": opp.compactions_before,
        "timestamp": opp.timestamp,
    }
    with open(TAGS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def _wrap(text: str, width: int = 100, indent: str = "    ") -> str:
    """Wrap text with indent for display."""
    lines = []
    for line in text.split("\n"):
        wrapped = textwrap.fill(line, width=width, initial_indent=indent, subsequent_indent=indent)
        lines.append(wrapped)
    return "\n".join(lines)


def run_tagger(opportunities: list[ActivationOpportunity]) -> None:
    """Run the interactive tagging CLI."""
    already_tagged = _load_existing_tags()
    remaining = [
        opp for opp in opportunities
        if (opp.session_id, opp.turn_index, opp.rule_id) not in already_tagged
    ]

    if not remaining:
        print("all opportunities already tagged. nothing to do.")
        return

    print(f"\n{len(remaining)} untagged opportunities ({len(already_tagged)} already tagged)\n")

    tagged_count = 0
    for i, opp in enumerate(remaining):
        print(f"─── [{i + 1}/{len(remaining)}] ─────────────────────────────────────────")
        print(f"  session:  {opp.session_id[:12]}...")
        print(f"  turn:     {opp.turn_index}")
        print(f"  tokens:   {opp.cumulative_input_tokens:,}")
        print(f"  compactions before: {opp.compactions_before}")
        print(f"  rule:     {opp.rule_id}")
        print(f"  context:")
        print(_wrap(opp.context_snippet[:500]))
        print(f"  response:")
        print(_wrap(opp.assistant_response[:800]))
        print()

        while True:
            try:
                answer = input("  compliant? [y/n/partial/skip/quit] > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nquitting tagger.")
                return

            if answer in ("y", "yes"):
                _save_tag(opp, 1.0)
                tagged_count += 1
                break
            elif answer in ("n", "no"):
                _save_tag(opp, 0.0)
                tagged_count += 1
                break
            elif answer in ("p", "partial"):
                _save_tag(opp, 0.5)
                tagged_count += 1
                break
            elif answer in ("s", "skip"):
                break
            elif answer in ("q", "quit"):
                print(f"\ntagged {tagged_count} opportunities this session.")
                return
            else:
                print("  enter y, n, partial, skip, or quit")

    print(f"\ndone. tagged {tagged_count} opportunities.")
