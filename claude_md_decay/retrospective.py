"""Parse JSONL conversations, find activation opportunities, compute token usage."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CompactionEvent:
    turn_index: int
    timestamp: str
    pre_tokens: int
    trigger: str  # "auto" or "manual"


@dataclass
class ActivationOpportunity:
    """A turn where a CLAUDE.md rule could have activated."""
    session_id: str
    turn_index: int
    cumulative_input_tokens: int
    rule_id: str
    context_snippet: str  # the user message or tool output that triggered
    assistant_response: str  # the assistant's full text response
    compactions_before: int  # how many compactions happened before this turn
    timestamp: str


@dataclass
class Turn:
    index: int
    role: str  # "user", "assistant", "system"
    text_content: str  # extracted text from message content
    tool_uses: list[dict]  # tool_use blocks (name, input)
    tool_results: list[dict]  # tool_result blocks
    usage: dict | None  # token usage from this turn
    timestamp: str
    cumulative_input_tokens: int = 0
    compactions_before: int = 0
    is_compaction_boundary: bool = False
    subtype: str = ""


@dataclass
class Session:
    id: str
    path: Path
    turns: list[Turn] = field(default_factory=list)
    compactions: list[CompactionEvent] = field(default_factory=list)
    total_lines: int = 0
    file_size_bytes: int = 0

    @property
    def total_input_tokens(self) -> int:
        for turn in reversed(self.turns):
            if turn.cumulative_input_tokens > 0:
                return turn.cumulative_input_tokens
        return 0


def _extract_text(message: dict | str | None) -> str:
    """Extract plain text from a message's content field."""
    if message is None:
        return ""
    if isinstance(message, str):
        return message

    content = message.get("content", "")
    if isinstance(content, str):
        return content

    parts = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "thinking":
                    pass  # skip thinking blocks
            elif isinstance(block, str):
                parts.append(block)
    return "\n".join(parts)


def _extract_tool_uses(message: dict | None) -> list[dict]:
    """Extract tool_use blocks from message content."""
    if not message or not isinstance(message, dict):
        return []
    content = message.get("content", [])
    if not isinstance(content, list):
        return []
    tools = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            tools.append({
                "name": block.get("name", ""),
                "input": block.get("input", {}),
            })
    return tools


def _extract_tool_results(message: dict | None) -> list[dict]:
    """Extract tool_result blocks from message content."""
    if not message or not isinstance(message, dict):
        return []
    content = message.get("content", [])
    if not isinstance(content, list):
        return []
    results = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            results.append(block)
    return results


def parse_session(path: Path) -> Session:
    """Parse a single JSONL file into a Session."""
    session_id = path.stem
    session = Session(id=session_id, path=path, file_size_bytes=path.stat().st_size)

    cumulative_input = 0
    compaction_count = 0
    turn_index = 0

    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            session.total_lines = line_num
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = obj.get("type", "")
            subtype = obj.get("subtype", "")
            timestamp = obj.get("timestamp", "")

            # Compaction boundary
            if msg_type == "system" and subtype == "compact_boundary":
                meta = obj.get("compactMetadata", {})
                compaction_count += 1
                ce = CompactionEvent(
                    turn_index=turn_index,
                    timestamp=timestamp,
                    pre_tokens=meta.get("preTokens", 0),
                    trigger=meta.get("trigger", "unknown"),
                )
                session.compactions.append(ce)
                turn = Turn(
                    index=turn_index,
                    role="system",
                    text_content="[compaction boundary]",
                    tool_uses=[],
                    tool_results=[],
                    usage=None,
                    timestamp=timestamp,
                    cumulative_input_tokens=cumulative_input,
                    compactions_before=compaction_count,
                    is_compaction_boundary=True,
                    subtype=subtype,
                )
                session.turns.append(turn)
                turn_index += 1
                continue

            # Skip non-message types
            if msg_type not in ("user", "assistant"):
                continue

            message = obj.get("message")
            if not isinstance(message, dict):
                continue

            text = _extract_text(message)
            tool_uses = _extract_tool_uses(message)
            tool_results = _extract_tool_results(message)

            # Extract usage
            usage = message.get("usage")
            if usage:
                input_toks = usage.get("input_tokens", 0)
                cache_read = usage.get("cache_read_input_tokens", 0)
                cache_create = usage.get("cache_creation_input_tokens", 0)
                cumulative_input += input_toks + cache_read + cache_create

            turn = Turn(
                index=turn_index,
                role=msg_type,
                text_content=text,
                tool_uses=tool_uses,
                tool_results=tool_results,
                usage=usage,
                timestamp=timestamp,
                cumulative_input_tokens=cumulative_input,
                compactions_before=compaction_count,
                subtype=subtype,
            )
            session.turns.append(turn)
            turn_index += 1

    return session


def parse_all_sessions(sessions_dir: Path) -> list[Session]:
    """Parse all JSONL files in a directory."""
    sessions = []
    for path in sorted(sessions_dir.glob("*.jsonl")):
        sessions.append(parse_session(path))
    return sessions


# ── Activation opportunity detectors ──────────────────────────────────────────

def _find_git_commands(turn: Turn) -> list[str]:
    """Extract git commands from tool_use Bash blocks."""
    cmds = []
    for tool in turn.tool_uses:
        if tool["name"] == "Bash":
            cmd = tool["input"].get("command", "")
            if "git " in cmd:
                cmds.append(cmd)
    return cmds


def _get_assistant_response_after(session: Session, turn_idx: int) -> str:
    """Get the text content of assistant turns following a given turn index."""
    parts = []
    for turn in session.turns[turn_idx + 1:]:
        if turn.role == "assistant":
            parts.append(turn.text_content)
            # Include tool commands too
            for tool in turn.tool_uses:
                if tool["name"] == "Bash":
                    parts.append(f"[bash: {tool['input'].get('command', '')}]")
        elif turn.role == "user":
            break  # stop at next user turn
    return "\n".join(parts)


def find_activation_opportunities(session: Session) -> list[ActivationOpportunity]:
    """Scan a session for turns where CLAUDE.md rules could have activated."""
    opportunities = []

    for i, turn in enumerate(session.turns):
        # Git destructive commands → git_status_before_destructive, no_amend_commits
        if turn.role == "assistant":
            git_cmds = _find_git_commands(turn)
            for cmd in git_cmds:
                if re.search(r"git\s+(reset|clean|checkout\s+\.)", cmd):
                    opportunities.append(ActivationOpportunity(
                        session_id=session.id,
                        turn_index=turn.index,
                        cumulative_input_tokens=turn.cumulative_input_tokens,
                        rule_id="git_status_before_destructive",
                        context_snippet=cmd[:300],
                        assistant_response=turn.text_content[:2000],
                        compactions_before=turn.compactions_before,
                        timestamp=turn.timestamp,
                    ))
                if "--amend" in cmd:
                    opportunities.append(ActivationOpportunity(
                        session_id=session.id,
                        turn_index=turn.index,
                        cumulative_input_tokens=turn.cumulative_input_tokens,
                        rule_id="no_amend_commits",
                        context_snippet=cmd[:300],
                        assistant_response=turn.text_content[:2000],
                        compactions_before=turn.compactions_before,
                        timestamp=turn.timestamp,
                    ))

        # User asks to commit → pre_commit_review
        if turn.role == "user" and re.search(r"(?i)\bcommit\b", turn.text_content):
            response = _get_assistant_response_after(session, i)
            if response:
                opportunities.append(ActivationOpportunity(
                    session_id=session.id,
                    turn_index=turn.index,
                    cumulative_input_tokens=turn.cumulative_input_tokens,
                    rule_id="pre_commit_review",
                    context_snippet=turn.text_content[:300],
                    assistant_response=response[:2000],
                    compactions_before=turn.compactions_before,
                    timestamp=turn.timestamp,
                ))

        # Code with logging + patient/PHI → phi_no_log
        if turn.role == "user":
            text = turn.text_content.lower()
            if ("logger" in text or "logging" in text or "log." in text) and \
               ("patient" in text or "ssn" in text or "mrn" in text or "dob" in text):
                response = _get_assistant_response_after(session, i)
                if response:
                    opportunities.append(ActivationOpportunity(
                        session_id=session.id,
                        turn_index=turn.index,
                        cumulative_input_tokens=turn.cumulative_input_tokens,
                        rule_id="phi_no_log",
                        context_snippet=turn.text_content[:300],
                        assistant_response=response[:2000],
                        compactions_before=turn.compactions_before,
                        timestamp=turn.timestamp,
                    ))

        # PR/issue comment drafting → sign_with_claude, ask_before_posting
        if turn.role == "user":
            text = turn.text_content.lower()
            is_draft = re.search(r"(?i)(draft|write|compose).{0,20}(comment|reply|response|pr|issue)", text)
            is_post = re.search(r"(?i)(post|comment on|reply to).{0,20}(pr|issue|github)", text)
            if is_draft or is_post:
                response = _get_assistant_response_after(session, i)
                if response:
                    if is_draft:
                        opportunities.append(ActivationOpportunity(
                            session_id=session.id,
                            turn_index=turn.index,
                            cumulative_input_tokens=turn.cumulative_input_tokens,
                            rule_id="sign_with_claude",
                            context_snippet=turn.text_content[:300],
                            assistant_response=response[:2000],
                            compactions_before=turn.compactions_before,
                            timestamp=turn.timestamp,
                        ))
                    if is_post:
                        opportunities.append(ActivationOpportunity(
                            session_id=session.id,
                            turn_index=turn.index,
                            cumulative_input_tokens=turn.cumulative_input_tokens,
                            rule_id="ask_before_posting",
                            context_snippet=turn.text_content[:300],
                            assistant_response=response[:2000],
                            compactions_before=turn.compactions_before,
                            timestamp=turn.timestamp,
                        ))

    return opportunities


def summarize_sessions(sessions: list[Session]) -> dict:
    """Print summary stats for parsed sessions."""
    total_turns = sum(len(s.turns) for s in sessions)
    total_compactions = sum(len(s.compactions) for s in sessions)
    sessions_with_compactions = sum(1 for s in sessions if s.compactions)
    total_size_mb = sum(s.file_size_bytes for s in sessions) / 1024 / 1024

    return {
        "sessions": len(sessions),
        "total_turns": total_turns,
        "total_compactions": total_compactions,
        "sessions_with_compactions": sessions_with_compactions,
        "total_size_mb": round(total_size_mb, 1),
    }
