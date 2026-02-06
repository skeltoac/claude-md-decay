"""Anthropic Messages API harness for running decay experiment trials."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

from .config import (
    SYSTEM_PROMPT_TEMPLATE,
    MAX_TOKENS_PER_RESPONSE,
    DATA_DIR,
)
from .detection import detect_compliance, ComplianceResult
from .rules import Rule, format_rules_block


@dataclass
class ProbeResult:
    rule_id: str
    probe_index: int
    turn_index: int
    cumulative_input_tokens: int
    probe_message: str
    response_text: str
    compliance: ComplianceResult
    filler_type_before: str
    timestamp: str


@dataclass
class TrialResult:
    trial_id: str
    condition: str
    model: str
    rules_used: list[str]
    probe_results: list[ProbeResult] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_turns: int = 0
    raw_log_path: Path | None = None


class APIHarness:
    """Drives a multi-turn conversation via the Anthropic Messages API."""

    def __init__(self, model: str, rules: list[Rule], max_tokens: int = MAX_TOKENS_PER_RESPONSE):
        self.client = anthropic.Anthropic()
        self.model = model
        self.rules = rules
        self.max_tokens = max_tokens
        self.messages: list[dict] = []
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            rules_block=format_rules_block(rules)
        )
        self.cumulative_input_tokens = 0
        self.cumulative_output_tokens = 0
        self.turn_count = 0
        self._raw_log: list[dict] = []

    def send(self, user_message: str) -> str:
        """Send a user message and return the assistant's text response."""
        self.messages.append({"role": "user", "content": user_message})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=self.messages,
        )

        # Extract text
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
        assistant_text = "\n".join(text_parts)

        self.messages.append({"role": "assistant", "content": assistant_text})

        # Track tokens
        usage = response.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0

        self.cumulative_input_tokens += input_tokens + cache_read + cache_create
        self.cumulative_output_tokens += output_tokens
        self.turn_count += 1

        # Log raw exchange
        self._raw_log.append({
            "turn": self.turn_count,
            "role": "user",
            "content": user_message,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        self._raw_log.append({
            "turn": self.turn_count,
            "role": "assistant",
            "content": assistant_text,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_create,
            },
            "cumulative_input_tokens": self.cumulative_input_tokens,
            "cumulative_output_tokens": self.cumulative_output_tokens,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

        return assistant_text

    def send_probe(
        self,
        rule_id: str,
        probe_message: str,
        probe_index: int,
        filler_type: str,
    ) -> ProbeResult:
        """Send a probe message and score compliance."""
        response_text = self.send(probe_message)
        compliance = detect_compliance(rule_id, response_text, probe_message)

        return ProbeResult(
            rule_id=rule_id,
            probe_index=probe_index,
            turn_index=self.turn_count,
            cumulative_input_tokens=self.cumulative_input_tokens,
            probe_message=probe_message,
            response_text=response_text,
            compliance=compliance,
            filler_type_before=filler_type,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    def save_raw_log(self, path: Path) -> None:
        """Save raw conversation log to JSONL."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for entry in self._raw_log:
                f.write(json.dumps(entry) + "\n")
