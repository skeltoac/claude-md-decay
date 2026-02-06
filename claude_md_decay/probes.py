"""Natural-feeling probe message generators.

Each probe is a realistic scenario that should trigger a specific rule
without directly asking about the rule itself.
"""

from __future__ import annotations

import random

from .rules import Rule, RULES


def get_probe(rule_id: str, variant: int = 0) -> str:
    """Get a probe message for a specific rule.

    Args:
        rule_id: The rule to probe.
        variant: Which probe variant to use (wraps around).
    """
    rule = RULES[rule_id]
    idx = variant % len(rule.probe_messages)
    return rule.probe_messages[idx]


def get_random_probe(rule_id: str) -> str:
    """Get a random probe message for a rule."""
    rule = RULES[rule_id]
    return random.choice(rule.probe_messages)


def get_probe_sequence(rule_ids: list[str], probes_per_rule: int = 2) -> list[tuple[str, str]]:
    """Generate a sequence of (rule_id, probe_message) tuples.

    Returns probes interleaved — not all probes for one rule followed by another.
    Early and late probes for each rule are spaced apart.
    """
    pairs = []
    for variant in range(probes_per_rule):
        for rule_id in rule_ids:
            pairs.append((rule_id, get_probe(rule_id, variant)))
    return pairs
