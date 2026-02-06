"""Orchestrates prospective experiment trials."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import pandas as pd

from .config import Condition, ExperimentConfig, DATA_DIR, PROBE_TOKEN_TARGETS
from .fillers import get_filler, estimate_filler_count
from .harness_api import APIHarness, TrialResult
from .probes import get_probe_sequence
from .rules import get_rules_for_condition, RULES


def run_trial(
    condition: Condition,
    config: ExperimentConfig,
    trial_num: int = 0,
) -> TrialResult:
    """Run a single experiment trial.

    1. Set up the API harness with the appropriate rules
    2. For each probe token target, send fillers to reach it, then send the probe
    3. Score compliance and record results
    """
    trial_id = f"{condition.value}_{trial_num}_{uuid.uuid4().hex[:8]}"
    rules = get_rules_for_condition(condition)
    rule_ids = [r.id for r in rules]

    harness = APIHarness(model=config.model, rules=rules, max_tokens=config.max_tokens)

    # Generate probe sequence
    probe_sequence = get_probe_sequence(rule_ids, probes_per_rule=2)

    # Limit probes if config says so
    targets = config.probe_token_targets
    if config.num_probes:
        targets = targets[:config.num_probes]

    # Assign probes to token targets
    # If more probes than targets, cycle targets; if fewer targets, skip some
    assignments = []
    for idx, (rule_id, probe_msg) in enumerate(probe_sequence):
        if idx >= len(targets):
            break
        assignments.append((targets[idx], rule_id, probe_msg))

    result = TrialResult(
        trial_id=trial_id,
        condition=condition.value,
        model=config.model,
        rules_used=rule_ids,
    )

    print(f"  trial {trial_id}: {len(assignments)} probes, targets up to {targets[-1] if targets else 0:,} tokens")

    for probe_idx, (target_tokens, rule_id, probe_msg) in enumerate(assignments):
        # Send fillers to approach the target token count
        fillers_needed = estimate_filler_count(
            harness.cumulative_input_tokens, target_tokens
        )
        filler_type = condition.value
        for fi in range(fillers_needed):
            filler = get_filler(condition, index=fi, rule_id=rule_id)
            print(f"    filler {fi + 1}/{fillers_needed} (target: {target_tokens:,}, current: {harness.cumulative_input_tokens:,})")
            harness.send(filler)

            # Check if we've overshot
            if harness.cumulative_input_tokens >= target_tokens:
                break

        # Send probe
        print(f"    probe {probe_idx + 1}/{len(assignments)}: {rule_id} @ {harness.cumulative_input_tokens:,} tokens")
        probe_result = harness.send_probe(
            rule_id=rule_id,
            probe_message=probe_msg,
            probe_index=probe_idx,
            filler_type=filler_type,
        )
        result.probe_results.append(probe_result)
        print(f"      compliance: {probe_result.compliance.score} ({probe_result.compliance.evidence[:80]})")

    result.total_input_tokens = harness.cumulative_input_tokens
    result.total_output_tokens = harness.cumulative_output_tokens
    result.total_turns = harness.turn_count

    # Save raw log
    log_path = config.data_dir / f"trial_{trial_id}.jsonl"
    harness.save_raw_log(log_path)
    result.raw_log_path = log_path

    return result


def run_experiment(config: ExperimentConfig) -> list[TrialResult]:
    """Run all trials for all conditions in the config."""
    all_results = []

    for condition in config.conditions:
        print(f"\ncondition: {condition.value}")
        for trial_num in range(config.replications):
            print(f"  replication {trial_num + 1}/{config.replications}")
            result = run_trial(condition, config, trial_num)
            all_results.append(result)

    return all_results


def results_to_dataframe(results: list[TrialResult]) -> pd.DataFrame:
    """Convert trial results to a pandas DataFrame matching the spec's CSV schema."""
    rows = []
    for trial in results:
        for pr in trial.probe_results:
            rows.append({
                "trial_id": trial.trial_id,
                "condition": trial.condition,
                "model": trial.model,
                "rule_id": pr.rule_id,
                "rule_category": RULES[pr.rule_id].category.value,
                "probe_index": pr.probe_index,
                "turn_index": pr.turn_index,
                "cumulative_tokens": pr.cumulative_input_tokens,
                "compliance_score": pr.compliance.score,
                "compliance_binary": pr.compliance.binary,
                "evidence": pr.compliance.evidence,
                "filler_type_before": pr.filler_type_before,
                "num_rules_in_prompt": len(trial.rules_used),
                "compactions_so_far": 0,  # API trials don't have compaction
                "timestamp": pr.timestamp,
            })

    return pd.DataFrame(rows)


def save_results(results: list[TrialResult], path: Path | None = None) -> Path:
    """Save results to CSV, appending to existing data if present."""
    df = results_to_dataframe(results)
    out_path = path or DATA_DIR / "prospective_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        existing = pd.read_csv(out_path)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(out_path, index=False)
    print(f"\nresults saved to {out_path} ({len(df)} rows total)")
    return out_path
