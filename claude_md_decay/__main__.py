"""CLI entry point: run/analyze/retro/tag subcommands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Condition, ExperimentConfig, DATA_DIR


def cmd_retro(args: argparse.Namespace) -> None:
    """Parse existing conversations and report activation opportunities."""
    from .retrospective import parse_all_sessions, find_activation_opportunities, summarize_sessions

    sessions_dir = Path(args.sessions_dir)
    if not sessions_dir.exists():
        print(f"error: directory not found: {sessions_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"parsing sessions from {sessions_dir}...")
    sessions = parse_all_sessions(sessions_dir)

    summary = summarize_sessions(sessions)
    print(f"\nsessions:     {summary['sessions']}")
    print(f"total turns:  {summary['total_turns']:,}")
    print(f"compactions:  {summary['total_compactions']}")
    print(f"sessions w/ compaction: {summary['sessions_with_compactions']}")
    print(f"total size:   {summary['total_size_mb']} MB")

    # Find activation opportunities
    print("\nscanning for activation opportunities...")
    all_opps = []
    for session in sessions:
        opps = find_activation_opportunities(session)
        all_opps.extend(opps)

    # Summarize by rule
    from collections import Counter
    by_rule = Counter(opp.rule_id for opp in all_opps)
    print(f"\ntotal opportunities: {len(all_opps)}")
    for rule_id, count in sorted(by_rule.items(), key=lambda x: -x[1]):
        print(f"  {rule_id}: {count}")

    # Save opportunities for tagging
    import json
    opps_path = DATA_DIR / "retrospective_opportunities.jsonl"
    opps_path.parent.mkdir(parents=True, exist_ok=True)
    with open(opps_path, "w") as f:
        for opp in all_opps:
            f.write(json.dumps({
                "session_id": opp.session_id,
                "turn_index": opp.turn_index,
                "cumulative_tokens": opp.cumulative_input_tokens,
                "rule_id": opp.rule_id,
                "context_snippet": opp.context_snippet,
                "assistant_response": opp.assistant_response,
                "compactions_before": opp.compactions_before,
                "timestamp": opp.timestamp,
            }) + "\n")
    print(f"\nopportunities saved to {opps_path}")


def cmd_tag(args: argparse.Namespace) -> None:
    """Launch interactive tagging CLI."""
    import json
    from .retrospective import ActivationOpportunity
    from .tagger import run_tagger

    opps_path = DATA_DIR / "retrospective_opportunities.jsonl"
    if not opps_path.exists():
        print("error: no opportunities file found. run 'retro' first.", file=sys.stderr)
        sys.exit(1)

    opportunities = []
    with open(opps_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            opportunities.append(ActivationOpportunity(
                session_id=rec["session_id"],
                turn_index=rec["turn_index"],
                cumulative_input_tokens=rec["cumulative_tokens"],
                rule_id=rec["rule_id"],
                context_snippet=rec["context_snippet"],
                assistant_response=rec["assistant_response"],
                compactions_before=rec["compactions_before"],
                timestamp=rec["timestamp"],
            ))

    run_tagger(opportunities)


def cmd_run(args: argparse.Namespace) -> None:
    """Run prospective experiment trials."""
    from dotenv import load_dotenv
    load_dotenv()

    from .runner import run_experiment, save_results

    conditions = [Condition(c) for c in args.condition]
    config = ExperimentConfig(
        conditions=conditions,
        replications=args.trials,
        model=args.model,
        num_probes=args.probes if args.probes else None,
    )

    print(f"running experiment: conditions={[c.value for c in conditions]}, "
          f"trials={args.trials}, model={args.model}")

    results = run_experiment(config)
    save_results(results)


def cmd_rescore(args: argparse.Namespace) -> None:
    """Re-score existing trial data using the LLM judge.

    Walks every assistant response in every trial:
    - "every response" rules (pineapple, noir) are checked on ALL turns
    - Conditional rules are checked only when the user message triggers them
    """
    from dotenv import load_dotenv
    load_dotenv()

    import json
    import glob
    import pandas as pd
    from .detection import detect_compliance, init_judge_log, close_judge_log
    from .rules import RULES

    data_dir = Path(args.data) if args.data else DATA_DIR
    trial_files = sorted(glob.glob(str(data_dir / "trial_*.jsonl")))

    if not trial_files:
        print("error: no trial JSONL files found", file=sys.stderr)
        sys.exit(1)

    log_path = init_judge_log(data_dir / "judge_log.jsonl")
    print(f"judge interactions will be logged to {log_path}")

    # Rules that apply to every single response
    EVERY_RESPONSE_RULES = {"pineapple_canary", "noir_monologue"}

    # Build lookup for conditional probe triggers
    probe_triggers: dict[str, str] = {}  # message text -> rule_id
    for rule_id, rule in RULES.items():
        if rule_id in EVERY_RESPONSE_RULES:
            continue
        for pm in rule.probe_messages:
            probe_triggers[pm.strip()] = rule_id

    print(f"rescoring {len(trial_files)} trials — every turn, every applicable rule\n")

    rows = []
    for trial_path in trial_files:
        trial_file = Path(trial_path).name
        trial_id = trial_file.replace("trial_", "").replace(".jsonl", "")

        # Infer condition
        parts = trial_id.split("_")
        condition = parts[0]
        for i, p in enumerate(parts[1:], 1):
            if p.isdigit():
                condition = "_".join(parts[:i])
                break

        entries = []
        with open(trial_path) as f:
            for line in f:
                entries.append(json.loads(line))

        # Figure out which rules are in this trial's system prompt
        trial_rules = set(RULES.keys())
        if "few" in condition:
            from .rules import FEW_RULES
            trial_rules = set(FEW_RULES)

        # Which every-response rules are active in this trial?
        active_every_response = EVERY_RESPONSE_RULES & trial_rules

        print(f"  {trial_id} ({len(entries)} entries, rules: {len(trial_rules)})")

        # Walk user/assistant pairs
        turn_num = 0
        for i, entry in enumerate(entries):
            if entry["role"] != "user":
                continue
            # Find paired assistant response
            if i + 1 >= len(entries) or entries[i + 1]["role"] != "assistant":
                continue

            user_text = entry["content"].strip()
            resp = entries[i + 1]
            response_text = resp["content"]
            cumulative_tokens = resp.get("cumulative_input_tokens", 0)
            timestamp = resp.get("timestamp", "")
            turn_num += 1

            # Collect all rules to check for this turn
            rules_to_check: list[tuple[str, str]] = []  # (rule_id, trigger_type)

            # Every-response rules always apply
            for rule_id in sorted(active_every_response):
                rules_to_check.append((rule_id, "every_response"))

            # Conditional rules only when probe matches
            triggered_rule = probe_triggers.get(user_text)
            if triggered_rule and triggered_rule in trial_rules:
                rules_to_check.append((triggered_rule, "probe"))

            # Score each applicable rule
            for rule_id, trigger_type in rules_to_check:
                result = detect_compliance(rule_id, response_text, user_text, method="llm_judge")
                tag = "P" if trigger_type == "probe" else "E"
                status = "ok" if result.score >= 0.8 else "FAIL" if result.score <= 0.2 else "partial"
                print(f"    [{tag}] turn {turn_num:2d} | {rule_id:30s} @ {cumulative_tokens:>7,}t  {result.score:.1f} {status}")

                rows.append({
                    "trial_id": trial_id,
                    "condition": condition,
                    "model": "claude-sonnet-4-5-20250929",
                    "rule_id": rule_id,
                    "rule_category": RULES[rule_id].category.value,
                    "probe_index": turn_num,
                    "turn_index": turn_num,
                    "cumulative_tokens": cumulative_tokens,
                    "compliance_score": result.score,
                    "compliance_binary": result.binary,
                    "evidence": result.evidence,
                    "filler_type_before": trigger_type,
                    "num_rules_in_prompt": len(trial_rules),
                    "compactions_so_far": 0,
                    "timestamp": timestamp,
                })

        print()

    close_judge_log()

    df = pd.DataFrame(rows)
    out_path = data_dir / "prospective_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nrescored {len(df)} judgments across {len(trial_files)} trials")
    print(f"results: {out_path}")
    print(f"judge log: {data_dir / 'judge_log.jsonl'}")

    # Summary
    print("\nsummary:")
    for rule_id in sorted(df["rule_id"].unique()):
        sub = df[df["rule_id"] == rule_id]
        print(f"  {rule_id:35s}  n={len(sub):3d}  mean={sub['compliance_score'].mean():.2f}  min={sub['compliance_score'].min():.1f}")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze results and generate plots."""
    from .analyze import run_analysis

    data_dir = Path(args.data) if args.data else DATA_DIR
    run_analysis(data_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="claude_md_decay",
        description="CLAUDE.md instruction decay experiment",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # retro
    p_retro = subparsers.add_parser("retro", help="parse existing conversations")
    p_retro.add_argument(
        "--sessions-dir",
        required=True,
        help="path to directory containing .jsonl session files",
    )

    # tag
    p_tag = subparsers.add_parser("tag", help="interactive compliance tagging")

    # run
    p_run = subparsers.add_parser("run", help="run prospective experiment trials")
    p_run.add_argument(
        "--condition",
        nargs="+",
        default=["baseline"],
        choices=[c.value for c in Condition],
        help="experiment conditions to run",
    )
    p_run.add_argument("--trials", type=int, default=3, help="replications per condition")
    p_run.add_argument("--model", default="claude-sonnet-4-5-20250929", help="model to use")
    p_run.add_argument("--probes", type=int, default=None, help="limit probes per trial (for testing)")

    # rescore
    p_rescore = subparsers.add_parser("rescore", help="re-score trials with LLM judge")
    p_rescore.add_argument("--data", default=None, help="path to data directory")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="analyze results and generate plots")
    p_analyze.add_argument("--data", default=None, help="path to data directory")

    args = parser.parse_args()

    commands = {
        "retro": cmd_retro,
        "tag": cmd_tag,
        "run": cmd_run,
        "rescore": cmd_rescore,
        "analyze": cmd_analyze,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
