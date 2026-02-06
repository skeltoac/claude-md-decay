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
    """Re-score existing trial data using the LLM judge."""
    from dotenv import load_dotenv
    load_dotenv()

    import json
    import glob
    import pandas as pd
    from .detection import detect_compliance
    from .rules import RULES

    data_dir = Path(args.data) if args.data else DATA_DIR
    trial_files = sorted(glob.glob(str(data_dir / "trial_*.jsonl")))

    if not trial_files:
        print("error: no trial JSONL files found", file=sys.stderr)
        sys.exit(1)

    print(f"rescoring {len(trial_files)} trials with LLM judge...\n")

    rows = []
    for trial_path in trial_files:
        trial_file = Path(trial_path).name
        trial_id = trial_file.replace("trial_", "").replace(".jsonl", "")

        # Infer condition from trial_id (e.g. "baseline_0_abc123" -> "baseline")
        parts = trial_id.split("_")
        # Find where the numeric trial number is
        condition = parts[0]
        for i, p in enumerate(parts[1:], 1):
            if p.isdigit():
                condition = "_".join(parts[:i])
                break

        entries = []
        with open(trial_path) as f:
            for line in f:
                entries.append(json.loads(line))

        # Pair user/assistant turns and identify probes
        # Probes are user messages that match a rule's probe_messages
        from .rules import RULES as all_rules
        probe_texts = {}
        for rule_id, rule in all_rules.items():
            for pm in rule.probe_messages:
                probe_texts[pm.strip()] = rule_id

        probe_index = 0
        for i, entry in enumerate(entries):
            if entry["role"] != "user":
                continue
            content = entry["content"].strip()
            rule_id = probe_texts.get(content)
            if rule_id is None:
                continue

            # Find the assistant response
            if i + 1 < len(entries) and entries[i + 1]["role"] == "assistant":
                resp = entries[i + 1]
                response_text = resp["content"]
                cumulative_tokens = resp.get("cumulative_input_tokens", 0)

                print(f"  {trial_id} | {rule_id:35s} @ {cumulative_tokens:>7,} tokens ...", end=" ", flush=True)
                result = detect_compliance(rule_id, response_text, content, method="llm_judge")
                print(f"{result.score:.1f}  ({result.evidence[:60]})")

                rows.append({
                    "trial_id": trial_id,
                    "condition": condition,
                    "model": "claude-sonnet-4-5-20250929",
                    "rule_id": rule_id,
                    "rule_category": RULES[rule_id].category.value,
                    "probe_index": probe_index,
                    "turn_index": i // 2,
                    "cumulative_tokens": cumulative_tokens,
                    "compliance_score": result.score,
                    "compliance_binary": result.binary,
                    "evidence": result.evidence,
                    "filler_type_before": condition,
                    "num_rules_in_prompt": 8 if "few" not in condition else 3,
                    "compactions_so_far": 0,
                    "timestamp": resp.get("timestamp", ""),
                })
                probe_index += 1

    df = pd.DataFrame(rows)
    out_path = data_dir / "prospective_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nrescored {len(df)} probes across {len(trial_files)} trials")
    print(f"saved to {out_path}")

    # Quick summary
    print("\nsummary:")
    for rule_id in sorted(df["rule_id"].unique()):
        sub = df[df["rule_id"] == rule_id]
        print(f"  {rule_id:35s}  n={len(sub):2d}  mean={sub['compliance_score'].mean():.2f}  min={sub['compliance_score'].min():.1f}")


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
