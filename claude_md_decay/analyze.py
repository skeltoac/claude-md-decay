"""Statistics, decay curves, half-life estimation, and plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import sem

from .config import TOKEN_BUCKETS, DATA_DIR
from .rules import RULES


# ── Decay model ───────────────────────────────────────────────────────────────

def logistic_decay(x: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
    """Logistic decay: compliance drops from L to 0 around x0."""
    return L / (1 + np.exp(k * (x - x0)))


def estimate_half_life(tokens: np.ndarray, scores: np.ndarray) -> float | None:
    """Fit logistic decay and return the token count where compliance = 0.5.

    Returns None if the fit fails or data is insufficient.
    """
    if len(tokens) < 3:
        return None
    try:
        popt, _ = curve_fit(
            logistic_decay,
            tokens,
            scores,
            p0=[1.0, 0.00005, 80000],
            bounds=([0.5, 0.0, 0.0], [1.0, 0.001, 300000]),
            maxfev=5000,
        )
        return float(popt[2])  # x0 is the inflection point ≈ half-life
    except (RuntimeError, ValueError):
        return None


# ── Token bucketing ───────────────────────────────────────────────────────────

def assign_bucket(tokens: int) -> str:
    """Assign a token count to a bucket label."""
    for lo, hi in TOKEN_BUCKETS:
        if lo <= tokens < hi:
            return f"{lo // 1000}k-{hi // 1000}k"
    return f">{TOKEN_BUCKETS[-1][1] // 1000}k"


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_decay_curves(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot 1: compliance vs cumulative tokens, one line per rule."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for rule_id in df["rule_id"].unique():
        subset = df[df["rule_id"] == rule_id].sort_values("cumulative_tokens")
        tokens = subset["cumulative_tokens"].values
        scores = subset["compliance_score"].values

        # Smooth with rolling mean if enough points
        if len(tokens) >= 3:
            window = min(3, len(tokens))
            smooth_scores = pd.Series(scores).rolling(window, center=True, min_periods=1).mean()
            se = pd.Series(scores).rolling(window, center=True, min_periods=1).apply(
                lambda x: sem(x) if len(x) > 1 else 0
            )
            ax.plot(tokens, smooth_scores, label=rule_id, marker="o", markersize=3)
            ax.fill_between(tokens, smooth_scores - se, smooth_scores + se, alpha=0.15)
        else:
            ax.plot(tokens, scores, label=rule_id, marker="o", markersize=3)

    ax.axvline(x=100_000, color="gray", linestyle="--", alpha=0.5, label="~compaction zone")
    ax.set_xlabel("cumulative input tokens")
    ax.set_ylabel("compliance score")
    ax.set_title("instruction decay curves")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "plot1_decay_curves.png", dpi=150)
    plt.close(fig)
    print("  plot 1: decay curves")


def plot_category_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot 2: boxplots of early vs late compliance by rule category."""
    df = df.copy()
    df["phase"] = df["cumulative_tokens"].apply(lambda t: "early (<50k)" if t < 50_000 else "late (>50k)")

    categories = df["rule_category"].unique()
    fig, axes = plt.subplots(1, len(categories), figsize=(4 * len(categories), 5), sharey=True)
    if len(categories) == 1:
        axes = [axes]

    for ax, cat in zip(axes, sorted(categories)):
        cat_df = df[df["rule_category"] == cat]
        early = cat_df[cat_df["phase"] == "early (<50k)"]["compliance_score"]
        late = cat_df[cat_df["phase"] == "late (>50k)"]["compliance_score"]

        data = [early.values, late.values] if len(late) > 0 else [early.values]
        labels = ["early", "late"] if len(late) > 0 else ["early"]

        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], ["#4CAF50", "#F44336"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(cat, fontsize=10)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle("early vs late compliance by rule category")
    fig.tight_layout()
    fig.savefig(out_dir / "plot2_category_comparison.png", dpi=150)
    plt.close(fig)
    print("  plot 2: category comparison")


def plot_condition_effects(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot 3: small multiples per rule showing all conditions overlaid."""
    rule_ids = sorted(df["rule_id"].unique())
    conditions = sorted(df["condition"].unique())
    n = len(rule_ids)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharey=True)
    axes = np.atleast_2d(axes)

    for idx, rule_id in enumerate(rule_ids):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        for cond in conditions:
            subset = df[(df["rule_id"] == rule_id) & (df["condition"] == cond)]
            if subset.empty:
                continue
            subset = subset.sort_values("cumulative_tokens")
            ax.plot(
                subset["cumulative_tokens"],
                subset["compliance_score"],
                label=cond,
                marker="o",
                markersize=3,
                alpha=0.7,
            )
        ax.set_title(rule_id, fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        if idx == 0:
            ax.legend(fontsize=7)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    fig.suptitle("condition effects by rule")
    fig.tight_layout()
    fig.savefig(out_dir / "plot3_condition_effects.png", dpi=150)
    plt.close(fig)
    print("  plot 3: condition effects")


def plot_novelty_vs_natural(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot 4: novelty rules (pineapple, noir) vs natural rules overlaid."""
    novelty_rules = ["pineapple_canary", "noir_monologue"]
    natural_rules = ["phi_no_log", "pre_commit_review"]
    target_rules = novelty_rules + natural_rules

    fig, ax = plt.subplots(figsize=(10, 6))
    styles = {
        "pineapple_canary": ("tab:orange", "--"),
        "noir_monologue": ("tab:purple", "--"),
        "phi_no_log": ("tab:blue", "-"),
        "pre_commit_review": ("tab:green", "-"),
    }

    for rule_id in target_rules:
        subset = df[df["rule_id"] == rule_id].sort_values("cumulative_tokens")
        if subset.empty:
            continue
        color, ls = styles[rule_id]
        ax.plot(
            subset["cumulative_tokens"],
            subset["compliance_score"],
            label=rule_id,
            color=color,
            linestyle=ls,
            marker="o",
            markersize=4,
        )

    ax.set_xlabel("cumulative input tokens")
    ax.set_ylabel("compliance score")
    ax.set_title("novelty rules vs natural rules")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "plot4_novelty_vs_natural.png", dpi=150)
    plt.close(fig)
    print("  plot 4: novelty vs natural")


def plot_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot 5: rules (y) x token buckets (x), color = compliance score."""
    df = df.copy()
    df["bucket"] = df["cumulative_tokens"].apply(assign_bucket)

    bucket_labels = [f"{lo // 1000}k-{hi // 1000}k" for lo, hi in TOKEN_BUCKETS]
    rule_ids = sorted(df["rule_id"].unique())

    matrix = np.full((len(rule_ids), len(bucket_labels)), np.nan)
    for ri, rule_id in enumerate(rule_ids):
        for bi, bucket in enumerate(bucket_labels):
            subset = df[(df["rule_id"] == rule_id) & (df["bucket"] == bucket)]
            if not subset.empty:
                matrix[ri, bi] = subset["compliance_score"].mean()

    fig, ax = plt.subplots(figsize=(12, max(4, len(rule_ids) * 0.6)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(bucket_labels)))
    ax.set_xticklabels(bucket_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(rule_ids)))
    ax.set_yticklabels(rule_ids)
    ax.set_xlabel("token bucket")
    ax.set_ylabel("rule")
    ax.set_title("compliance heatmap")
    fig.colorbar(im, label="mean compliance")

    # Annotate cells
    for ri in range(len(rule_ids)):
        for bi in range(len(bucket_labels)):
            val = matrix[ri, bi]
            if not np.isnan(val):
                ax.text(bi, ri, f"{val:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if val < 0.5 else "black")

    fig.tight_layout()
    fig.savefig(out_dir / "plot5_heatmap.png", dpi=150)
    plt.close(fig)
    print("  plot 5: heatmap")


def plot_half_lives(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot 6: half-life bar chart — token count where compliance drops below 50%."""
    rule_ids = sorted(df["rule_id"].unique())
    half_lives = {}

    for rule_id in rule_ids:
        subset = df[df["rule_id"] == rule_id].sort_values("cumulative_tokens")
        tokens = subset["cumulative_tokens"].values.astype(float)
        scores = subset["compliance_score"].values.astype(float)
        hl = estimate_half_life(tokens, scores)
        half_lives[rule_id] = hl

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    rules_with_hl = {k: v for k, v in half_lives.items() if v is not None}
    rules_without = [k for k, v in half_lives.items() if v is None]

    if rules_with_hl:
        sorted_rules = sorted(rules_with_hl.items(), key=lambda x: x[1])
        names = [r[0] for r in sorted_rules]
        values = [r[1] for r in sorted_rules]
        colors = ["#F44336" if v < 50_000 else "#FF9800" if v < 100_000 else "#4CAF50" for v in values]
        ax.barh(names, values, color=colors)
        ax.set_xlabel("estimated half-life (tokens)")
        ax.set_title("instruction half-life (token count at 50% compliance)")
        for i, v in enumerate(values):
            ax.text(v + 1000, i, f"{v:,.0f}", va="center", fontsize=8)

    if rules_without:
        ax.text(0.98, 0.02, f"no fit: {', '.join(rules_without)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(out_dir / "plot6_half_lives.png", dpi=150)
    plt.close(fig)
    print("  plot 6: half-lives")


# ── Retrospective plots ──────────────────────────────────────────────────────

def plot_retrospective(tags_path: Path, out_dir: Path) -> None:
    """Generate plots from retrospective tagging data."""
    if not tags_path.exists():
        print("  no retrospective tags found, skipping")
        return

    rows = []
    with open(tags_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("  no retrospective tags found, skipping")
        return

    df = pd.DataFrame(rows)
    print(f"  retrospective data: {len(df)} tagged opportunities")

    # Decay curve with compaction markers
    fig, ax = plt.subplots(figsize=(12, 7))
    for rule_id in df["rule_id"].unique():
        subset = df[df["rule_id"] == rule_id].sort_values("cumulative_tokens")
        ax.scatter(
            subset["cumulative_tokens"],
            subset["compliance"],
            label=rule_id,
            alpha=0.6,
            s=30,
        )

    # Mark compaction boundaries
    compaction_tokens = df[df["compactions_before"] > 0].groupby("compactions_before")["cumulative_tokens"].min()
    for ct in compaction_tokens:
        ax.axvline(x=ct, color="red", linestyle=":", alpha=0.5)

    ax.set_xlabel("cumulative input tokens")
    ax.set_ylabel("compliance (tagged)")
    ax.set_title("retrospective compliance (real conversations)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_retro_decay.png", dpi=150)
    plt.close(fig)
    print("  plot: retrospective decay")


# ── Main entry ────────────────────────────────────────────────────────────────

def run_analysis(data_dir: Path) -> None:
    """Run all analysis and generate plots."""
    plots_dir = data_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"analyzing data in {data_dir}...")
    print(f"plots will be saved to {plots_dir}\n")

    # Load prospective results
    prospective_path = data_dir / "prospective_results.csv"
    if prospective_path.exists():
        df = pd.read_csv(prospective_path)
        print(f"prospective data: {len(df)} rows, {df['trial_id'].nunique()} trials\n")
        print("generating plots...")

        plot_decay_curves(df, plots_dir)
        plot_category_comparison(df, plots_dir)
        plot_condition_effects(df, plots_dir)
        plot_novelty_vs_natural(df, plots_dir)
        plot_heatmap(df, plots_dir)
        plot_half_lives(df, plots_dir)
    else:
        print("no prospective results found (prospective_results.csv)")

    # Retrospective
    print()
    tags_path = data_dir / "retrospective_tags.jsonl"
    plot_retrospective(tags_path, plots_dir)

    print(f"\ndone. all plots saved to {plots_dir}")
