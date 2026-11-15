"""Figures for bench/paper/b0_v0.5.md, read only from data/processed/bench/b0_partial.json.

matplotlib is not a project dependency; run it in a throwaway environment:
uv run --with matplotlib python bench/paper/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PARTIAL = ROOT / "data/processed/bench/b0_partial.json"
OUT = ROOT / "bench/paper/figures"

MODELS = [
    ("anthropic/claude-opus-5-5", "Claude Opus 5.5"),
    ("opencode-go/gpt-6-luna", "GPT-6 Luna"),
    ("opencode-go/glm-5.3-flash", "GLM 5.3 Flash"),
    ("opencode-go/deepseek-v4.1-flash", "DeepSeek V4.1 Flash"),
]
FAMILIES = [
    ("hidden_limit", "hidden\nlimit"), ("quoted_price", "quoted\nprice"), ("field_fact", "field\nfact"),
    ("priority_order", "priority\norder"), ("infeasible", "infeasible\nrequest"), ("missing_data", "missing\ndata"),
]
COLOURS = ["#1b6ca8", "#e08a1e", "#3a9a5b", "#b8433c"]


def metrics_figure(d):
    names = [
        ("accuracy", "Exact cycle = key"),
        ("task_accuracy_including_behavioural_rubrics", "Task correct (key or rubric)"),
        (("v2", "grounded_accuracy"), "v2 grounded accuracy (post hoc)"),
        ("hygienic_accuracy_v1", "v1 hygienic accuracy (registered)"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    width = 0.2
    for j, (field, label) in enumerate(names):
        vals = []
        for alias, _ in MODELS:
            o = d["models"][alias]["overall"]
            vals.append(100 * (o[field[0]][field[1]] if isinstance(field, tuple) else o[field]))
        xs = [i + (j - 1.5) * width for i in range(len(MODELS))]
        ax.bar(xs, vals, width, label=label, color=COLOURS[j])
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([f"{name}\nn = {d['models'][alias]['overall']['n']}" for alias, name in MODELS])
    ax.set_ylabel("% of completed repeat-0 episodes")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    ax.set_title("B0 v0.5, repeat 0 only (one repeat of three); unequal n per model", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_accuracy_by_model.png", dpi=200)
    plt.close(fig)


def family_figure(d):
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    grid = []
    for alias, _ in MODELS:
        row = []
        for fam, _ in FAMILIES:
            b = d["models"][alias]["families"][fam]
            row.append((b["v2"]["grounded_accuracy"], b["n"]))
        grid.append(row)
    im = ax.imshow([[v for v, _ in row] for row in grid], vmin=0, vmax=1, cmap="viridis", aspect="auto")
    for i, row in enumerate(grid):
        for j, (v, n) in enumerate(row):
            k = round(v * n)
            ax.text(j, i, f"{k}/{n}", ha="center", va="center", fontsize=8,
                    color="white" if v < 0.6 else "black")
    ax.set_xticks(range(len(FAMILIES)))
    ax.set_xticklabels([label for _, label in FAMILIES], fontsize=8)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([name for _, name in MODELS], fontsize=8)
    ax.set_title("v2 grounded accuracy by family (k/n; cells hold 6-12 episodes)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_grounded_by_family.png", dpi=200)
    plt.close(fig)


def cost_figure(d):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for (alias, name), colour in zip(MODELS, COLOURS):
        runs = d["models"][alias]["runs"]
        ax.scatter([r["seconds"] for r in runs], [r["output_tokens"] for r in runs], s=12, alpha=0.7,
                   color=colour, label=f"{name} (n = {len(runs)})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Episode wall time (s)")
    ax.set_ylabel("Generated tokens (output + reasoning)")
    ax.set_title("Per-episode time and generated tokens, repeat 0", fontsize=9)
    ax.legend(fontsize=7.5, framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_time_tokens.png", dpi=200)
    plt.close(fig)


def main():
    d = json.loads(PARTIAL.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    metrics_figure(d)
    family_figure(d)
    cost_figure(d)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
