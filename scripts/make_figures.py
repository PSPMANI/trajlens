"""Regenerate the README figures from the live code: python scripts/make_figures.py

Every number in the pictures comes from running the rubric, so the README cannot drift
from what the code actually does.
"""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trajlens import CRITERIA, mutate  # noqa: E402
from trajlens.cli import corpus_report  # noqa: E402

BG, FG, MUTED = "#0E1117", "#E6EAF1", "#94A3B8"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "text.color": FG,
                     "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                     "axes.edgecolor": "#2A3140", "font.size": 10})


def heatmap(rep, out):
    ops = rep["operators"]
    rows = [f"{op}  ({r['target']})" for op, r in ops.items()]
    grid = [[r["fires"][c] / r["mutants"] for c in CRITERIA] for r in ops.values()]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.imshow(grid, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CRITERIA)), list(CRITERIA))
    ax.set_yticks(range(len(rows)), rows)
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            if v > 0:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if v > 0.5 else "#7F1D1D")
    ax.set_title(f"Mutation test: {rep['mutants']} injected faults, "
                 f"{rep['detection_rate']:.0%} caught by the targeted criterion, "
                 f"{len(rep['false_alarms'])} false alarms", fontsize=11, pad=12)
    ax.set_xlabel("criterion that fired")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def graders(stats, out):
    names = list(stats)
    fig, ax = plt.subplots(figsize=(9, 1.1 + 0.55 * len(names)))
    for i, name in enumerate(names):
        g = stats[name]
        lo, hi = g["kappa_ci"]
        color = "#22C55E" if "Rubric" in name else "#EF4444"
        ax.plot([lo, hi], [i, i], color=color, lw=3, alpha=0.5, solid_capstyle="round")
        ax.plot(g["kappa"], i, "o", color=color, ms=9)
        ax.text(1.12, i, f"kappa {g['kappa']:.2f}   agree {g['agree']}/{g['n']}   "
                f"recall {g['recall']:.2f}", va="center", fontsize=9, color=MUTED)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlim(-0.6, 1.1)
    ax.axvline(0, color="#2A3140", lw=1)
    ax.set_ylim(len(names) - 0.4, -0.6)
    ax.set_xlabel("Cohen's kappa vs human labels (dot) with 95% bootstrap CI (bar)")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    corpus = json.loads((ROOT / "data" / "trajectories.json").read_text(encoding="utf-8"))
    docs = ROOT / "docs"
    heatmap(mutate.run(corpus), docs / "stress_heatmap.png")
    graders(corpus_report(corpus)["graders"], docs / "grader_agreement.png")
    print("Wrote docs/stress_heatmap.png and docs/grader_agreement.png")
