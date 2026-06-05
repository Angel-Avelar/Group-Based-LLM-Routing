import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import json
import argparse

parser = argparse.ArgumentParser(description="Plot pairwise win rate grid for top models.")
parser.add_argument("n_groups", type=int, help="Number of groups")
parser.add_argument("--top_n", type=int, default=5, help="Number of top models to include")
args = parser.parse_args()
n_groups = args.n_groups
top_n = args.top_n

# ── Load data ─────────────────────────────────────────────────────────────────
comparison_pairs   = np.load("comparison_pairs_no_ties.npy")
comparison_results = np.load("comparison_results_no_ties.npy")
train_idx          = np.load("train_idx.npy")
groups_train       = np.load(f"learned_groups_u_{n_groups}_train.npy")

comparison_pairs   = comparison_pairs[train_idx]
comparison_results = comparison_results[train_idx]

with open("idx_to_model.json", "r") as f:
    idx_to_model = json.load(f)

# ── Find top_n most common models ─────────────────────────────────────────────
model_counts = pd.Series(comparison_pairs.flatten()).value_counts()
top_model_idxs = model_counts.head(top_n).index.tolist()
top_model_names = [idx_to_model[str(i)] for i in top_model_idxs]

print(f"Top {top_n} models: {top_model_names}")

# ── Build pairwise win rate per group ─────────────────────────────────────────
def compute_pairwise(pairs, results, groups, model_idxs, group_id=None):
    """
    Returns a (n x n) DataFrame where cell [i, j] = win rate of model i over model j.
    If group_id is None, uses all rows.
    """
    n = len(model_idxs)
    wins   = np.zeros((n, n), dtype=int)
    totals = np.zeros((n, n), dtype=int)

    mask = np.ones(len(pairs), dtype=bool) if group_id is None else (groups == group_id)
    pairs_g   = pairs[mask]
    results_g = results[mask]

    idx_map = {m: i for i, m in enumerate(model_idxs)}

    for (m0, m1), result in zip(pairs_g, results_g):
        if m0 not in idx_map or m1 not in idx_map:
            continue
        i, j = idx_map[m0], idx_map[m1]
        totals[i, j] += 1
        totals[j, i] += 1
        # result=0 -> m0 preferred, result=1 -> m1 preferred
        if result == 0:
            wins[i, j] += 1   # m0 beat m1
        else:
            wins[j, i] += 1   # m1 beat m0

    with np.errstate(invalid='ignore'):
        rate = np.where(totals > 0, wins / totals, np.nan)

    return rate, wins, totals


def plot_grid(ax, rate, wins, totals, model_names, title):
    n = len(model_names)
    display = np.ma.masked_where(np.eye(n, dtype=bool), rate)

    im = ax.imshow(display, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=13, color="#888")
            elif np.isnan(rate[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=11, color="#aaa")
            else:
                pct   = f"{rate[i, j]:.0%}"
                frac  = f"{wins[i,j]}/{totals[i,j]}"
                color = "white" if (rate[i, j] > 0.65 or rate[i, j] < 0.35) else "black"
                ax.text(j, i, f"{pct}\n{frac}",
                        ha="center", va="center", fontsize=11,
                        color=color, fontweight="bold")

    short_names = [m.replace("-", "-\n") if len(m) > 18 else m for m in model_names]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, fontsize=11, rotation=30, ha="right")
    ax.set_yticklabels(short_names, fontsize=11)
    ax.set_xlabel("Opponent (column model)", fontsize=12)
    ax.set_ylabel("Model (row wins over column)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)

    return im


# ── Plot: one file per subplot ────────────────────────────────────────────────
import os
output_dir = f"pairwise_winrate_k{n_groups}_top{top_n}"
os.makedirs(output_dir, exist_ok=True)

cell_size = max(top_n * 1.5, 5)

def save_grid(rate, wins, totals, model_names, title, filename):
    fig, ax = plt.subplots(figsize=(cell_size, cell_size))
    im = plot_grid(ax, rate, wins, totals, model_names, title)

    # cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
    #                     shrink=0.8, pad=0.18, aspect=30)
    # cbar.set_label("Win rate (row over column)", fontsize=11)
    # cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    # cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")

# Overall
rate, wins, totals = compute_pairwise(
    comparison_pairs, comparison_results, groups_train,
    top_model_idxs, group_id=None
)
save_grid(rate, wins, totals, top_model_names, "Overall",
          f"{output_dir}/overall.pdf")

# Per group
for g in range(n_groups):
    rate_g, wins_g, totals_g = compute_pairwise(
        comparison_pairs, comparison_results, groups_train,
        top_model_idxs, group_id=g
    )
    n_in_group = int((groups_train == g).sum())
    save_grid(rate_g, wins_g, totals_g, top_model_names,
              f"Group {g}  (n={n_in_group})",
              f"{output_dir}/group_{g}.pdf")