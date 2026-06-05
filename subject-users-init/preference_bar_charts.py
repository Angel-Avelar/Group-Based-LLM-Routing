import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
from util import get_data, load_category_tags
import argparse
import json

parser = argparse.ArgumentParser(description="Plot bar charts for learned groups.")
parser.add_argument("n_groups", type=int, nargs="+", help="List of group counts, e.g. 1 2 4 8")
parser.add_argument("--model_a", type=str, default="gemini-2.5-pro")
parser.add_argument("--model_b", type=str, default="gemini-2.5-flash")
args = parser.parse_args()

n_groups_str = "_".join(map(str, args.n_groups))
output_pdf = f"./punnet_squares/bar_groups_{n_groups_str}_models_{args.model_a}_vs_{args.model_b}.pdf"

split = get_data()
df = load_category_tags(split)

train_idx = np.load("train_idx.npy")
df = df.iloc[train_idx]
prompt_tags = df.columns.tolist()

for n in args.n_groups:
    df[f'groups_{n}'] = np.load(f"learned_groups_u_{n}_train.npy")

groupings = [f"groups_{n}" for n in args.n_groups]

# Filter for model pair
comparison_pairs = np.load("comparison_pairs_no_ties.npy")
comparison_pairs = comparison_pairs[train_idx]
comparison_results = np.load("comparison_results_no_ties.npy")
comparison_results = comparison_results[train_idx]

with open('model_to_idx.json', 'r') as file:
    model_to_idx = json.load(file)

model_a_idx = model_to_idx[args.model_a]
model_b_idx = model_to_idx[args.model_b]

mask_ab = ((comparison_pairs[:, 0] == model_a_idx) & (comparison_pairs[:, 1] == model_b_idx))
mask_ba = ((comparison_pairs[:, 0] == model_b_idx) & (comparison_pairs[:, 1] == model_a_idx))
pair_mask = mask_ab | mask_ba

print(f"Total comparisons for pair: {sum(pair_mask)}")

df = df[pair_mask].copy()

model_a_preferred = np.where(
    mask_ab[pair_mask],
    comparison_results[pair_mask] == 0,
    comparison_results[pair_mask] == 1
)
df['model_a_preferred'] = model_a_preferred

print("Loaded data")
print("Creating bar charts...")

COLOR_A = "#2471A3"   # blue  - model_a wins
COLOR_B = "#E4977B"   # orange - model_b wins

def category_distributions(df: pd.DataFrame, cat: str, grouping: str) -> pd.DataFrame:
    """
    For each group and each value of cat (True/False), compute:
      - total comparisons
      - model_a wins
      - model_b wins
      - win rate for model_a
    """
    records = []
    for group_id, gdf in df.groupby(grouping):
        for cat_val in [True, False]:
            subset = gdf[gdf[cat] == cat_val]
            total = len(subset)
            wins_a = subset['model_a_preferred'].sum()
            wins_b = total - wins_a
            rate_a = wins_a / total if total > 0 else 0.0
            records.append({
                grouping: group_id,
                cat: cat_val,
                'total': total,
                'wins_a': int(wins_a),
                'wins_b': int(wins_b),
                'rate_a': rate_a,
                'rate_b': 1.0 - rate_a,
            })
    return pd.DataFrame(records)


def plot_category_bars(df, prompt_tags, groupings, output_pdf, model_a_name, model_b_name):
    """
    One PDF page per category.
    Each page has one row per grouping setting, one pair of bars (True/False) per group.
    Bars are stacked: model_a wins (blue) + model_b wins (orange) = 100%.
    """
    n_rows = len(groupings)

    with pdf_backend.PdfPages(output_pdf) as pdf:
        for cat in prompt_tags:
            max_groups = max(int(g.split("_")[1]) for g in groupings)

            fig, axes = plt.subplots(
                n_rows, 1,
                figsize=(max(6, max_groups * 2.5), 4 * n_rows),
                squeeze=False
            )

            fig.suptitle(
                f"Category: {cat}",
                fontsize=13, fontweight="bold", y=1.01
            )

            for row_idx, grouping in enumerate(groupings):
                ax = axes[row_idx, 0]
                n_groups = int(grouping.split("_")[1])
                dist = category_distributions(df, cat, grouping)

                groups = sorted(dist[grouping].unique())
                # Two bars per group: cat=False, cat=True
                # Layout: [group0_False, group0_True, gap, group1_False, group1_True, ...]
                bar_width = 0.35
                group_spacing = 1.0
                x_ticks = []
                x_labels = []

                for g_idx, g in enumerate(groups):
                    gdf = dist[dist[grouping] == g]
                    base_x = g_idx * group_spacing

                    for b_idx, cat_val in enumerate([False, True]):
                        row = gdf[gdf[cat] == cat_val]
                        if row.empty:
                            continue
                        row = row.iloc[0]

                        x = base_x + b_idx * bar_width
                        rate_a = row['rate_a']
                        rate_b = row['rate_b']
                        wins_a = row['wins_a']
                        wins_b = row['wins_b']
                        total = row['total']

                        # Stacked bar
                        ax.bar(x, rate_a, width=bar_width, color=COLOR_A, edgecolor='white', linewidth=0.5)
                        ax.bar(x, rate_b, width=bar_width, bottom=rate_a, color=COLOR_B, edgecolor='white', linewidth=0.5)

                        # Annotations inside bars (only if bar segment is tall enough)
                        if rate_a > 0.12:
                            ax.text(x, rate_a / 2,
                                    f"{rate_a:.0%}\n({wins_a})",
                                    ha='center', va='center', fontsize=7,
                                    color='white', fontweight='bold')
                        if rate_b > 0.12:
                            ax.text(x, rate_a + rate_b / 2,
                                    f"{rate_b:.0%}\n({wins_b})",
                                    ha='center', va='center', fontsize=7,
                                    color='white', fontweight='bold')

                        # n label above bar
                        ax.text(x, 1.02, f"n={total}",
                                ha='center', va='bottom', fontsize=7, color='#444')

                        x_ticks.append(x)
                        x_labels.append(f"g{g}\n{cat_val}")

                ax.set_xlim(-bar_width, len(groups) * group_spacing - bar_width * 0.5)
                ax.set_ylim(0, 1.15)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_labels, fontsize=8)
                ax.set_ylabel("Win rate", fontsize=9)
                ax.set_title(f"{grouping}  ({n_groups} groups)", fontsize=10, fontweight="bold")
                ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
                ax.spines[["top", "right"]].set_visible(False)
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)

            # Legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=COLOR_A, label=f"{model_a_name} preferred"),
                Patch(facecolor=COLOR_B, label=f"{model_b_name} preferred"),
            ]
            fig.legend(handles=legend_elements, loc='upper right', fontsize=9,
                       bbox_to_anchor=(1.0, 1.0), framealpha=0.9)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        print(f"Saved {len(prompt_tags)} figures to {output_pdf}")


plot_category_bars(df, prompt_tags, groupings, output_pdf, args.model_a, args.model_b)