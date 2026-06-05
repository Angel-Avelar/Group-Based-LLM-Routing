import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
from itertools import combinations
from util import get_data
import argparse

parser = argparse.ArgumentParser(description="Plot Punnett squares for learned groups.")
parser.add_argument("n_groups", type=int, nargs="+", help="List of group counts, e.g. 1 2 4 8")
args = parser.parse_args()

split = get_data()

rows = [
    {
        'creative_writing': comparison['category_tag']['creative_writing_v0.1']['creative_writing'],
        **{k: comparison['category_tag']['criteria_v0.1'][k] for k in 
           ['complexity', 'creativity', 'domain_knowledge', 'problem_solving', 
            'real_world', 'specificity', 'technical_accuracy']},
        'math': comparison['category_tag']['math_v0.1']['math'],
        'instruction_following': comparison['category_tag']['if_v0.1']['if'],
        'is_code': comparison['is_code'],
        'english': comparison['language'] == 'en'
    }
    for comparison in split
]

train_idx = np.load("train_idx.npy")
rows = [rows[i] for i in train_idx]

df = pd.DataFrame(rows)
prompt_tags = df.columns.tolist()
# prompt_tags = ['is_code', 'instruction_following', 'math', 'english']

for n in args.n_groups:
    df[f'groups_{n}'] = np.load(f"learned_groups_u_{n}_train.npy")

groupings = [f"groups_{n}" for n in args.n_groups]
print("loaded data")
print("Creating Punnett Squares")

def punnett_distributions(df: pd.DataFrame, cat_x: str, cat_y: str, grouping: str) -> pd.DataFrame:
    """
    Compute the joint (cat_x, cat_y) distribution for each group.
    Returns a long-form DataFrame with columns:
      group, cat_x_val, cat_y_val, count, pct
    """
    combos = pd.MultiIndex.from_tuples(
        [(False, False), (False, True), (True, False), (True, True)],
        names=[cat_x, cat_y]
    )

    records = []
    for group_id, gdf in df.groupby(grouping):
        counts = (
            gdf.groupby([cat_x, cat_y], observed=True)
               .size()
               .reindex(combos, fill_value=0)
               .reset_index(name="count")
        )
        counts["pct"] = counts["count"] / counts["count"].sum()
        counts[grouping] = group_id
        records.append(counts)

    return pd.concat(records, ignore_index=True)

def plot_punnett_pair(dist, cat_x, cat_y, grouping, axes_row):
    """
    Plot one row of punnett squares for a given (cat_x, cat_y, grouping) combo.
    axes_row: array of axes, one per group.
    """
    groups = sorted(dist[grouping].unique())
    for ax, g in zip(axes_row, groups):
        gdf = dist[dist[grouping] == g]
        total = gdf["count"].sum()

        # Build 2x2 matrix: rows = cat_y (T/F), cols = cat_x (T/F)
        pct = {}
        cnt = {}
        for _, row in gdf.iterrows():
            key = (bool(row[cat_x]), bool(row[cat_y]))
            pct[key] = row["pct"]
            cnt[key] = int(row["count"])

        matrix = np.array([
            [pct.get((True,  True),  0), pct.get((False,  True), 0)],
            [pct.get((True,  False), 0), pct.get((False, False),  0)],
        ])
        counts = np.array([
            [cnt.get((True,  True),  0), cnt.get((False,  True), 0)],
            [cnt.get((True,  False), 0), cnt.get((False, False),  0)],
        ])

        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")

        for i in range(2):
            for j in range(2):
                v = matrix[i, j]
                ax.text(j, i,
                    f"{v:.1%}\n(n={counts[i,j]})",
                    ha="center", va="center", fontsize=8,
                    color="white" if v > 0.4 else "black"
                )

        ax.set_xticks([0, 1]); ax.set_xticklabels(["True", "False"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True", "False"], fontsize=8)
        ax.set_xlabel(cat_x, fontsize=8)
        ax.set_ylabel(cat_y, fontsize=8)
        ax.set_title(f"group {g}  (n={total})", fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    # Hide unused axes if n_groups < len(axes_row)
    for ax in axes_row[len(groups):]:
        ax.set_visible(False)


def visualize_all_punnett(df, prompt_tags_x, prompt_tags_y, groupings, output_pdf="punnett_squares.pdf"):
    """
    For each (cat_x, cat_y) pair, produce one figure with one row per grouping.
    All figures saved to a single PDF.
    """
    pairs = [(x, y) for x in prompt_tags_x for y in prompt_tags_y if x != y]
    max_groups = max(int(g.split("_")[1]) for g in groupings)

    with pdf_backend.PdfPages(output_pdf) as pdf:
        for cat_x, cat_y in pairs:
            # print(f"Calculating {cat_x} vs {cat_y} ...")
            n_rows = len(groupings)
            n_cols = max_groups

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(3 * n_cols, 3 * n_rows),
                squeeze=False
            )

            fig.suptitle(f"{cat_x}  ×  {cat_y}", fontsize=13, fontweight="bold", y=1.01)

            for row_idx, grouping in enumerate(groupings):
                n_groups = int(grouping.split("_")[1])
                dist = punnett_distributions(df, cat_x, cat_y, grouping)
                dist["cell"] = (
                    dist[cat_x].map({False: "F", True: "T"}) +
                    dist[cat_y].map({False: "F", True: "T"})
                )

                # Row label on the leftmost axis
                axes[row_idx, 0].set_ylabel(
                    f"{grouping}\n{cat_y}", fontsize=9, fontweight="bold"
                )

                plot_punnett_pair(dist, cat_x, cat_y, grouping, axes[row_idx])

                # Hide unused columns for smaller groupings
                for col_idx in range(n_groups, n_cols):
                    axes[row_idx, col_idx].set_visible(False)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved {len(pairs)} figures to {output_pdf}")

# visualize_all_punnett(df, prompt_tags, prompt_tags, groupings, output_pdf="punnett_squares.pdf")
visualize_all_punnett(df, ['domain_knowledge', 'problem_solving', 'real_world', 'specificity'], ['english'], groupings, output_pdf="punnett_squares_english.pdf")
