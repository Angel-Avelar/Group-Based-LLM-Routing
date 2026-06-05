import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
from util import get_data, load_category_tags
from scipy.stats import fisher_exact
import argparse
import json

parser = argparse.ArgumentParser(description="Plot bar charts for a single model's overall win rate.")
parser.add_argument("n_groups", type=int, nargs="+", help="List of group counts, e.g. 1 2 4 8")
parser.add_argument("--model", type=str, default="gemini-2.5-pro", help="Model to evaluate")
parser.add_argument("--alpha", type=float, default=1.1, help="Significance threshold")
args = parser.parse_args()

n_groups_str = "_".join(map(str, args.n_groups))
output_pdf = f"./punnet_squares/bar_groups_{n_groups_str}_model_{args.model}_significant.pdf"
output_tex = f"./punnet_squares/table_groups_{n_groups_str}_model_{args.model}.tex"

split = get_data()
df = load_category_tags(split)

train_idx = np.load("train_idx.npy")
df = df.iloc[train_idx]
prompt_tags = df.columns.tolist()

for n in args.n_groups:
    df[f'groups_{n}'] = np.load(f"learned_groups_u_{n}_train.npy")

groupings = [f"groups_{n}" for n in args.n_groups]

# Load all comparisons
comparison_pairs = np.load("comparison_pairs_no_ties.npy")
comparison_pairs = comparison_pairs[train_idx]
comparison_results = np.load("comparison_results_no_ties.npy")
comparison_results = comparison_results[train_idx]

with open('model_to_idx.json', 'r') as file:
    model_to_idx = json.load(file)

model_idx = model_to_idx[args.model]

# Find all comparisons where the model appears (either side)
mask_as_0 = comparison_pairs[:, 0] == model_idx
mask_as_1 = comparison_pairs[:, 1] == model_idx
pair_mask = mask_as_0 | mask_as_1

print(f"Total comparisons for {args.model}: {sum(pair_mask)}")

df = df[pair_mask].copy()

model_preferred = np.where(
    mask_as_0[pair_mask],
    comparison_results[pair_mask] == 0,
    comparison_results[pair_mask] == 1
)
df['model_preferred'] = model_preferred

print("Loaded data")
print("Creating bar charts...")

COLOR_WIN  = "#4994C5"
COLOR_LOSS = "#E4847B"


def category_distributions(df: pd.DataFrame, cat: str, grouping: str) -> pd.DataFrame:
    records = []
    for group_id, gdf in df.groupby(grouping):
        for cat_val in [True, False]:
            subset = gdf[gdf[cat] == cat_val]
            total = len(subset)
            wins = subset['model_preferred'].sum()
            losses = total - wins
            rate_win = wins / total if total > 0 else 0.0
            records.append({
                grouping: group_id,
                cat: cat_val,
                'total': total,
                'wins': int(wins),
                'losses': int(losses),
                'rate_win': rate_win,
                'rate_loss': 1.0 - rate_win,
            })
    return pd.DataFrame(records)


def is_significant(dist, cat, grouping, group_id, alpha=0.05):
    """
    Test if win rate differs significantly between cat=True and cat=False for a given group.
    Uses Fisher's exact test.
    """
    gdf = dist[dist[grouping] == group_id]
    true_rows  = gdf[gdf[cat] == True]
    false_rows = gdf[gdf[cat] == False]

    if true_rows.empty or false_rows.empty:
        return False, 1.0

    row_true  = true_rows.iloc[0]
    row_false = false_rows.iloc[0]

    table = [
        [row_true['wins'],  row_true['losses']],
        [row_false['wins'], row_false['losses']],
    ]
    _, p_val = fisher_exact(table)
    return p_val < alpha, p_val


def plot_category_bars(df, prompt_tags, groupings, output_pdf, model_name, alpha=0.05):
    n_rows = len(groupings)

    with pdf_backend.PdfPages(output_pdf) as pdf:
        for cat in prompt_tags:
            # Pre-compute sig_groups per grouping so we can size the figure correctly
            all_dists = {}
            all_sig_groups = {}
            for grouping in groupings:
                dist = category_distributions(df, cat, grouping)
                all_groups = sorted(dist[grouping].unique())
                sig = [(g, p_val)
                       for g in all_groups
                       for (sig_flag, p_val) in [is_significant(dist, cat, grouping, g, alpha)]
                       if sig_flag]
                all_dists[grouping] = (dist, all_groups)
                all_sig_groups[grouping] = sig

            # Figure width based on max significant groups across all groupings
            max_sig = max(max(len(s), 1) for s in all_sig_groups.values())
            fig, axes = plt.subplots(
                n_rows, 1,
                figsize=(max(4, max_sig * 2.0), 4 * n_rows),
                squeeze=False
            )

            fig.suptitle(
                f"{cat} | {model_name} overall win rate",
                fontsize=13, fontweight="bold", y=1.01
            )

            for row_idx, grouping in enumerate(groupings):
                ax = axes[row_idx, 0]
                n_groups = int(grouping.split("_")[1])
                dist, all_groups = all_dists[grouping]
                sig_groups = all_sig_groups[grouping]

                if not sig_groups:
                    ax.text(0.5, 0.5, "No significant differences found",
                            ha='center', va='center', transform=ax.transAxes,
                            fontsize=10, color='gray')
                    ax.set_title(f"{grouping}  ({n_groups} groups)", fontsize=10, fontweight="bold")
                    ax.axis('off')
                    continue

                bar_width = 0.35
                group_spacing = 1.0
                x_ticks = []
                x_labels = []

                for g_idx, (g, p_val) in enumerate(sig_groups):
                    gdf = dist[dist[grouping] == g]
                    base_x = g_idx * group_spacing

                    ax.text(base_x + bar_width / 2, 1.10, f"p={p_val:.3f}",
                            ha='center', va='bottom', fontsize=7,
                            color='#222', style='italic')

                    for b_idx, cat_val in enumerate([False, True]):
                        row = gdf[gdf[cat] == cat_val]
                        if row.empty:
                            continue
                        row = row.iloc[0]

                        x = base_x + b_idx * bar_width
                        rate_win  = row['rate_win']
                        rate_loss = row['rate_loss']
                        wins   = row['wins']
                        losses = row['losses']
                        total  = row['total']

                        ax.bar(x, rate_win,  width=bar_width, color=COLOR_WIN,  edgecolor='white', linewidth=0.5)
                        ax.bar(x, rate_loss, width=bar_width, bottom=rate_win,  color=COLOR_LOSS, edgecolor='white', linewidth=0.5)

                        if rate_win > 0.12:
                            ax.text(x, rate_win / 2,
                                    f"{rate_win:.0%}\n({wins})",
                                    ha='center', va='center', fontsize=7,
                                    color='white', fontweight='bold')
                        if rate_loss > 0.12:
                            ax.text(x, rate_win + rate_loss / 2,
                                    f"{rate_loss:.0%}\n({losses})",
                                    ha='center', va='center', fontsize=7,
                                    color='white', fontweight='bold')

                        ax.text(x, 1.02, f"n={total}",
                                ha='center', va='bottom', fontsize=7, color='#444')

                        x_ticks.append(x)
                        x_labels.append(f"g{g}\n{cat_val}")

                ax.set_xlim(-bar_width, len(sig_groups) * group_spacing - bar_width * 0.5)
                ax.set_ylim(0, 1.15)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_labels, fontsize=8)
                ax.set_ylabel("Win rate", fontsize=9)
                ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
                ax.spines[["top", "right"]].set_visible(False)
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=COLOR_WIN,  label="Model wins"),
                Patch(facecolor=COLOR_LOSS, label="Model loses"),
            ]
            fig.legend(handles=legend_elements, loc='lower right', fontsize=7,
                       bbox_to_anchor=(0.67, 0.87), bbox_transform=fig.transFigure, framealpha=0.9)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        print(f"Saved {len(prompt_tags)} figures to {output_pdf}")


def build_latex_table(df, prompt_tags, groupings, output_tex, model_name, alpha=0.05):
    """
    Build a LaTeX table:
      rows    = categories
      columns = (grouping, group_id, cat_val) triples
      cells   = win% (wins/total), bolded + * if the group is significant
    """

    # Gather all column keys: (grouping, group_id, cat_val)
    col_keys = []
    for grouping in groupings:
        n_g = int(grouping.split("_")[1])
        for g in range(n_g):
            for cat_val in [False, True]:
                col_keys.append((grouping, g, cat_val))

    # ── Build cell data ────────────────────────────────────────────────────────
    table_rows = []
    for cat in prompt_tags:
        row = {'Category': cat.replace('_', ' ').title()}

        dists = {}
        sig_map = {}
        for grouping in groupings:
            dist = category_distributions(df, cat, grouping)
            dists[grouping] = dist
            for g in sorted(dist[grouping].unique()):
                sig_flag, p_val = is_significant(dist, cat, grouping, g, alpha)
                sig_map[(grouping, g)] = (sig_flag, p_val)

        for (grouping, g, cat_val) in col_keys:
            dist = dists[grouping]
            gdf  = dist[dist[grouping] == g]
            sub  = gdf[gdf[cat] == cat_val]
            if sub.empty:
                cell = "--"
            else:
                r    = sub.iloc[0]
                pct  = f"{r['rate_win']:.0%}".replace("%", "\\%")
                cell = f"{pct} ({r['wins']}/{r['total']})"
                sig_flag, p_val = sig_map.get((grouping, g), (False, 1.0))
                if sig_flag:
                    # cell = f"\\textbf{{{cell}}}$^{{p={p_val:.3f}}}$"
                    cell = f"\\textbf{{{cell}}}"
            row[(grouping, g, cat_val)] = cell

        table_rows.append(row)

    # ── Compute header spans ───────────────────────────────────────────────────
    # Level 1: grouping
    grouping_spans = []
    cur, span = None, 0
    for (grouping, g, cv) in col_keys:
        if grouping != cur:
            if cur is not None:
                grouping_spans.append((cur, span))
            cur, span = grouping, 1
        else:
            span += 1
    if cur:
        grouping_spans.append((cur, span))

    # Level 2: (grouping, group_id)
    group_spans = []
    cur, span = None, 0
    for (grouping, g, cv) in col_keys:
        key = (grouping, g)
        if key != cur:
            if cur is not None:
                group_spans.append((cur, span))
            cur, span = key, 1
        else:
            span += 1
    if cur:
        group_spans.append((cur, span))

    # ── Write LaTeX ────────────────────────────────────────────────────────────
    n_data_cols = len(col_keys)
    col_fmt = "l" + "c" * n_data_cols

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.2}")
    safe_model = model_name.replace('_', r'\_').replace('-', r'-')
    lines.append(
        f"\\caption{{Win rate of {safe_model} by category and group "
        f"(\\textbf{{bold}} = significant at $\\alpha={alpha}$)}}"
    )
    lines.append(f"\\label{{tab:winrate_{n_groups_str}}}")
    lines.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    lines.append(r"\toprule")

    # Row 1: grouping-level headers
    h1 = [""]
    for (grouping, span) in grouping_spans:
        k = grouping.split("_")[1]
        h1.append(f"\\multicolumn{{{span}}}{{c}}{{$k={k}$ groups}}")
    lines.append(" & ".join(h1) + r" \\")

    # Cmidrules under grouping headers
    cmidrules = []
    offset = 2
    for (_, span) in grouping_spans:
        cmidrules.append(f"\\cmidrule(lr){{{offset}-{offset+span-1}}}")
        offset += span
    lines.append("".join(cmidrules))

    # Row 2: group-id headers
    h2 = [""]
    for ((grouping, g), span) in group_spans:
        label = f"group {g}"
        if span > 1:
            h2.append(f"\\multicolumn{{{span}}}{{c}}{{{label}}}")
        else:
            h2.append(label)
    lines.append(" & ".join(h2) + r" \\")

    # Row 3: False / True headers
    h3 = ["Category"]
    for (_, _, cat_val) in col_keys:
        h3.append("T" if cat_val else "F")
    lines.append(" & ".join(h3) + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for row in table_rows:
        cells = [row["Category"]]
        for key in col_keys:
            cells.append(row.get(key, "--"))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)
    with open(output_tex, "w") as f:
        f.write(latex_str)
    print(f"Saved LaTeX table to {output_tex}")
    return latex_str


plot_category_bars(df, prompt_tags, groupings, output_pdf, args.model, alpha=args.alpha)
build_latex_table(df, prompt_tags, groupings, output_tex, args.model, alpha=args.alpha)