import sys
sys.path.append("../")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from config import HUMANITIES, STEM

# ── Load ──────────────────────────────────────────────────────────────────────
results_df        = pd.read_parquet("router_results.parquet")
results_base_df   = pd.read_parquet("router_base_results.parquet")
subj_df           = pd.read_parquet("router_results_by_subject.parquet")
subj_base_df      = pd.read_parquet("router_base_results_by_subject.parquet")

ALPHAS      = [0, 0.25, 0.5, 0.75, 1.0]
ALPHA_TICKS = [f"{int(a*100)}%" for a in ALPHAS]
COLOR_PREF  = "#4E7AD7"
COLOR_BASE  = "#D76464"
COLOR_ROUTE = "#888888"   # grey – routing baseline
FILL_ALPHA  = 0.15


# ── Helpers ───────────────────────────────────────────────────────────────────
def mean_std(df):
    return df.mean() * 100, df.std() * 100


def add_routing_baseline(ax, mean_p, x, name_low="Mistral-Medium", name_high="Claude-Opus-4"):
    """
    Draw a straight line from the accuracy at alpha=0% to alpha=100%.
    This is the naive linear routing baseline.
    """
    y_start = mean_p.iloc[0]
    y_end   = mean_p.iloc[-1]
    ax.plot([x[0], x[-1]], [y_start, y_end],
            color=COLOR_ROUTE, linestyle=":", linewidth=1.5,
            label="Linear baseline", zorder=1)

    low_name,  low_y,  low_x  = (name_low,  y_start, x[0])  if y_start <= y_end else (name_low,  y_start, x[0])
    high_name, high_y, high_x = (name_high, y_end,   x[-1]) if y_end   >= y_start else (name_high, y_end,  x[-1])

    # Label the lower endpoint (left or right depending on which is lower)
    low_name,  low_y,  low_x  = (name_low,  y_start, x[0])  if y_start <= y_end else (name_high, y_end,   x[-1])
    high_name, high_y, high_x = (name_high, y_end,   x[-1]) if y_end   >= y_start else (name_low,  y_start, x[0])

    ax.annotate(low_name,  xy=(low_x,  low_y),  xytext=(4,  -5),
                textcoords="offset points", fontsize=8, color=COLOR_ROUTE)
    ax.annotate(high_name, xy=(high_x, high_y), xytext=(-80, 0),
                textcoords="offset points", fontsize=8, color=COLOR_ROUTE)


def plot_overall(ax, results_df, results_base_df):
    mean_p, std_p = mean_std(results_df)
    mean_b, std_b = mean_std(results_base_df)
    x = np.arange(len(ALPHA_TICKS))

    add_routing_baseline(ax, mean_p, x)

    ax.plot(x, mean_p, marker="o", color=COLOR_PREF, label="Group-Aware")
    ax.fill_between(x, mean_p - std_p, mean_p + std_p, color=COLOR_PREF, alpha=FILL_ALPHA)

    ax.plot(x, mean_b, marker="s", color=COLOR_BASE, label="No-Groups", linestyle="--")
    ax.fill_between(x, mean_b - std_b, mean_b + std_b, color=COLOR_BASE, alpha=FILL_ALPHA)

    ax.set_xticks(x); ax.set_xticklabels(ALPHA_TICKS)
    ax.set_xlabel("Budget α"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Overall Routing Accuracy")
    ax.legend(); ax.grid(axis="y", linestyle=":", alpha=0.6)

def plot_single(ax, results_df):
    mean_p, std_p = mean_std(results_df)
    x = np.arange(len(ALPHA_TICKS))

    add_routing_baseline(ax, mean_p, x)

    ax.plot(x, mean_p, marker="o", color=COLOR_PREF, label="The Draw Router")
    ax.fill_between(x, mean_p - std_p, mean_p + std_p, color=COLOR_PREF, alpha=FILL_ALPHA)

    ax.set_xticks(x); ax.set_xticklabels(ALPHA_TICKS)
    ax.set_xlabel("Budget α"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Overall Routing Accuracy")
    ax.legend(); ax.grid(axis="y", linestyle=":", alpha=0.6)


def plot_grouped_bars(ax, df, df_base, title, rotate_labels=False):
    groups   = df.index.tolist()
    n_groups = len(groups)
    n_alpha  = len(ALPHA_TICKS)
    x        = np.arange(n_groups)
    width    = 0.8 / (n_alpha * 2)

    cmap_p = plt.cm.Blues(np.linspace(0.4, 0.85, n_alpha))
    cmap_b = plt.cm.Reds (np.linspace(0.4, 0.85, n_alpha))

    for i, col in enumerate(df.columns):
        offset_p = (2*i     - n_alpha + 0.5) * width
        offset_b = (2*i + 1 - n_alpha + 0.5) * width
        ax.bar(x + offset_p, df[col]      * 100, width, color=cmap_p[i],
               label=f"Pref {col}"  if i == 0 else f"Pref {col}")
        ax.bar(x + offset_b, df_base[col] * 100, width, color=cmap_b[i],
               label=f"Base {col}", hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45 if rotate_labels else 0,
                       ha="right" if rotate_labels else "center", fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.6)


def plot_grouped_lines(ax, df, df_base, title, add_baseline=True):
    cmap = plt.cm.tab10(np.linspace(0, 1, len(df.index)))
    x    = np.arange(len(ALPHA_TICKS))

    for idx, (group, color) in enumerate(zip(df.index, cmap)):
        vals = df.loc[group] * 100
        if add_baseline:
            ax.plot([x[0], x[-1]], [vals.iloc[0], vals.iloc[-1]],
                    color=color, linestyle=":", linewidth=1.0, alpha=0.5, zorder=1)
        ax.plot(x, vals,                  marker="o", color=color,
                label=str(group), linewidth=1.5)
        ax.plot(x, df_base.loc[group]*100, marker="s", color=color,
                linestyle="--", linewidth=1.0, alpha=0.7)

    ax.set_xticks(x); ax.set_xticklabels(ALPHA_TICKS)
    ax.set_xlabel("Budget α"); ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(axis="y", linestyle=":", alpha=0.6)


# ── Figure 1: Overall ─────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(6, 4))
plot_overall(ax1, results_df, results_base_df)
fig1.tight_layout()
fig1.savefig("plot_overall.pdf", dpi=150)
fig1.savefig("plot_overall.png", dpi=150)
plt.show()

# ── Figure 3: By subject ──────────────────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(15, 5))
plot_grouped_bars(axes3[0], subj_df, subj_base_df,
                  "Accuracy by Subject (bar)", rotate_labels=True)
plot_grouped_lines(axes3[1], subj_df, subj_base_df,
                   "Accuracy by Subject (line)", add_baseline=False)
fig3.suptitle("Routing Accuracy by Subject", fontsize=12, fontweight="bold")
fig3.tight_layout()
fig3.savefig("plot_by_subject.pdf", dpi=150)
fig3.savefig("plot_by_subject.png", dpi=150)
plt.show()


# ── Subject category mapping ──────────────────────────────────────────────────
def categorize_subject(subject):
    s = subject.lower().strip()
    if any(h in s for h in HUMANITIES):
        return "Humanities"
    if any(m in s for m in STEM):
        return "STEM"
    return "Other"

def aggregate_by_category(df):
    out = df.copy()
    out.index = out.index.map(categorize_subject)
    return out.groupby(level=0).mean()

cat_df      = aggregate_by_category(subj_df)
cat_base_df = aggregate_by_category(subj_base_df)

print("Categories found:", cat_df.index.tolist())

# ── Figure 5: By category ─────────────────────────────────────────────────────
categories = cat_df.index.tolist()
x          = np.arange(len(ALPHA_TICKS))
fig5, ax5 = plt.subplots(figsize=(7, 4))

cmap_cat = plt.cm.Set2(np.linspace(0, 0.6, len(categories)))
for cat, color in zip(categories, cmap_cat):
    vals = cat_df.loc[cat] * 100
    ax5.plot([x[0], x[-1]], [vals.iloc[0], vals.iloc[-1]],
             color=color, linestyle=":", linewidth=1.2, alpha=0.6, zorder=1)
    ax5.plot(x, vals,                   marker="o", color=color, linewidth=2, label=cat)
    ax5.plot(x, cat_base_df.loc[cat]*100, marker="s", color=color,
             linewidth=1.5, linestyle="--", alpha=0.7)

ax5.plot([], [], color="grey",          linewidth=2,   label="Group-Aware (solid)")
ax5.plot([], [], color="grey", ls="--", linewidth=1.5, label="No-Groups (dashed)")
ax5.plot([], [], color="grey", ls=":",  linewidth=1.2, label="Linear baseline (dotted)")
ax5.set_xticks(x); ax5.set_xticklabels(ALPHA_TICKS)
ax5.set_xlabel("Budget α"); ax5.set_ylabel("Accuracy (%)")
ax5.set_title("Accuracy by Category vs Budget α")
ax5.legend(fontsize=8); ax5.grid(axis="y", linestyle=":", alpha=0.6)

fig5.suptitle("Routing Accuracy: Humanities vs STEM", fontsize=12, fontweight="bold")
fig5.tight_layout()
fig5.savefig("plot_by_category.pdf", dpi=150)
fig5.savefig("plot_by_category.png", dpi=150)
plt.show()


# ── Figure 6: Notable subjects — where pref vs base differ most ───────────────
# Compute max absolute difference across alphas between pref and base per subject
diff = (subj_df - subj_base_df).abs().max(axis=1)
notable_subjects = diff.nlargest(4).index.tolist()
 
print(f"\nNotable subjects (largest pref vs base gap): {notable_subjects}")
 
fig6, axes6 = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
axes6 = axes6.flatten()
 
for ax, subj in zip(axes6, notable_subjects):
    vals_p = subj_df.loc[subj]      * 100
    vals_b = subj_base_df.loc[subj] * 100
 
    # Routing baseline for pref-aware
    ax.plot([x[0], x[-1]], [vals_p.iloc[0], vals_p.iloc[-1]],
            color=COLOR_ROUTE, linestyle=":", linewidth=1.2, zorder=1)
 
    ax.plot(x, vals_p, marker="o", color=COLOR_PREF, linewidth=2,
            label="Group-aware")
    ax.plot(x, vals_b, marker="s", color=COLOR_BASE, linewidth=1.5,
            linestyle="--", label="No-Groups")
 
    max_gap = (vals_p - vals_b).abs().max()
    ax.set_title(f"{subj}\n(max gap: {max_gap:.1f}pp)", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(ALPHA_TICKS, fontsize=7, rotation=30)
    ax.set_ylabel("Accuracy (%)", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)
 
# Single shared legend
handles, labels = axes6[0].get_legend_handles_labels()
fig6.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
            bbox_to_anchor=(0.5, -0.02))
 
fig6.suptitle("Notable Subjects: Largest Group-Aware vs No-Groups Gap",
              fontsize=12, fontweight="bold")
fig6.tight_layout()
fig6.savefig("plot_notable_subjects.pdf", dpi=150, bbox_inches="tight")
fig6.savefig("plot_notable_subjects.png", dpi=150, bbox_inches="tight")
plt.show()

print("All plots saved.")