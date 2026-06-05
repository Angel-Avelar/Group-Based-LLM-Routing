import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import argparse
from config import bool_cols

parser = argparse.ArgumentParser(description="Analyze category user group assignments.")
parser.add_argument("n_groups", type=int, default=2, help="Number of groups for the EM algorithm.")
args = parser.parse_args()
n_groups = args.n_groups

n_features = len(bool_cols)

# ── Load data ─────────────────────────────────────────────────────────────────
category_users = np.load("category_users.npy")
train_idx      = np.load('train_idx.npy')
category_users_train = category_users[train_idx]
groups_train         = np.load(f"learned_groups_u_{n_groups}_train.npy")

# ── Decode user integers into boolean feature vectors ─────────────────────────
# user integer is binary encoding: bit (n_features-1) = first col, bit 0 = last col
all_user_ids = np.arange(2 ** n_features)  # 0..63
def decode_user(user_id):
    return [(user_id >> (n_features - 1 - i)) & 1 for i in range(n_features)]

user_features = pd.DataFrame(
    [decode_user(u) for u in all_user_ids],
    columns=bool_cols,
    index=all_user_ids
).astype(bool)

# ── Map each user to its group (from training data) ───────────────────────────
user_to_group = {}
user_to_count = {}
for user_id in all_user_ids:
    mask = category_users_train == user_id
    count = mask.sum()
    user_to_count[user_id] = int(count)
    if count > 0:
        # majority vote in case of inconsistency (should be deterministic)
        assigned_groups = groups_train[mask]
        user_to_group[user_id] = int(pd.Series(assigned_groups).mode()[0])
    else:
        user_to_group[user_id] = -1  # unseen user

user_features['group']  = user_features.index.map(user_to_group)
user_features['count']  = user_features.index.map(user_to_count)
user_features['seen']   = user_features['count'] > 0

# ── Figure 1: Heatmap — user feature combinations coloured by group ───────────
# Sort users: first by group, then by feature values for readability
seen = user_features[user_features['seen']].copy()
seen_sorted = seen.sort_values(['group'] + bool_cols).reset_index(drop=False)
seen_sorted = seen_sorted.rename(columns={'index': 'user_id'})

group_colors = plt.cm.Set2.colors
cmap_groups  = {g: group_colors[g % len(group_colors)] for g in range(n_groups)}

fig, ax = plt.subplots(figsize=(max(8, n_features * 1.2), max(6, len(seen_sorted) * 0.35)))

# Draw heatmap cells for boolean features
feature_matrix = seen_sorted[bool_cols].values.astype(float)
ax.imshow(feature_matrix, cmap='Blues', vmin=0, vmax=1,
          aspect='auto', alpha=0.7)

# Overlay group colour as left margin strip
for row_idx, (_, row) in enumerate(seen_sorted.iterrows()):
    g = int(row['group'])
    ax.add_patch(mpatches.Rectangle(
        (-0.9, row_idx - 0.5), 0.8, 1.0,
        color=cmap_groups[g], clip_on=False
    ))
    # Count label on the right
    ax.text(n_features - 0.3, row_idx, f"n={int(row['count'])}",
            va='center', ha='left', fontsize=7, color='#444')

# Cell text: T / F
for r in range(len(seen_sorted)):
    for c in range(n_features):
        val = feature_matrix[r, c]
        ax.text(c, r, 'T' if val else 'F',
                ha='center', va='center', fontsize=7,
                color='white' if val else '#aaa', fontweight='bold' if val else 'normal')

ax.set_xticks(range(n_features))
ax.set_xticklabels(bool_cols, rotation=40, ha='right', fontsize=9)
ax.set_yticks(range(len(seen_sorted)))
ax.set_yticklabels(
    [f"user {int(row['user_id'])}" for _, row in seen_sorted.iterrows()],
    fontsize=7
)
ax.set_xlim(-1, n_features + 1.5)

# Legend for groups
legend_patches = [mpatches.Patch(color=cmap_groups[g], label=f'Group {g}')
                  for g in range(n_groups)]
ax.legend(handles=legend_patches, loc='upper right',
          bbox_to_anchor=(1.18, 1.0), fontsize=9, title='Group')

ax.set_title(f"Category user group assignment (k={n_groups})\n"
             f"Sorted by group then features. Colour strip = assigned group.",
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f"category_user_heatmap_k{n_groups}.pdf", bbox_inches='tight')
print(f"Saved category_user_heatmap_k{n_groups}.pdf")
plt.close()

# ── Figure 2: For each group, show which features are True more often ─────────
fig2, axes2 = plt.subplots(1, n_groups, figsize=(4 * n_groups, 4), sharey=True)
if n_groups == 1:
    axes2 = [axes2]

for g, ax2 in enumerate(axes2):
    g_users = seen_sorted[seen_sorted['group'] == g]
    # Weighted by count: how often is each feature True across all prompts in this group
    weighted_true = (g_users[bool_cols].values * g_users['count'].values[:, None]).sum(axis=0)
    weighted_total = g_users['count'].sum()
    feature_pct = weighted_true / weighted_total * 100 if weighted_total > 0 else np.zeros(n_features)

    bars = ax2.barh(bool_cols, feature_pct, color=cmap_groups[g], edgecolor='white')
    ax2.set_xlim(0, 100)
    ax2.set_title(f"Group {g}\n(n={int(weighted_total)} prompts, {len(g_users)} users)",
                  fontsize=10, fontweight='bold', color=cmap_groups[g])
    ax2.set_xlabel("% of prompts with feature = True", fontsize=8)
    ax2.spines[['top', 'right']].set_visible(False)
    for bar, val in zip(bars, feature_pct):
        ax2.text(val + 1, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va='center', fontsize=8)

plt.suptitle(f"Feature prevalence per group (k={n_groups})\n"
             f"Weighted by number of prompts per user",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"category_feature_prevalence_k{n_groups}.pdf", bbox_inches='tight')
print(f"Saved category_feature_prevalence_k{n_groups}.pdf")
plt.close()

# ── Print summary table ───────────────────────────────────────────────────────
print(f"\n── User-to-group assignments (k={n_groups}) ──")
print(seen_sorted[['user_id'] + bool_cols + ['group', 'count']].to_string(index=False))

# Check if math=True users are all in same group
math_users = seen_sorted[seen_sorted['math'] == True]
math_groups = math_users['group'].unique()
print(f"\nmath=True users are in groups: {sorted(math_groups)}")
if len(math_groups) == 1:
    print(f"  All math=True users are in the same group (group {math_groups[0]})")
else:
    print(f"  math=True users are split across {len(math_groups)} groups")
    for g in sorted(math_groups):
        u_in_g = math_users[math_users['group'] == g]['user_id'].tolist()
        print(f"    Group {g}: user_ids {u_in_g}")