import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Evaluate EM reward model accuracy.")
parser.add_argument("n_groups", type=int, default=2, help="Number of groups for the EM algorithm.")
args = parser.parse_args()
n_groups = args.n_groups

language_names = {
    'en': 'English',
    'pl': 'Polish',
    'und': 'Undetermined',
    'ru': 'Russian',
    'zh': 'Chinese (Simplified)',
    'de': 'German',
    'ko': 'Korean',
    'ja': 'Japanese',
    'fr': 'French',
    'fa': 'Persian',
    'pt': 'Portuguese',
    'es': 'Spanish',
    'it': 'Italian',
    'zh-Hant': 'Chinese (Traditional)',
    'vi': 'Vietnamese',
    'tr': 'Turkish',
    'cs': 'Czech',
    'ar': 'Arabic',
    'id': 'Indonesian',
    'uk': 'Ukrainian',
    'bn': 'Bengali',
    'hu': 'Hungarian',
    'nl': 'Dutch',
    'sv': 'Swedish',
    'th': 'Thai',
    'el': 'Greek',
    'sk': 'Slovak',
    'ro': 'Romanian',
    'da': 'Danish',
    'fi': 'Finnish',
    'sr': 'Serbian',
    'iw': 'Hebrew',
    'ms': 'Malay',
    'la': 'Latin',
    'no': 'Norwegian',
    'ca': 'Catalan',
    '<err>': 'Error',
    'lt': 'Lithuanian',
    'bg': 'Bulgarian',
    'hi': 'Hindi',
    'nn': 'Norwegian Nynorsk',
    'gl': 'Galician',
    'ml': 'Malayalam',
    'rw': 'Kinyarwanda',
    'sl': 'Slovenian',
    'lv': 'Latvian',
    'hr': 'Croatian',
    'my': 'Burmese',
    'sco': 'Scots',
    'et': 'Estonian',
}

# users of each prompt in train split
language_users = np.load("language_users.npy")
train_idx = np.load('train_idx.npy')
language_users_train = language_users[train_idx]

# learned group of each user in train split
groups_train = np.load(f"learned_groups_u_{n_groups}_train.npy")

# Build a DataFrame for easier grouping
df_lang = pd.DataFrame({'language': language_users_train, 'group': groups_train})
df_lang['language_full'] = df_lang['language'].map(lambda x: language_names.get(x, x))

# Compute percentage of each language within each group
group_lang_pct = (df_lang.groupby(['group', 'language_full'])
                          .size()
                          .groupby(level=0, group_keys=False)
                          .apply(lambda x: x / x.sum() * 100)
                          .unstack(fill_value=0))

# Only keep top N languages for readability
top_n = 5
top_languages = (df_lang['language_full'].value_counts().head(top_n).index.tolist())
# group_lang_pct = group_lang_pct[top_languages]

# Plot
n_groups_plot = len(group_lang_pct)
n_cols = min(4, n_groups_plot)
n_rows = int(np.ceil(n_groups_plot / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
axes = np.array(axes).flatten()  # works for both 1D and 2D

# Hide unused axes
for ax in axes[n_groups_plot:]:
    ax.set_visible(False)

colors = plt.cm.tab10.colors

for ax, (group_id, row) in zip(axes, group_lang_pct.iterrows()):
    # Only keep languages present in this group, sorted by percentage
    row = row[row > 0].sort_values(ascending=False).head(top_n)
    
    bars = ax.bar(range(len(row)), row.values, color=colors[:len(row)], edgecolor='white')
    ax.set_xticks(range(len(row)))
    ax.set_xticklabels(row.index, rotation=45, ha='right', fontsize=9)
    group_size = (groups_train == group_id).sum()
    ax.set_title(f"Group {group_id} (n={group_size})", fontsize=11, fontweight='bold')
    ax.set_ylabel("% of group" if group_id == 0 else "")
    ax.set_ylim(0, 100)
    ax.spines[['top', 'right']].set_visible(False)

    for bar, val in zip(bars, row.values):
        if val > 1.5:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha='center', va='bottom', fontsize=7)

plt.suptitle(f"Language breakdown by group (k={n_groups}, top {top_n} languages)",
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"language_breakdown_k{n_groups}.pdf", bbox_inches='tight')
print(f"Saved to language_breakdown_k{n_groups}.pdf")