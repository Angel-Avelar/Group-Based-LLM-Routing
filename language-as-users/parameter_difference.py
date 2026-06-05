import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
from itertools import combinations
import argparse

parser = argparse.ArgumentParser(description="Plot Punnett squares for learned groups.")
parser.add_argument("n_groups", type=int, nargs="+", help="List of group counts, e.g. 1 2 4 8")
args = parser.parse_args()

def plot_pairwise_cosine_similarities(phi, feature_size=385, bins=7):
    n_groups = phi.shape[1]
    n_models = phi.shape[0] // feature_size
    groups = [phi[:, g].reshape((n_models, feature_size)) for g in range(n_groups)]

    pairs = list(combinations(range(n_groups), 2))

    sims = {}
    for i, j in pairs:
        norms_i = np.linalg.norm(groups[i], axis=1)
        norms_j = np.linalg.norm(groups[j], axis=1)
        dot = np.einsum('ij,ij->i', groups[i], groups[j])
        sims[(i, j)] = dot / (norms_i * norms_j)

    if n_groups == 2:
        fig, axes = plt.subplots(1, 1, figsize=(4, 3))
        axes = np.array([[axes]])
    else:
        fig, axes = plt.subplots(
            n_groups - 1, n_groups - 1,
            figsize=(3 * (n_groups - 1), 2.5 * (n_groups - 1)),
            sharex=True, sharey=False
        )

    for ax in axes.flat:
        ax.set_visible(False)

    for i, j in pairs:
        row, col = j - 1, i
        ax = axes[row, col]
        ax.set_visible(True)

        data = sims[(i, j)]
        ax.hist(data, bins=bins, color='#378ADD', edgecolor='white', linewidth=0.5)
        ax.set_title(f'group {i} vs {j}', fontsize=11, fontweight='500')
        ax.set_xlabel('cosine similarity', fontsize=9)
        ax.axvline(np.mean(data), color='#D85A30', linewidth=1.2, linestyle='--',
                   label=f'mean {np.mean(data):.2f}')
        ax.legend(fontsize=8, frameon=False)
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(f'Pairwise cosine similarities — {n_groups} groups', fontsize=13, fontweight='500', y=1.02)
    plt.tight_layout()
    return fig


phis = {n_group: np.load(f"phi_learned_u_{n_group}_train.npy") for n_group in args.n_groups}

with pdf_backend.PdfPages("cosine_similarities.pdf") as pdf:
    for n_groups, phi in phis.items():
        fig = plot_pairwise_cosine_similarities(phi)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print("Saved cosine_similarities.pdf")