import numpy as np
import pandas as pd
import json
import argparse
from sklearn.model_selection import train_test_split
from util import rewardEM

parser = argparse.ArgumentParser(description="Train EM reward model with a given number of groups.")
parser.add_argument("n_groups", type=int, help="Number of groups for the EM algorithm.")
args = parser.parse_args()
n_groups = args.n_groups

embeddings = np.load("prompt_embeddings_no_ties.npy")
comparison_pairs = np.load("comparison_pairs_no_ties.npy")
comparison_results = np.load("comparison_results_no_ties.npy")
users = np.load("language_users.npy")

with open('idx_to_model.json', 'r') as file:
    idx_to_model = json.load(file)
n_items = len(idx_to_model)

n_comparisons = embeddings.shape[0]
indices = np.arange(n_comparisons)
train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=255)

embeddings_train = embeddings[train_idx]
comparison_pairs_train = comparison_pairs[train_idx]
comparison_results_train = comparison_results[train_idx]
users_train = users[train_idx]

embeddings_test = embeddings[test_idx]
comparison_pairs_test = comparison_pairs[test_idx]
comparison_results_test = comparison_results[test_idx]
users_test = users[test_idx]

np.save("train_idx.npy", train_idx)
np.save("test_idx.npy", test_idx)

print(f"Training EM algorithm on {n_groups} groups.")
np.random.seed(226)
phi, final_groups, n_iters = rewardEM(embeddings_train, comparison_pairs_train, comparison_results_train, 
                                        n_items, n_groups, users_train, max_iter=20, cycle_window=2)
np.save(f"phi_learned_u_{n_groups}_train.npy", phi)
np.save(f"learned_groups_u_{n_groups}_train.npy", final_groups)
