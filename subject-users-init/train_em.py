import numpy as np
import pandas as pd
import json
import argparse
from sklearn.model_selection import train_test_split
from util import rewardEM
from config import bool_cols, group_tags

np.random.seed(101)

parser = argparse.ArgumentParser(description="Train EM reward model with a given number of groups.")
parser.add_argument("n_groups", type=int, help="Number of groups for the EM algorithm.")
args = parser.parse_args()
n_groups = args.n_groups

embeddings = np.load("prompt_embeddings_no_ties.npy")
comparison_pairs = np.load("comparison_pairs_no_ties.npy")
comparison_results = np.load("comparison_results_no_ties.npy")
category_users = np.load("category_users.npy")

with open('idx_to_model.json', 'r') as file:
    idx_to_model = json.load(file)
n_items = len(idx_to_model)

n_comparisons = embeddings.shape[0]
indices = np.arange(n_comparisons)
train_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=255)

embeddings_train = embeddings[train_idx]
comparison_pairs_train = comparison_pairs[train_idx]
comparison_results_train = comparison_results[train_idx]
category_users_train = category_users[train_idx]

embeddings_test = embeddings[test_idx]
comparison_pairs_test = comparison_pairs[test_idx]
comparison_results_test = comparison_results[test_idx]
category_users_test = category_users[test_idx]

np.save("train_idx.npy", train_idx)
np.save("test_idx.npy", test_idx)

# ── Tag-based initialization ───────────────────────────────────────────────
n_bool_features = len(bool_cols)

def decode_bit(user_id, bit_idx, n_features):
    return bool((user_id >> (n_features - 1 - bit_idx)) & 1)

def get_initial_group(user_id):
    for group, tags in group_tags.items():
        if any(decode_bit(user_id, bool_cols.index(tag), n_bool_features) for tag in tags):
            return group
    return 0  # default group

initial_groups_train = np.array([get_initial_group(u) for u in category_users_train])

print(f"Training EM algorithm on {n_groups} groups.")
phi, final_groups, n_iters = rewardEM(
    embeddings_train, comparison_pairs_train, comparison_results_train,
    n_items, n_groups, category_users_train,
    initial_groups=initial_groups_train,
    max_iter=20, cycle_window=2
)
np.save(f"phi_learned_u_{n_groups}_train.npy", phi)
np.save(f"learned_groups_u_{n_groups}_train.npy", final_groups)
