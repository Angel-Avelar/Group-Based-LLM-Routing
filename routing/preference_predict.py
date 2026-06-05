# "claude-opus-4-20250514": 7
# "mistral-medium-2505": 40

import sys

sys.path.append("../language-as-users/")

import numpy as np
import json
import pandas as pd
from scipy.special import expit

def predict_from_phi(phi_a, phi_b, group_assignments, embeddings):
    n_comparisons = embeddings.shape[0]
    y_pred = np.zeros(n_comparisons, dtype=int)
    
    for u in range(phi.shape[1]):
        in_group = group_assignments == u
        if sum(in_group) == 0:
            continue
        X_u = embeddings[in_group,]
        beta_u_a = phi_a[:, u]
        beta_u_b = phi_b[:, u]
        log_odds = X_u @ (beta_u_a - beta_u_b)
        # probability of a being chosen over b
        probs = expit(log_odds)
        y_pred[in_group] = (probs >= 0.5).astype(int)

    return y_pred

n_groups = 2

embeddings = np.load("mmlu_prompts_embeddings.npy")
comparison_pairs = np.tile([40, 7], (len(embeddings), 1))
response_data = pd.read_parquet("mmlu_model_responses.parquet")

with open('../language-as-users/idx_to_model.json', 'r') as file:
    idx_to_model = json.load(file)
n_items = len(idx_to_model)

n_comparisons = len(embeddings)
intercept = np.ones((n_comparisons, 1))
embeddings = np.hstack([intercept, embeddings])
n_features = embeddings.shape[1]

phi = np.load(f"../language-as-users/phi_learned_u_{n_groups}_train.npy")
a, b = 7, 40
d = n_features

phi_a = phi[np.r_[a*n_features:(a+1)*n_features]]
phi_b = phi[np.r_[b*n_features:(b+1)*n_features]]

groups = (response_data['language'] == "en").astype(int)
response_data['preference'] = predict_from_phi(phi_a, phi_b, groups, embeddings)
response_data.to_parquet("mmlu_model_responses.parquet")

phi_base = np.load(f"../language-as-users/phi_learned_u_1_train.npy")
a, b = 7, 40
d = n_features

phi_a = phi_base[np.r_[a*n_features:(a+1)*n_features]]
phi_b = phi_base[np.r_[b*n_features:(b+1)*n_features]]
groups = np.zeros(n_comparisons)
response_data['preference_base'] = predict_from_phi(phi_a, phi_b, groups, embeddings)
response_data.to_parquet("mmlu_model_responses.parquet")
