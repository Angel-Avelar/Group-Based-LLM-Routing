from scipy.sparse import coo_matrix
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from datasets import load_dataset
import os
import json
import pandas as pd

def get_data():
    ds = load_dataset("lmarena-ai/arena-human-preference-140k")
    ds_filtered = ds.filter(lambda x: x["winner"] in ['model_a', 'model_b'])
    split = ds_filtered["train"]
    return split

def load_category_tags(split):
    if os.path.exists("category_tags.json"):
        with open("category_tags.json", "r") as file:
            rows = json.load(file)
    else:
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
        with open("category_tags.json", "w") as file:
            json.dump(rows, file)

    return pd.DataFrame(rows)

def create_sparse_features(n_items, feature_dim, n_comparisons, comparison_pairs, features_original=None):
    expanded_dim = n_items * feature_dim

    # Extract item indices
    item1_idx = comparison_pairs[:, 0].astype(int)
    item2_idx = comparison_pairs[:, 1].astype(int)

    # Add intercept dimension
    intercept = np.ones((n_comparisons, 1))
    if features_original is None:
        features_original = intercept

    # 1. Flatten the data values
    # features_original is (n_comparisons, feature_dim)
    data_item2 = features_original.ravel()
    data_item1 = -features_original.ravel()

    # 2. Create flattened row indices
    # Each row index repeats `feature_dim` times to match the flattened data
    row_indices = np.repeat(np.arange(n_comparisons), feature_dim)

    # 3. Create flattened column indices
    feature_offsets = np.arange(feature_dim)
    # Broadcasting creates (n_comparisons, feature_dim), then ravel flattens it
    col_indices_item2 = (item2_idx[:, np.newaxis] * feature_dim + feature_offsets).ravel()
    col_indices_item1 = (item1_idx[:, np.newaxis] * feature_dim + feature_offsets).ravel()

    # Combine data, rows, and columns for both items
    data = np.concatenate([data_item2, data_item1])
    rows_combined = np.concatenate([row_indices, row_indices])
    cols_combined = np.concatenate([col_indices_item2, col_indices_item1])

    # Construct the sparse matrix directly in COO format, then convert to CSR
    X_expanded = coo_matrix(
        (data, (rows_combined, cols_combined)),
        shape=(n_comparisons, expanded_dim)
    ).tocsr()

    return X_expanded

def project_parameter(beta, n_items, feature_dim):
    beta_matrix = beta.reshape(n_items, feature_dim)
    feature_means = np.mean(beta_matrix, axis=0)
    beta_projected = (beta_matrix - feature_means).flatten()

    return beta_projected

def cluster_assignment(phi, embeddings, comparison_results, user):
    """
    Returns the hard cluster assignment for each comparison as an index
    """
    utility_difference = embeddings @ phi
    w = np.clip(1 / (1 + np.exp(-utility_difference)), 1e-10, 1 - 1e-10)

    # Log BCE per row per group: (n_comparisons, n_groups)
    log_bce = (comparison_results[:, np.newaxis] * np.log(w) +
               (1 - comparison_results)[:, np.newaxis] * np.log(1 - w))

    # Sum log-likelihoods per user per group
    unique_users = np.unique(user)
    user_assignments = {}

    for u_id in unique_users:
        mask = user == u_id
        user_log_likelihood = log_bce[mask].sum(axis=0)  # (n_groups,)
        user_assignments[u_id] = np.argmax(user_log_likelihood)

    # Map back to comparison-level assignments
    u = np.array([user_assignments[u_id] for u_id in user])

    return u

def rewardEM(embeddings, comparison_pairs, comparison_results, n_items, n_groups, 
             user, phi_true=None, max_iter=100, cycle_window=10, initial_groups=None):
    """
    embeddings: array of size n x d, n is the number of comparisons and d is the feature size, each row corresponds to an input
    comparison_pairs: array of size n x 2, each row represents the index of the two models compared
                        assumes items are indexed from 0 to n_items - 1
    comparison_results: array of size n, each item represents if item comparison_pairs[i, 1] preferred over comparison_pairs[i, 0]
    """
    # data setup
    n_comparisons = len(embeddings)
    intercept = np.ones((n_comparisons, 1))
    embeddings = np.hstack([intercept, embeddings])
    n_features = embeddings.shape[1]

    embeddings_expanded = create_sparse_features(n_items, n_features, n_comparisons,
                                            comparison_pairs, embeddings)

    # initialize group reward parameters
    if phi_true is None:
        phi = np.random.normal(size=(n_items * n_features, n_groups)) # group parameter vector per column
    else:
        phi = phi_true
    u_prev = -np.ones(n_comparisons)
    # u_current = cluster_assignment(phi, embeddings_expanded, comparison_results, user)
    # u_current = np.random.choice(n_groups, size=n_comparisons)
    # initialize group assignments
    if initial_groups is not None:
        u_current = initial_groups.copy()
    else:
        u_current = np.random.choice(n_groups, size=n_comparisons)
    for u in range(n_groups):
        print("initial groups")
        print(u, sum(u_current == u))
    assignment_history = []
    i = 0

    # EM Algorithm
    with tqdm(total=max_iter, desc="EM iterations") as pbar:
      while i < max_iter:
      # loop until the assignments are the same (nothing will change if the assignments are the same)
          # loop through each group
          for u in range(n_groups):
              in_group = u_current == u
              # what to do if at some point a group has 0 members?
              # skip for now
              if sum(in_group) > 0:
                  # get the data based on group assignments
                  X_u = embeddings_expanded[in_group, n_features:]
                  Y_u = comparison_results[in_group]
                  # logistic regression
                  lr = LogisticRegression(fit_intercept=False,
                                  max_iter=1000,
                                  C=1e10,
                                  solver='lbfgs')
                  lr.fit(X_u, Y_u)
                  # Reconstruct full coefficient vector with zeros for first item
                  beta_learned = np.zeros(n_items * n_features)
                  beta_learned[n_features:] = lr.coef_[0]

                  # Subtract mean
                  beta_learned = project_parameter(beta_learned, n_items, n_features)
                  # assign new phi_u
                  phi[:, u] = beta_learned

          i += 1
          u_prev = u_current
          u_current = cluster_assignment(phi, embeddings_expanded, comparison_results, user)
          pbar.update(1)
          pbar.set_postfix({"iter": i, "group_sizes": [int((u_current == u).sum()) for u in range(n_groups)]})

          # Check for exact convergence
          if np.array_equal(u_current, u_prev):
            break

          # Check for cycling: has this assignment appeared before in recent history?
          if any(np.array_equal(u_current, h) for h in assignment_history):
            break
            
          assignment_history.append(u_current.copy())
          if len(assignment_history) > cycle_window:
             assignment_history.pop(0)

    print(f"\nConverged: {np.array_equal(u_current, u_prev)} after {i} iterations")
    return phi, u_current, i