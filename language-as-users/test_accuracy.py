import numpy as np
import json
import argparse
import pandas as pd
from scipy.special import expit
from util import create_sparse_features

def predict_from_phi(phi, group_assignments, embeddings_expanded):
    """
    Reconstruct predictions using learned phi parameters and group assignments.
    
    phi: (n_items * n_features, n_groups)
    group_assignments: learned groupings list
    embeddings_expanded: sparse feature matrix from create_sparse_features
    """
    n_comparisons = embeddings_expanded.shape[0]
    y_pred = np.zeros(n_comparisons, dtype=int)
    
    for u in range(phi.shape[1]):
        in_group = group_assignments == u
        if sum(in_group) == 0:
            continue
        X_u = embeddings_expanded[in_group,]
        beta_u = phi[:, u]
        log_odds = X_u @ beta_u
        probs = expit(log_odds)
        y_pred[in_group] = (probs >= 0.5).astype(int)

    return y_pred


def compute_accuracy_records(phi, groups_train, groups_test,
                              comparison_results_train, comparison_results_test,
                              embeddings_expanded_train, embeddings_expanded_test,
                              model_label):
    """
    Compute overall and per-group train/test accuracy for a given phi and group assignments.
    Returns a list of record dicts.
    """
    records = []
    n_groups = phi.shape[1]

    # --- Train ---
    y_pred_train = predict_from_phi(phi, groups_train, embeddings_expanded_train)
    train_acc = np.mean(y_pred_train == comparison_results_train)
    records.append({'model': model_label, 'split': 'train', 'group': 'overall',
                    'n': len(comparison_results_train), 'accuracy': round(float(train_acc), 4)})
    # print(f"\n[{model_label}] Training Accuracy: {train_acc:.4f}")

    for u in range(n_groups):
        in_group = groups_train == u
        if sum(in_group) > 0:
            acc = np.mean(y_pred_train[in_group] == comparison_results_train[in_group])
            records.append({'model': model_label, 'split': 'train', 'group': f'g{u}',
                            'n': int(sum(in_group)), 'accuracy': round(float(acc), 4)})
            # print(f"  Group {u} (n={sum(in_group)}): Accuracy = {acc:.4f}")

    # --- Test ---
    y_pred_test = predict_from_phi(phi, groups_test, embeddings_expanded_test)
    test_acc = np.mean(y_pred_test == comparison_results_test)
    records.append({'model': model_label, 'split': 'test', 'group': 'overall',
                    'n': len(comparison_results_test), 'accuracy': round(float(test_acc), 4)})
    # print(f"[{model_label}] Test Accuracy: {test_acc:.4f}")

    for u in range(n_groups):
        in_group = groups_test == u
        if sum(in_group) > 0:
            acc = np.mean(y_pred_test[in_group] == comparison_results_test[in_group])
            records.append({'model': model_label, 'split': 'test', 'group': f'g{u}',
                            'n': int(sum(in_group)), 'accuracy': round(float(acc), 4)})
            # print(f"  Group {u} (n={sum(in_group)}): Accuracy = {acc:.4f}")

    return records, y_pred_train, y_pred_test


parser = argparse.ArgumentParser(description="Evaluate EM reward model accuracy.")
parser.add_argument("n_groups", type=int, help="Number of groups for the EM algorithm.")
args = parser.parse_args()
n_groups = args.n_groups

embeddings = np.load("prompt_embeddings_no_ties.npy")
comparison_pairs = np.load("comparison_pairs_no_ties.npy")
comparison_results = np.load("comparison_results_no_ties.npy")
language_users = np.load("language_users.npy")

with open('idx_to_model.json', 'r') as file:
    idx_to_model = json.load(file)
n_items = len(idx_to_model)

train_idx = np.load('train_idx.npy')
test_idx = np.load('test_idx.npy')

embeddings_train = embeddings[train_idx]
comparison_pairs_train = comparison_pairs[train_idx]
comparison_results_train = comparison_results[train_idx]
language_users_train = language_users[train_idx]

embeddings_test = embeddings[test_idx]
comparison_pairs_test = comparison_pairs[test_idx]
comparison_results_test = comparison_results[test_idx]
language_users_test = language_users[test_idx]

# --- Build expanded embeddings (shared for both models) ---
n_comparisons = len(embeddings_train)
intercept = np.ones((n_comparisons, 1))
embeddings_train_int = np.hstack([intercept, embeddings_train])
n_features = embeddings_train_int.shape[1]

embeddings_expanded_train = create_sparse_features(n_items, n_features, n_comparisons,
                                                   comparison_pairs_train, embeddings_train_int)

n_comparisons_test = len(embeddings_test)
intercept_test = np.ones((n_comparisons_test, 1))
embeddings_test_int = np.hstack([intercept_test, embeddings_test])

embeddings_expanded_test = create_sparse_features(n_items, n_features, n_comparisons_test,
                                                  comparison_pairs_test, embeddings_test_int)

all_records = []

# ── mixture model (k = n_groups) ──────────────────────────────────────────────
phi = np.load(f"phi_learned_u_{n_groups}_train.npy")
groups_train = np.load(f"learned_groups_u_{n_groups}_train.npy")

user_assignments = {u_id: groups_train[language_users_train == u_id][0]
                    for u_id in np.unique(language_users_train)}
groups_test = np.array([user_assignments.get(u_id, 0) for u_id in language_users_test])

records, _, _ = compute_accuracy_records(
    phi, groups_train, groups_test,
    comparison_results_train, comparison_results_test,
    embeddings_expanded_train, embeddings_expanded_test,
    model_label=f"mixture_k{n_groups}"
)
all_records += records

# ── base model (k = 1) — overall accuracy ─────────────────────────────────────
phi_base = np.load("phi_learned_u_1_train.npy")
groups_train_base = np.zeros(len(groups_train), dtype=int)
groups_test_base  = np.zeros(len(groups_test),  dtype=int)

records, y_pred_train_base, y_pred_test_base = compute_accuracy_records(
    phi_base, groups_train_base, groups_test_base,
    comparison_results_train, comparison_results_test,
    embeddings_expanded_train, embeddings_expanded_test,
    model_label="base_k1"
)
all_records += records

# ── base model (k = 1) — accuracy broken down by the k=n_groups splits ────────
# print(f"\n[base_k1 | evaluated per k={n_groups} groups]")
for u in range(n_groups):
    in_train = groups_train == u
    in_test  = groups_test  == u
    if sum(in_train) > 0:
        acc_tr = np.mean(y_pred_train_base[in_train] == comparison_results_train[in_train])
        all_records.append({'model': f'base_k1_by_k{n_groups}', 'split': 'train',
                            'group': f'g{u}', 'n': int(sum(in_train)),
                            'accuracy': round(float(acc_tr), 4)})
        # print(f"  Group {u} train (n={sum(in_train)}): {acc_tr:.4f}")
    if sum(in_test) > 0:
        acc_te = np.mean(y_pred_test_base[in_test] == comparison_results_test[in_test])
        all_records.append({'model': f'base_k1_by_k{n_groups}', 'split': 'test',
                            'group': f'g{u}', 'n': int(sum(in_test)),
                            'accuracy': round(float(acc_te), 4)})
        # print(f"  Group {u} test  (n={sum(in_test)}): {acc_te:.4f}")

# ── Save results ──────────────────────────────────────────────────────────────

# Build per-group rows for mixture model
mixture_train = {r['group']: (r['n'], r['accuracy'])
                 for r in all_records
                 if r['model'] == f'mixture_k{n_groups}' and r['split'] == 'train'}
mixture_test  = {r['group']: (r['n'], r['accuracy'])
                 for r in all_records
                 if r['model'] == f'mixture_k{n_groups}' and r['split'] == 'test'}

# Build per-group rows for base model broken down by n_groups splits
base_train = {r['group']: (r['n'], r['accuracy'])
              for r in all_records
              if r['model'] == f'base_k1_by_k{n_groups}' and r['split'] == 'train'}
base_test  = {r['group']: (r['n'], r['accuracy'])
              for r in all_records
              if r['model'] == f'base_k1_by_k{n_groups}' and r['split'] == 'test'}

# Also include overall row using base_k1 overall
base_train['overall'] = next((r['n'], r['accuracy']) for r in all_records
                              if r['model'] == 'base_k1' and r['split'] == 'train'
                              and r['group'] == 'overall')
base_test['overall']  = next((r['n'], r['accuracy']) for r in all_records
                              if r['model'] == 'base_k1' and r['split'] == 'test'
                              and r['group'] == 'overall')

all_groups = ['overall'] + [f'g{u}' for u in range(n_groups)]
rows = []
for group in all_groups:
    train_n,    mix_train_acc  = mixture_train.get(group, (None, None))
    _,          mix_test_acc   = mixture_test.get(group,  (None, None))
    test_n_val, base_test_acc  = base_test.get(group,     (None, None))
    _,          base_train_acc = base_train.get(group,    (None, None))

    # use mixture n for train/test (same grouping)
    test_n, _ = mixture_test.get(group, (None, None))

    rows.append({
        'group':                group,
        'train_n':              train_n,
        f'train_acc_k{n_groups}': mix_train_acc,
        'train_acc_base':       base_train_acc,
        'test_n':               test_n,
        f'test_acc_k{n_groups}':  mix_test_acc,
        'test_acc_base':        base_test_acc,
    })

df_wide = pd.DataFrame(rows)

print("\n── Results table ──")
print(df_wide.to_string(index=False))

df_wide.to_json(f"accuracy_results_k{n_groups}.json", orient='records', indent=2)
# print(f"\nSaved to accuracy_results_k{n_groups}.json")

latex = df_wide.to_latex(
    index=False,
    escape=False,
    caption=f"Train/test accuracy comparison (k={n_groups} vs base k=1)",
    label=f"tab:accuracy_k{n_groups}",
    column_format='l' + 'r' * (len(df_wide.columns) - 1)
)

with open(f"accuracy_results_k{n_groups}.tex", "w") as f:
    f.write(latex)

print(latex)
print(f"Saved to accuracy_results_k{n_groups}.tex")