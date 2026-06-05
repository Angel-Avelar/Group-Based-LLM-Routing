import pandas as pd
import numpy as np

data = pd.read_parquet("mmlu_model_responses.parquet")
np.random.seed(4)

def draw_route(preferences, alpha):
    n_prompts = len(preferences)
    n_limit = int(n_prompts * alpha)
    random_order = np.random.choice(n_prompts, n_prompts, replace=False)
    order_sort = np.argsort(random_order)
    
    router = np.zeros(n_prompts)
    strong_preferred = np.argwhere(preferences == 1).flatten()
    weak_preferred = np.argwhere(preferences == 0).flatten()

    if len(strong_preferred) >= n_limit:
        strong_random_ranks = order_sort[strong_preferred]
        top_strong = strong_preferred[np.argsort(strong_random_ranks)[:n_limit]]
        router[top_strong] = 1
    else:
        router[strong_preferred] = 1
        slots_remaining = n_limit - len(strong_preferred)
        weak_random_ranks = order_sort[weak_preferred]
        top_weak = weak_preferred[np.argsort(weak_random_ranks)[:slots_remaining]]
        router[top_weak] = 1

    return router

def compute_accuracies(router, accuracies, data, alpha_label):
    """
    Returns a dict of DataFrames:
      - 'overall': mean accuracy across all prompts
      - 'by_language': mean accuracy per language
      - 'by_subject': mean accuracy per subject
      - 'by_subject_language': mean accuracy per (subject, language)
    Each is a single float or Series indexed by group, for one run.
    """
    routed_acc = accuracies[np.arange(len(router)), router.astype(int)]
    df = data[['subject', 'language']].copy()
    df['accuracy'] = routed_acc
    return df

preferences = np.array(data['preference'])
preferences_base = np.array(data['preference_base'])

accuracies = np.zeros((len(preferences), 2))
accuracies[:, 0] = data['mistral-correct']
accuracies[:, 1] = data['claude-correct']

alphas = [0, 0.25, 0.5, 0.75, 1.0]
n_repeat = 1000

# Each stores: { alpha_label: list of per-run values }
# For overall: list of floats
# For grouped: list of Series
overall        = {}
overall_base   = {}
by_subj        = {}
by_subj_base   = {}

for alpha in alphas:
    alpha_label = f"{int(alpha * 100)}%"
    overall[alpha_label]           = []
    overall_base[alpha_label]      = []
    by_subj[alpha_label]           = []
    by_subj_base[alpha_label]      = []

    for _ in range(n_repeat):
        # Preference-aware router
        router = draw_route(preferences, alpha)
        df = compute_accuracies(router, accuracies, data, alpha_label)
        overall[alpha_label].append(df['accuracy'].mean())
        by_subj[alpha_label].append(df.groupby('subject')['accuracy'].mean())

        # Base router
        router_base = draw_route(preferences_base, alpha)
        df_base = compute_accuracies(router_base, accuracies, data, alpha_label)
        overall_base[alpha_label].append(df_base['accuracy'].mean())
        by_subj_base[alpha_label].append(df_base.groupby('subject')['accuracy'].mean())


def summarize_overall(runs_dict):
    """List of floats per alpha → DataFrame(index=run, columns=alpha)"""
    return pd.DataFrame(runs_dict)

def summarize_grouped(runs_dict):
    """List of Series per alpha → DataFrame(index=group, columns=alpha) of means across runs"""
    return pd.DataFrame({
        alpha: pd.concat(series_list, axis=1).mean(axis=1)
        for alpha, series_list in runs_dict.items()
    })


# Overall
results_df      = summarize_overall(overall)
results_base_df = summarize_overall(overall_base)

# By subject
subj_df      = summarize_grouped(by_subj)
subj_base_df = summarize_grouped(by_subj_base)


print("=== Overall accuracy (preference-aware) ===")
print(results_df.mean())
print("\n=== Overall accuracy (base) ===")
print(results_base_df.mean())

print("\n=== By subject (preference-aware) ===")
print(subj_df)
print("\n=== By subject (base) ===")
print(subj_base_df)

# Save all to parquet
results_df.to_parquet("router_results.parquet")
results_base_df.to_parquet("router_base_results.parquet")
subj_df.to_parquet("router_results_by_subject.parquet")
subj_base_df.to_parquet("router_base_results_by_subject.parquet")