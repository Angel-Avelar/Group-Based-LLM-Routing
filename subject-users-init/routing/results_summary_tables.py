import pandas as pd
from pathlib import Path

Path("latex_tables").mkdir(exist_ok=True)

def save_latex(df, filename, caption, label):
    latex = df.to_latex(
        caption=caption,
        label=label,
        position='h',
        escape=True,
        multirow=True,
    )
    with open(f"latex_tables/{filename}.tex", "w") as f:
        f.write(latex)

data = pd.read_parquet("mmlu_model_responses.parquet")
data = data[data['language'] == 'en']

# Overall accuracy per model
overall = pd.DataFrame({
    'Accuracy': {
        'Mistral': f"{data['mistral-correct'].mean():.1%}",
        'Claude':  f"{data['claude-correct'].mean():.1%}",
    },
    'N': {
        'Mistral': data['mistral-correct'].count(),
        'Claude':  data['claude-correct'].count(),
    }
})
save_latex(overall, "overall_accuracy", "Overall Accuracy per Model", "tab:overall_accuracy")

# Overall preference
preference = pd.DataFrame({
    'Claude Preferred (Mixture)': [f"{data['preference'].mean():.1%}"],
    'Claude Preferred (Base)':    [f"{data['preference_base'].mean():.1%}"],
    'N': [len(data)]
}, index=['All'])
save_latex(preference, "overall_preference", "Overall Claude Preference over Mistral", "tab:overall_preference")

# Accuracy + preference per subject
subject = data.groupby('subject').agg(
    mistral_acc=('mistral-correct', 'mean'),
    claude_acc=('claude-correct', 'mean'),
    claude_preferred=('preference', 'mean'),
    claude_preferred_base=('preference_base', 'mean'),
    n=('answer', 'count')
).sort_values('claude_acc', ascending=False)
subject.columns = ['Mistral Acc.', 'Claude Acc.', 'Claude Preferred', 'Claude Preferred (Base)', 'N']
subject[['Mistral Acc.', 'Claude Acc.', 'Claude Preferred', 'Claude Preferred (Base)']] = \
    subject[['Mistral Acc.', 'Claude Acc.', 'Claude Preferred', 'Claude Preferred (Base)']].applymap('{:.1%}'.format)
save_latex(subject, "by_subject", "Accuracy and Preference by Subject", "tab:by_subject")

# Agreement breakdown — per preference model
def agreement_block(pref_col):
    return pd.DataFrame({
        'Count': {
            'Both Correct':    data['both_correct'].sum(),
            'Neither Correct': data['neither_correct'].sum(),
            'Only Mistral':    data['only_mistral'].sum(),
            'Only Claude':     data['only_claude'].sum(),
        },
        'Pct': {
            'Both Correct':    f"{data['both_correct'].mean():.1%}",
            'Neither Correct': f"{data['neither_correct'].mean():.1%}",
            'Only Mistral':    f"{data['only_mistral'].mean():.1%}",
            'Only Claude':     f"{data['only_claude'].mean():.1%}",
        },
        'Claude Preferred': {
            k: f"{data.loc[data[col], pref_col].mean():.1%}" if data[col].sum() > 0 else '—'
            for k, col in [
                ('Both Correct',    'both_correct'),
                ('Neither Correct', 'neither_correct'),
                ('Only Mistral',    'only_mistral'),
                ('Only Claude',     'only_claude'),
            ]
        }
    })

data['both_correct']    = data['mistral-correct'] & data['claude-correct']
data['neither_correct'] = ~data['mistral-correct'] & ~data['claude-correct']
data['only_mistral']    = data['mistral-correct'] & ~data['claude-correct']
data['only_claude']     = ~data['mistral-correct'] & data['claude-correct']

agreement_mixture = agreement_block('preference')
agreement_mixture.columns = ['Count', 'Pct', 'Claude Preferred (Mixture)']

agreement_base = agreement_block('preference_base')[['Claude Preferred']]
agreement_base.columns = ['Claude Preferred (Base)']

agreement = pd.concat([agreement_mixture, agreement_base], axis=1)
save_latex(agreement, "model_agreement", "Model Agreement Breakdown", "tab:model_agreement")

print("All tables saved to latex_tables/")