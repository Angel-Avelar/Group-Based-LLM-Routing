import pandas as pd

# Load all parquet files
results_df          = pd.read_parquet("router_results.parquet")
results_base_df     = pd.read_parquet("router_base_results.parquet")
subj_df             = pd.read_parquet("router_results_by_subject.parquet")
subj_base_df        = pd.read_parquet("router_base_results_by_subject.parquet")


def format_mean_std(runs_df):
    """
    For overall results: runs_df has shape (n_repeat x alphas).
    Returns a DataFrame with 'mean (std)' strings, one row, columns = alphas.
    """
    mean = runs_df.mean() * 100
    std  = runs_df.std()  * 100
    return pd.DataFrame(
        {col: [f"{mean[col]:.1f} ({std[col]:.1f})"] for col in runs_df.columns},
        index=["Accuracy"]
    )


def to_latex(df, caption, label, bold_max=True):
    """
    Renders a DataFrame to a LaTeX table string.
    Optionally bolds the max value in each row.
    """
    if bold_max:
        def bold_max_row(row):
            # Only bold numeric-looking cells
            try:
                numeric = row.map(lambda x: float(x.split()[0]) if isinstance(x, str) else x)
                max_val = numeric.max()
                return [
                    r"\textbf{" + str(v) + "}" if (
                        isinstance(v, str) and abs(float(v.split()[0]) - max_val) < 1e-9
                    ) else str(v)
                    for v in row
                ]
            except Exception:
                return row

        df = df.apply(bold_max_row, axis=1, result_type="expand")
        df.columns = pd.Index(df.columns) if not isinstance(df.columns, pd.Index) else df.columns

    latex = df.to_latex(
        escape=False,
        caption=caption,
        label=label,
        column_format="l" + "c" * len(df.columns),
        position="ht",
    )
    return latex


def grouped_mean_to_display(df):
    """
    Grouped summary df has shape (group x alphas), values already averaged over runs.
    Format as percentages with 1 decimal.
    """
    return (df * 100).round(1).astype(str)


# ── Overall ──────────────────────────────────────────────────────────────────
overall_display      = format_mean_std(results_df)
overall_base_display = format_mean_std(results_base_df)
overall_combined = pd.concat([
    overall_display.rename(index={"Accuracy": "Preference-aware"}),
    overall_base_display.rename(index={"Accuracy": "Base"}),
])

latex_overall = to_latex(
    overall_combined,
    caption="Overall routing accuracy (\\%) across budget levels $\\alpha$. Mean (std) over 500 runs.",
    label="tab:overall_accuracy",
)

# ── By subject ────────────────────────────────────────────────────────────────
subj_combined = pd.concat([
    grouped_mean_to_display(subj_df),
    grouped_mean_to_display(subj_base_df),
], keys=["Preference-aware", "Base"])
subj_combined.index = subj_combined.index.map(lambda x: f"{x[0]} – {x[1]}")

latex_subj = to_latex(
    subj_combined,
    caption="Routing accuracy (\\%) by subject. Averaged over 500 runs.",
    label="tab:accuracy_by_subject",
)

# ── Print all ─────────────────────────────────────────────────────────────────
for name, latex in [
    ("Overall",            latex_overall),
    ("By Subject",         latex_subj),
]:
    print(f"\n% {'─'*60}")
    print(f"% {name}")
    print(f"% {'─'*60}")
    print(latex)

# ── Save to .tex file ─────────────────────────────────────────────────────────
with open("routing_tables.tex", "w") as f:
    f.write("% Auto-generated routing accuracy tables\n\n")
    for latex in [latex_overall, latex_subj]:
        f.write(latex + "\n\n")

print("Saved to routing_tables.tex")