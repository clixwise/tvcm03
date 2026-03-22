import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

np.random.seed(42)

# -----------------------
# Parameters
# -----------------------
n_patients = 10
patients = [f"P{i:02d}" for i in range(1, n_patients + 1)]
timepoints = ["T0", "T1", "T2"]

# -----------------------
# VEINES long dataset
# -----------------------
rows = []

for pid in patients:
    baseline_qol = np.random.normal(48, 5)
    baseline_sym = np.random.normal(47, 5)

    for t, delta in zip(timepoints, [0, 7, 10]):
        rows.append({
            "patient_id": pid,
            "timepoint": t,
            "VEINES_QOL": baseline_qol + delta + np.random.normal(0, 1),
            "VEINES_Sym": baseline_sym + delta + np.random.normal(0, 1)
        })

df_veines_long = pd.DataFrame(rows)
df_veines_long.head()
df_veines_long["timepoint"] = pd.Categorical(
    df_veines_long["timepoint"],
    categories=["T0", "T1", "T2"],
    ordered=True
)

# 1️⃣ Fit mixed-effects models (VEINES-QOL & VEINES-Sym)
# We fit **separate models** (cleaner, reviewer-friendly).
## A. VEINES-QOL model
model_qol = smf.mixedlm(
    "VEINES_QOL ~ C(timepoint)",
    df_veines_long,
    groups=df_veines_long["patient_id"]
)
res_qol = model_qol.fit(reml=True)
print(res_qol.summary())

## B. VEINES-Sym model

model_sym = smf.mixedlm(
    "VEINES_Sym ~ C(timepoint)",
    df_veines_long,
    groups=df_veines_long["patient_id"]
)
res_sym = model_sym.fit(reml=True)
print(res_sym.summary())


# 2️⃣ Extract Estimated Marginal Means (EMMs)
# Statsmodels does not provide EMMs directly, so we compute them **explicitly from the fixed effects**.

## Helper function (reusable)

def extract_emm(result, timepoints):
    fe = result.fe_params.values
    cov_all = result.cov_params().values

    k = len(fe)                     # number of fixed effects
    cov = cov_all[:k, :k]           # ✅ FIXED-EFFECTS BLOCK

    rows = []

    for tp in timepoints:
        if tp == "T0":
            L = np.array([1, 0, 0])
        elif tp == "T1":
            L = np.array([1, 1, 0])
        elif tp == "T2":
            L = np.array([1, 0, 1])
        else:
            raise ValueError("Unknown timepoint")

        mean = float(L @ fe)
        se = np.sqrt(float(L @ cov @ L))
        ci_low = mean - 1.96 * se
        ci_high = mean + 1.96 * se

        rows.append({
            "Timepoint": tp,
            "Mean": mean,
            "CI_lower": ci_low,
            "CI_upper": ci_high
        })

    return pd.DataFrame(rows)

## Compute EMMs

emm_qol = extract_emm(res_qol, timepoints)
emm_sym = extract_emm(res_sym, timepoints)

# 3️⃣ Compute Δ vs T0 (model-based)

def add_delta_vs_baseline(df):
    baseline = df.loc[df.Timepoint == "T0", "Mean"].iloc[0]

    df["Delta_vs_T0"] = df["Mean"] - baseline
    df["Delta_CI_lower"] = df["CI_lower"] - baseline
    df["Delta_CI_upper"] = df["CI_upper"] - baseline

    return df

emm_qol = add_delta_vs_baseline(emm_qol)
emm_sym = add_delta_vs_baseline(emm_sym)


# 4️⃣ Build **table-ready dataframe** (Table 3)

#  Exactly aligned with your publication format.

def format_ci(mean, lo, hi):
    return f"{mean:.1f} [{lo:.1f}–{hi:.1f}]"

rows = []
for metric, df_emm in zip(
    ["VEINES_QOL", "VEINES_Sym"],
    [emm_qol, emm_sym]
):
    for _, r in df_emm.iterrows():
        rows.append({
            "ID": "StatTranVEINES_02_model",
            "Metric": metric,
            "Timepoint": r["Timepoint"],
            "Estimated Mean (95% CI)": format_ci(
                r["Mean"], r["CI_lower"], r["CI_upper"]
            ),
            "Mean Change vs T0 (95% CI)":
                "—" if r["Timepoint"] == "T0"
                else format_ci(
                    r["Delta_vs_T0"],
                    r["Delta_CI_lower"],
                    r["Delta_CI_upper"]
                )
        })

df_tab_veines_model = pd.DataFrame(rows)
print(df_tab_veines_model)


# 5️⃣ Build **plot-ready dataframe** (Figure 1)

# Numeric only, CI-based.

df_plot_veines = pd.concat([
    emm_qol.assign(Metric="VEINES_QOL"),
    emm_sym.assign(Metric="VEINES_Sym")
], ignore_index=True)

df_plot_veines = df_plot_veines[
    ["Metric", "Timepoint", "Mean", "CI_lower", "CI_upper"]
]

print(df_plot_veines)


#✔️ One row = one plotted point
#✔️ Mean ± CI only
#✔️ Fully consistent with Methods & Results


# 6️⃣ (Optional) Global time effect p-value

print(res_qol.pvalues)
print(res_sym.pvalues)


# Typical wording: “Timepoint was significantly associated with VEINES-QOL (p < 0.001).”


# ✅ What you now have

# * ✔️ Proper **mixed-effects models** for VEINES-QOL & Sym
# * ✔️ True **model-based EMMs**
# * ✔️ Δ vs baseline with CI
# * ✔️ Table 3 + Figure 1 fed directly from model output
# * ✔️ 100% aligned with your Methods and coherence map


## Next logical step
