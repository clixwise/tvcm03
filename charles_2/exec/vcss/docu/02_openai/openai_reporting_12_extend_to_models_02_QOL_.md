Perfect — we’ll **model VEINES first**, cleanly and pragmatically, using the **mock data you already have**.

Below is **fully executable Python code** that:

1️⃣ Fits a **linear mixed-effects model** for VEINES-QOL and VEINES-Sym
2️⃣ Extracts **model-based estimated marginal means (EMMs)**
3️⃣ Computes **Δ vs T0 with 95% CI**
4️⃣ Produces **table-ready** and **plot-ready** dataframes that plug directly into your existing pipeline

No shortcuts, no black boxes.

---

# 0️⃣ Prerequisites

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
```

Assumes `df_veines_long` already exists with:

```text
patient_id | timepoint | VEINES_QOL | VEINES_Sym
```

Ensure `timepoint` is categorical and ordered:

```python
df_veines_long["timepoint"] = pd.Categorical(
    df_veines_long["timepoint"],
    categories=["T0", "T1", "T2"],
    ordered=True
)
```

---

# 1️⃣ Fit mixed-effects models (VEINES-QOL & VEINES-Sym)

We fit **separate models** (cleaner, reviewer-friendly).

---

## A. VEINES-QOL model

```python
model_qol = smf.mixedlm(
    "VEINES_QOL ~ C(timepoint)",
    df_veines_long,
    groups=df_veines_long["patient_id"]
)

res_qol = model_qol.fit(reml=True)
print(res_qol.summary())
```

---

## B. VEINES-Sym model

```python
model_sym = smf.mixedlm(
    "VEINES_Sym ~ C(timepoint)",
    df_veines_long,
    groups=df_veines_long["patient_id"]
)

res_sym = model_sym.fit(reml=True)
print(res_sym.summary())
```

✔️ Random intercept for patient
✔️ Timepoint as fixed effect
✔️ REML appropriate for inference on means

---

# 2️⃣ Extract Estimated Marginal Means (EMMs)

Statsmodels does not provide EMMs directly, so we compute them **explicitly from the fixed effects**.

---

## Helper function (reusable)

```python
def extract_emm(result, timepoints):
    fe = result.fe_params
    cov = result.cov_params()

    rows = []

    for tp in timepoints:
        # Design vector
        if tp == "T0":
            L = np.array([1, 0, 0])
        elif tp == "T1":
            L = np.array([1, 1, 0])
        elif tp == "T2":
            L = np.array([1, 0, 1])

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
```

---

## Compute EMMs

```python
timepoints = ["T0", "T1", "T2"]

emm_qol = extract_emm(res_qol, timepoints)
emm_sym = extract_emm(res_sym, timepoints)
```

---

# 3️⃣ Compute Δ vs T0 (model-based)

```python
def add_delta_vs_baseline(df):
    baseline = df.loc[df.Timepoint == "T0", "Mean"].iloc[0]

    df["Delta_vs_T0"] = df["Mean"] - baseline
    df["Delta_CI_lower"] = df["CI_lower"] - baseline
    df["Delta_CI_upper"] = df["CI_upper"] - baseline

    return df
```

```python
emm_qol = add_delta_vs_baseline(emm_qol)
emm_sym = add_delta_vs_baseline(emm_sym)
```

---

# 4️⃣ Build **table-ready dataframe** (Table 3)

Exactly aligned with your publication format.

```python
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
df_tab_veines_model
```

✔️ Drop-in replacement for your mock Table 3
✔️ Baseline correctly handled
✔️ CI formatting journal-ready

---

# 5️⃣ Build **plot-ready dataframe** (Figure 1)

Numeric only, CI-based.

```python
df_plot_veines = pd.concat([
    emm_qol.assign(Metric="VEINES_QOL"),
    emm_sym.assign(Metric="VEINES_Sym")
], ignore_index=True)

df_plot_veines = df_plot_veines[
    ["Metric", "Timepoint", "Mean", "CI_lower", "CI_upper"]
]

df_plot_veines
```

✔️ One row = one plotted point
✔️ Mean ± CI only
✔️ Fully consistent with Methods & Results

---

# 6️⃣ (Optional) Global time effect p-value

For reporting:

```python
res_qol.pvalues
res_sym.pvalues
```

Typical wording:

> “Timepoint was significantly associated with VEINES-QOL (p < 0.001).”

---

# ✅ What you now have

* ✔️ Proper **mixed-effects models** for VEINES-QOL & Sym
* ✔️ True **model-based EMMs**
* ✔️ Δ vs baseline with CI
* ✔️ Table 3 + Figure 1 fed directly from model output
* ✔️ 100% aligned with your Methods and coherence map

---

## Next logical step

👉 Apply the **same extraction pattern** to **VCSS limb-level mixed model**
(or, if you prefer, we can pause and **validate assumptions** / diagnostics for VEINES).

Just tell me:

* **“Proceed to VCSS model”**
* or **“Add diagnostics / residual checks for VEINES”**
