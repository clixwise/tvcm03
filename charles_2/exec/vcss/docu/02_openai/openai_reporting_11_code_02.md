Below is **clean, executable Python code** that does exactly what you asked:

1️⃣ creates **small mock long datasets** (VEINES + VCSS)
2️⃣ generates the **table-ready dataframes** in the exact reporting formats we agreed on

No plotting code yet, no statsmodels yet — just **data generation + table construction**, fully aligned with the *Final coherence map*.

---

# 1️⃣ Create small mock *long* datasets

```python
import numpy as np
import pandas as pd

np.random.seed(42)

# -----------------------
# Parameters
# -----------------------
n_patients = 10
patients = [f"P{i:02d}" for i in range(1, n_patients + 1)]
timepoints = ["T0", "T1", "T2"]
limbs = ["R", "L"]
```

---

## A. VEINES mock (patient-level, long)

```python
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
```

---

## B. VCSS mock (limb-level, long)

```python
# -----------------------
# VCSS long dataset
# -----------------------
rows = []

for pid in patients:
    for limb in limbs:
        baseline = np.random.randint(6, 12)

        for t, delta in zip(timepoints, [0, -4, -6]):
            rows.append({
                "patient_id": pid,
                "limb": limb,
                "timepoint": t,
                "VCSS": max(0, baseline + delta + np.random.randint(-1, 2))
            })

df_vcss_long = pd.DataFrame(rows)
df_vcss_long.head()
```

---

# 2️⃣ Generation of **table-ready dataframes**

These outputs are **copy/paste ready** for your manuscript.

---

## **Table 2 – VEINES descriptive (Mean ± SD)**

```python
def mean_sd(x):
    return f"{x.mean():.1f} ± {x.std(ddof=1):.1f}"

df_tab_veines_desc = (
    df_veines_long
    .melt(id_vars=["patient_id", "timepoint"],
          value_vars=["VEINES_QOL", "VEINES_Sym"],
          var_name="Metric",
          value_name="ValueNum")
    .groupby(["Metric", "timepoint"])["ValueNum"]
    .apply(mean_sd)
    .reset_index(name="Value")
)

df_tab_veines_desc.insert(0, "ID", "StatTranVEINES_01_desc")
df_tab_veines_desc.rename(columns={"timepoint": "Timepoint"}, inplace=True)

df_tab_veines_desc
```

---

## **Table 4 – VCSS descriptive (Median [IQR])**

```python
def median_iqr(x):
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    return f"{x.median():.1f} [{q1:.1f}–{q3:.1f}]"

df_tab_vcss_desc = (
    df_vcss_long
    .assign(Metric=lambda d: "VCSS_" + d["limb"])
    .groupby(["Metric", "timepoint"])["VCSS"]
    .apply(median_iqr)
    .reset_index(name="Value")
)

df_tab_vcss_desc.insert(0, "ID", "StatTranVCSS_01_desc")
df_tab_vcss_desc.rename(columns={"timepoint": "Timepoint"}, inplace=True)
df_tab_vcss_desc = df_tab_vcss_desc.drop(columns="limb", errors="ignore")

df_tab_vcss_desc
```

---

## **Table 3 – VEINES model-style table (mock EMM + CI)**

> ⚠️ For now, this uses **group means ± 1.96·SE** as a placeholder
> (later we’ll replace with `statsmodels` EMMs)

```python
def mean_ci(x):
    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se
    return mean, ci_low, ci_high

rows = []

for metric in ["VEINES_QOL", "VEINES_Sym"]:
    baseline_mean, _, _ = mean_ci(
        df_veines_long.loc[df_veines_long.timepoint == "T0", metric]
    )

    for tp in timepoints:
        mean, lo, hi = mean_ci(
            df_veines_long.loc[df_veines_long.timepoint == tp, metric]
        )

        rows.append({
            "ID": "StatTranVEINES_02_model",
            "Metric": metric,
            "Timepoint": tp,
            "Estimated Mean (95% CI)": f"{mean:.1f} [{lo:.1f}–{hi:.1f}]",
            "Mean Change vs T0 (95% CI)":
                "—" if tp == "T0"
                else f"{mean - baseline_mean:+.1f} [{lo - baseline_mean:+.1f}–{hi - baseline_mean:+.1f}]"
        })

df_tab_veines_model = pd.DataFrame(rows)
df_tab_veines_model
```

---

## **Table 5 – VCSS change vs baseline (mock model-style)**

```python
rows = []

for limb in limbs:
    base = df_vcss_long.query("limb == @limb and timepoint == 'T0'")["VCSS"].mean()

    for tp in ["T1", "T2"]:
        vals = df_vcss_long.query("limb == @limb and timepoint == @tp")["VCSS"]
        mean = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(len(vals))
        lo = mean - 1.96 * se
        hi = mean + 1.96 * se

        rows.append({
            "ID": "StatTranVCSS_02_model",
            "Metric": f"VCSS_{limb}",
            "Timepoint": tp,
            "Mean Change vs T0 (95% CI)":
                f"{mean - base:+.1f} [{lo - base:+.1f}–{hi - base:+.1f}]"
        })

df_tab_vcss_change = pd.DataFrame(rows)
df_tab_vcss_change
```

---

# ✅ What you now have

* ✔️ `df_veines_long`, `df_vcss_long` → **canonical mock long data**
* ✔️ `df_tab_veines_desc` → Table 2
* ✔️ `df_tab_veines_model` → Table 3
* ✔️ `df_tab_vcss_desc` → Table 4
* ✔️ `df_tab_vcss_change` → Table 5
* ✔️ All tables in **copy/paste-friendly string format**

---

### Next step (recommended)

👉 Replace the placeholder CI logic with **true mixed-model EMMs** using `statsmodels`
👉 Or generate the **plot-ready numeric dfs** directly from these mocks

Just tell me where you want to go next.
