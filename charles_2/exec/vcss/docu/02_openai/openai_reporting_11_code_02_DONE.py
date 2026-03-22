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

## A. VEINES mock (patient-level, long)

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

## B. VCSS mock (limb-level, long)

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
'''
  patient_id limb timepoint  VCSS
0        P01    R        T0    10
1        P01    R        T1     8
2        P01    R        T2     6
3        P01    L        T0    11
4        P01    L        T1     7
'''
print(df_vcss_long.head())

# 2️⃣ Generation of **table-ready dataframes**

## **Table 2 – VEINES descriptive (Mean ± SD)**
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
'''
       Metric timepoint       Value
0  VEINES_QOL        T0  47.4 ± 3.7
1  VEINES_QOL        T1  54.6 ± 3.0
2  VEINES_QOL        T2  57.5 ± 3.9
3  VEINES_Sym        T0  47.5 ± 5.4
4  VEINES_Sym        T1  53.8 ± 4.9
5  VEINES_Sym        T2  57.5 ± 4.8
'''
print (df_tab_veines_desc)
df_tab_veines_desc.insert(0, "ID", "StatTranVEINES_01_desc")
df_tab_veines_desc.rename(columns={"timepoint": "Timepoint"}, inplace=True)

print (df_tab_veines_desc)


## **Table 4 – VCSS descriptive (Median [IQR])**
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

print (df_tab_vcss_desc)


## **Table 5 – VCSS change vs baseline (mock model-style)**

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
print(df_tab_vcss_change)


## **Table 3 – VEINES model-style table (mock EMM + CI)**

# ⚠️ For now, this uses **group means ± 1.96·SE** as a placeholder
# (later we’ll replace with `statsmodels` EMMs)

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
print(df_tab_veines_model)
