import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(123)

patients = [f"P{i:02d}" for i in range(1, 11)]
timepoints = ["T0", "T1", "T2"]
limbs = ["L", "R"]

rows = []

for pid in patients:
    base_left = np.random.randint(6, 10)
    base_right = base_left + np.random.randint(0, 3)

    for t, delta in zip(timepoints, [0, -3, -5]):
        for limb, base in zip(limbs, [base_left, base_right]):
            rows.append({
                "patient_id": pid,
                "timepoint": t,
                "Limb": limb,
                "VCSS": max(0, base + delta + np.random.normal(0, 1))
            })

df_vcss_long = pd.DataFrame(rows)

df_vcss_long["timepoint"] = pd.Categorical(
    df_vcss_long["timepoint"],
    categories=["T0", "T1", "T2"],
    ordered=True
)
df_vcss_long["Limb"] = df_vcss_long["Limb"].astype("category")

print (df_vcss_long.head())

# ----
#
# ----
def median_iqr(x):
    return f"{np.median(x):.1f} [{np.percentile(x,25):.1f}–{np.percentile(x,75):.1f}]"
def median_iqr_2(x):
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    return f"{x.median():.1f} [{q1:.1f}–{q3:.1f}]"
df_vcss_desc = (
    df_vcss_long
    .assign(Metric=lambda d: "VCSS_" + d["Limb"].astype(str))
    .groupby(["Metric", "timepoint"])["VCSS"]
    .apply(median_iqr)
    .reset_index(name="Value")
)
df_vcss_desc_2 = (
    df_vcss_long
    .assign(Metric=lambda d: "VCSS_" + d["Limb"].astype(str))
    .groupby(["Metric", "timepoint"])["VCSS"]
    .apply(median_iqr_2)
    .reset_index(name="Value")
)

print(df_vcss_desc)
print(df_vcss_desc_2)

# ----
#
# ----
model_vcss = smf.mixedlm(
    "VCSS ~ C(timepoint) + C(Limb)",
    df_vcss_long,
    groups=df_vcss_long["patient_id"]
)

res_vcss = model_vcss.fit(reml=True)
print(res_vcss.summary())

# ----
#
# ----
def extract_emm_vcss(result, timepoints):
    fe = result.fe_params.values
    cov_all = result.cov_params().values
    k = len(fe)
    cov = cov_all[:k, :k]

    rows = []

    # Average over Limb (L, R)
    limb_effect = fe[2] / 2 if len(fe) > 2 else 0

    for tp in timepoints:
        if tp == "T0":
            L = np.array([1, 0, 0])
        elif tp == "T1":
            L = np.array([1, 1, 0])
        elif tp == "T2":
            L = np.array([1, 0, 1])

        mean = float(L @ fe) + limb_effect
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


emm_vcss = extract_emm_vcss(res_vcss, ["T0", "T1", "T2"])
print(emm_vcss)
