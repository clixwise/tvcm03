import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# -----------------------------
# MOCK DATA (same as before)
# -----------------------------

np.random.seed(42)

n_patients = 50
time_points = ["Baseline", "3m", "12m"]

data = []

for patient in range(n_patients):
    baseline = np.random.normal(45, 8)
    m3 = baseline + np.random.normal(10, 4)
    m12 = baseline + np.random.normal(14, 5)
    
    values = [baseline, m3, m12]
    
    for t, value in zip(time_points, values):
        data.append([patient, t, value])

df = pd.DataFrame(data, columns=["Patient", "Time", "VEINES_QOL"])
df["Time"] = pd.Categorical(df["Time"], categories=time_points)

# -----------------------------
# LMM (Baseline reference)
# -----------------------------

model = smf.mixedlm("VEINES_QOL ~ C(Time, Treatment(reference='Baseline'))",
                    df,
                    groups=df["Patient"])
result = model.fit()

print(result.summary())

# -----------------------------
# Extract Model Estimates
# -----------------------------

params = result.params
conf = result.conf_int()

intercept = params["Intercept"]
ci_intercept = conf.loc["Intercept"]

change_3m = params["C(Time, Treatment(reference='Baseline'))[T.3m]"]
ci_3m = conf.loc["C(Time, Treatment(reference='Baseline'))[T.3m]"]

change_12m = params["C(Time, Treatment(reference='Baseline'))[T.12m]"]
ci_12m = conf.loc["C(Time, Treatment(reference='Baseline'))[T.12m]"]

# Estimated means
mean_baseline = intercept
mean_3m = intercept + change_3m
mean_12m = intercept + change_12m

summary = pd.DataFrame({
    "Time": ["Baseline", "3m", "12m"],
    "Estimated Mean": [mean_baseline, mean_3m, mean_12m],
    "Mean Change vs Baseline": [0, change_3m, change_12m],
    "Lower CI Change": [0, ci_3m[0], ci_12m[0]],
    "Upper CI Change": [0, ci_3m[1], ci_12m[1]]
})

print(summary)

# -----------------------------
# Publication-Grade Figure
# -----------------------------

plt.figure()
plt.plot(summary["Time"], summary["Estimated Mean"], marker='o')

# CI bars for estimated means (approximate using delta method)
# For simplicity we show CI for changes only here

plt.xlabel("Time")
plt.ylabel("VEINES-QOL Score")
plt.title("Model-Estimated VEINES-QOL Over Time")
plt.show()
