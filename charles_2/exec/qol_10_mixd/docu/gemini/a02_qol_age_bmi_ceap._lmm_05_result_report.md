To finalize your Tier 2 analysis, here is the formal **Statistical Methods** section and the corresponding **LaTeX** framework. This write-up is designed to satisfy a peer-reviewer or a thesis committee by clearly justifying the move from RM-ANOVA to LMM.

---

## 1. Statistical Methods (Draft)

> **Statistical Analysis:** Primary inferential analysis of the longitudinal VEINES-QOL scores was performed using a **Linear Mixed-Effects Model (LMM)** to account for the nested structure of the data (repeated measurements within individual patients).
> The model included **Time (Months)**, **BMI**, and **CEAP Clinical Stage** as fixed effects. To account for baseline heterogeneity and individual recovery trajectories, a **random intercept** was specified for each patient. Parameters were estimated using **Restricted Maximum Likelihood (REML)**, which provides unbiased estimates of variance components.
> Unlike traditional repeated-measures ANOVA, the LMM approach utilized all available data points without the need for listwise deletion or imputation, maintaining the **Intent-to-Treat (ITT)** principle despite patient attrition. A -value of  was considered statistically significant. All analyses were conducted in Python 3.x using the `statsmodels` library.

---

## 2. Formal Model Specification (LaTeX)

In your report, you should define the model mathematically. The following LaTeX represents the structure of the model you just ran:

The VEINES-QOL score  for patient  at time  is modeled as:

**Where:**

* : The fixed intercept (reference: CEAP C1).
* : The **Random Intercept** for patient  (the "Group Var" in your output).
* : The fixed effect of **Time** (rate of QoL change per month).
* : The fixed effect of **BMI**.
* : The partial regression coefficients for **CEAP stages** C2 through C6.
* : The residual error.

---

## 3. Visualizing the Coefficients

To present your "Run 1" and "Run 2" results effectively, a **Forest Plot** is superior to a table. It allows the reader to quickly see that while C2 might be "unstable" (crossing the zero line), C4–C6 and BMI are consistently significant.

### Quick Summary of your findings for the Discussion:

1. **CEAP Sensitivity:** The model confirms that CEAP C4 (skin changes) is the "tipping point" for significant QoL degradation.
2. **BMI Stability:** Regardless of the clinical stage, BMI acts as a constant "weight" on the patient's recovery, with a stable coefficient of .
3. **Variance Recovery:** The high "Group Var" () relative to the residual error () confirms that clinical outcomes are highly personalized, justifying the use of random effects.

---

Would you like me to help you generate the **Python code to produce a Forest Plot** of these specific results for your presentation?