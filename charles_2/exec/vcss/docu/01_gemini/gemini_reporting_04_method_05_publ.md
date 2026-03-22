To provide a high-level "Methods" section for your publication, you must clearly distinguish between how you handled **Patient-level** data (QOL) and **Limb-level** data (VCSS and CEAP).

Here is the detailed "Statistical Methods" draft and the breakdown of how to report them.

---

## 1. Drafted Statistical Methods Paragraph

This text is written in standard academic style for a medical journal:

> "Continuous variables were expressed as Mean  Standard Deviation (SD) or Median (Interquartile Range [IQR]) based on normality testing. The evolution of patient-reported quality of life (**VEINES-QOL/Sym**) and clinical severity (**VCSS**) was analyzed using **Linear Mixed-Effects Models (LMM)** to account for the longitudinal structure of the data and to handle missing values at follow-up timepoints (T1, T2) without excluding patients (listwise deletion). For VCSS, the model included `patient_id` as a random effect to adjust for the correlation between right and left limbs in bilateral cases.
> Changes in the **CEAP** clinical classification were treated as categorical ordinal data. Redistribution across C-classes was visualized via **stacked bar charts**, and the significance of clinical migration from baseline to follow-up was assessed using the **Stuart-Maxwell test** (or Wilcoxon Signed-Rank test for median class shift). A p-value of < 0.05 was considered statistically significant."

---

## 2. Reporting Details for CEAP (Step 3 Summary)

Since you are analyzing Right and Left limbs separately at T0, T1, and T2, your CEAP reporting should follow these "real-world" standards:

### The Frequency Table

Instead of means, present the "n (%)" for each class.
| CEAP Class | T0 (n=XX limbs) | T1 (n=XX limbs) | T2 (n=XX limbs) |
| :--- | :--- | :--- | :--- |
| **C2** | 45 (60%) | 10 (13%) | 5 (7%) |
| **C3** | 20 (27%) | 5 (7%) | 2 (3%) |
| **C4** | 10 (13%) | 2 (3%) | 1 (1%) |
| **C0-C1** | 0 (0%) | 58 (77%) | 67 (89%) |

### The "Clinical Downgrade" Metric

In your Results text, emphasize the **success rate**:

* *"At T2, 89% of limbs showed at least a one-class improvement in the CEAP clinical scale."*
* This is the "pseudo-standard" that surgeons use to quickly communicate the efficacy of RFA.

---

## 3. Data Integrity: Handling the "R/L" Distinction

Because you tracked R and L limbs, you have two options for the final tables:

1. **Pooled Analysis:** Combine R and L limbs into one "Total Limbs" pool (). This is the most common approach in RFA papers.
2. **Sub-group Analysis:** Only necessary if you performed different procedures on the Right vs. Left (e.g., RFA on one, sclerotherapy on the other). Since you are likely reporting on RFA evolution, **Pooled Analysis** (with your Mixed Model) is the most robust way to go.

---

## Final Check of Your Publication Strategy

| Metric | Level | Statistical Tool | Visual |
| --- | --- | --- | --- |
| **VEINES-QOL** | Patient | LMM (Random intercept: Patient) | Line Graph |
| **VCSS** | Limb | LMM (Random intercept: Patient) | Box Plot |
| **CEAP** | Limb | Stuart-Maxwell / Wilcoxon | Stacked Bar Chart |

### Next Step

Would you like me to help you format a **Master Results Table** that combines the T0, T1, and T2 findings for all three metrics in a way that is ready for a manuscript submission?

****
****
You are correct. In the specific field of Patient-Reported Outcome Measures (PROMs) like **VEINES-QOL/Sym**, there is a strong convention to report **Mean and Standard Deviation (SD)** regardless of the underlying distribution.

This "standard" exists for two main reasons:

1. **Comparability:** Almost all validation studies and historical RFA trials report VEINES in Mean/SD. Using Medians would make it impossible for readers to compare your results with the existing literature.
2. **The T-Score Nature:** Since T-scores are themselves a product of a linear transformation intended to normalize data (), reporting them via Mean/SD is mathematically consistent with the design of the instrument.

---

## Refined Statistical Methods for QOL

You should adjust your "Methods" section to reflect this field-specific standard. Here is the revised wording:

> "VEINES-QOL and VEINES-Sym scores were transformed into standardized T-scores. Following convention for validated quality-of-life instruments, these are presented as **Mean  SD** to allow for comparison with established normative data. Clinical improvement was modeled using a Linear Mixed-Effects approach, which is robust to deviations from normality in large clinical samples."

---

## Updated Reporting Table (Step 1 & 2)

By using Mean/SD for both QOL and VCSS, your results table becomes much cleaner and more "standardized" for a vascular surgery journal:

| Metric | Level | T0 (Baseline) | T1 | T2 |  (T2-T0) | p-value* |
| --- | --- | --- | --- | --- | --- | --- |
| **VEINES-QOL** | Patient |  |  |  |  |  |
| **VEINES-Sym** | Patient |  |  |  |  |  |
| **VCSS** | Limb |  |  |  |  |  |

**p-value for the effect of time in the Linear Mixed-Effects Model.*

---

## A Note on VCSS Normality

While QOL has a "free pass" for Mean/SD, some strict reviewers might still ask about VCSS distribution since it is a clinical score.

* **Pro-tip:** If your VCSS data is slightly skewed (which it often is at T2, as many patients hit "0"), you can still report the **Mean  SD** in your main table for readability, but add a small footnote: *"VCSS distribution was assessed via Shapiro-Wilk; despite slight skewness at follow-up, means are reported to maintain consistency with historical surgical reporting."*

---

## Summary of your Reporting Standards

1. **VEINES-QOL/Sym:** Always **Mean  SD** (Standard for T-scores).
2. **VCSS:** **Mean  SD** (Standard for surgical "evolution" papers).
3. **CEAP:** **n (%)** and **Shift Table** (Standard for categorical classification).

Would you like me to generate the **Python code** (using `statsmodels`) to extract these specific Mean/SD values and p-values from your `mixedlm` results for your final table?