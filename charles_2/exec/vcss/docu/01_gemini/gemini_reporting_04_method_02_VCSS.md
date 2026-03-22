Since you are following the evolution of **VCSS** (Venous Clinical Severity Score) across three timepoints for both Right (R) and Left (L) limbs, the complexity increases because you have **nested data** (multiple limbs within one patient).

Here is how to detail the VCSS reporting for your publication.

---

## 1. The Metric: Why VCSS?

Unlike CEAP, the VCSS is a **summative score (0–30)** designed to detect small clinical changes. In an RFA study, it is usually your **Primary Clinical Outcome**.

## 2. Statistical Reporting: The Mixed Model

You should use the same `smf.mixedlm` approach you used for QOL, but with one critical adjustment in your data structure: **The Unit of Analysis.**

* **The Limb as the Unit:** For VCSS, each limb is its own "case." If a patient has RFA on both limbs, you have two sets of VCSS scores.
* **The Model:**
`model = smf.mixedlm("vcss_score ~ C(timepoint)", df_vcss, groups=df_vcss["patient_id"])`
* **Why this works:** By using `patient_id` as the group, the model understands that the Right and Left limbs of "Patient A" are more similar to each other than to "Patient B." It prevents you from "cheating" by artificially inflating your sample size ().

## 3. Data Presentation in Tables

In your results, provide a table that shows the clinical improvement clearly.

| Timepoint | Mean VCSS ( SD) | Range |  Coefficient (95% CI)* | p-value |
| --- | --- | --- | --- | --- |
| **T0 (Baseline)** | 7.8 ( 2.4) | 3–15 | *Reference* | - |
| **T1 (Post-Op)** | 3.1 ( 1.8) | 0–8 | -4.7 (-5.2, -4.2) | < 0.001 |
| **T2 (Follow-up)** | 2.2 ( 1.5) | 0–6 | -5.6 (-6.1, -5.1) | < 0.001 |

> **The  coefficient from your LMM represents the average point drop in VCSS compared to T0.*

---

## 4. Visualizing VCSS: The Boxplot

While QOL is best shown with line graphs to see "trends," VCSS is best shown with **Box-and-Whisker plots** at each timepoint. This allows reviewers to see:

* The **Median** (the line in the box).
* The **Variance** (how much patients differ).
* The **Outliers** (patients who did not improve much after RFA).

---

## 5. Reporting the "Success Rate"

In the text of your publication, it is highly recommended to report the **Clinical Success Rate** based on the VCSS:

* **Significant Improvement:** Percentage of limbs that improved by  points (the MCID).
* **Complete Resolution of Symptoms:** Percentage of limbs that reached a VCSS of 0 or 1 at T2.
* **Lateralization:** Briefly state if there was any significant difference in improvement between Right and Left limbs (though usually, there isn't, and they are pooled).

---

## 6. Scientific Notation Tip

When describing your VCSS results in the "Results" section, use this phrasing:

> *"The mean VCSS significantly decreased from  at baseline to  at T2 (). This reduction was maintained throughout the follow-up period, with  of limbs achieving a clinically meaningful improvement (reduction  points)."*

**Would you like to move to Step 3: Detailing the CEAP "Shift Table" and how to handle the categorical evolution?**