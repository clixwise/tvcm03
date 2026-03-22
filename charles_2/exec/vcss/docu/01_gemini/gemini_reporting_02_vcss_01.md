To report the **Venous Clinical Severity Score (VCSS)** effectively in a scientific publication, you must treat it as a dynamic tool. Unlike the CEAP classification, which is relatively static, VCSS was specifically designed to be "change-sensitive" to reflect the success of interventions like RFA.

Here is the breakdown of how to detail VCSS reporting according to academic standards.

---

## 1. Data Structure: Total Score vs. Individual Attributes

While the **Total VCSS** (sum of 10 attributes, range 0–30) is the most common metric, comprehensive papers often break it down further.

* **Total Score:** Reported as a continuous variable. Use **Mean (± SD)** if the data is normally distributed, or **Median (IQR)** if it is skewed.
* **Individual Attributes:** For RFA, reviewers are often interested in specific attributes like **Pain**, **Edema**, and **Varicose Veins**. These can be reported as the percentage of patients showing improvement in that specific category.

## 2. Statistical Comparison Across T0, T1, and T2

Since you are following the same limbs over three timepoints, you should use **repeated measures** statistics:

* **Primary Analysis:** Use a **Repeated Measures ANOVA** (parametric) or **Friedman Test** (non-parametric) to determine if there is a significant change across the entire timeline (T0  T1  T2).
* **Post-hoc Testing:** Use paired tests (e.g., **Paired t-test** or **Wilcoxon Signed-Rank**) to compare specific intervals:
* **T0 vs. T1:** Shows the immediate impact of the RFA.
* **T0 vs. T2:** Shows the durability of the clinical improvement.



## 3. Reporting the "Delta" () and MCID

In modern vascular literature, simply saying a result is "statistically significant" () is often considered insufficient. You should also report:

* **Absolute Change:** .
* **MCID (Minimal Clinically Important Difference):** For VCSS, a decrease of ** points** is generally accepted in the literature as a "clinically meaningful" improvement for the patient. Reporting the percentage of "Responders" (those who met the MCID) adds significant weight to your publication.

---

## 4. Visualizing VCSS Evolution

A **Box-and-Whisker Plot** is the standard for VCSS. It allows you to show the median, the interquartile range, and any outliers at T0, T1, and T2 on a single graph.

| Timepoint | Mean VCSS (± SD) | Median (IQR) | p-value (vs. T0) |
| --- | --- | --- | --- |
| **T0 (Baseline)** | 8.4 (± 2.1) | 8 (6–10) | - |
| **T1 (Post-Op)** | 3.2 (± 1.5) | 3 (2–4) | < 0.001 |
| **T2 (Follow-up)** | 2.1 (± 1.2) | 2 (1–3) | < 0.001 |

---

## 5. Handling R and L Limbs (The "N" Issue)

This is a common point of critique from reviewers. You must clarify if your  represents **Patients** or **Limbs**.

* **Recommendation:** Report results **per limb**. If a patient had bilateral RFA, both limbs are included.
* **Technical Note:** If you have many bilateral cases, acknowledge that you used a "marginal model" or "Generalized Estimating Equations (GEE)" to account for the fact that two limbs from the same person are not strictly independent.

**Would you like me to move on to the specific "Shift Table" methodology for reporting the CEAP evolution?**