For **Step 2: VCSS (Venous Clinical Severity Score)**, the reporting shifts from a "patient-centered" view to a "limb-centered" view. In RFA surgery, the limb is the biological unit of interest.

Here is how to detail the VCSS reporting, specifically addressing the Right (R) and Left (L) limb evolution.

---

### 1. The Denominator: Limbs vs. Patients

In your publication, you must be explicit about your .

* **Total Patients:** Used for QOL.
* **Total Limbs ():** Used for VCSS.
* **Reporting:** You should state: *"A total of X limbs in Y patients were analyzed."* ### 2. Statistical Nuance: The Nested Mixed Model
Since you are using `smf.mixedlm`, your code is already handling the "real" standard for bilateral data.
* **Random Effect:** By setting `groups=df_modl["patient_id"]`, the model acknowledges that a Right limb and a Left limb belonging to the same person are not independent. This is more scientifically rigorous than a simple t-test, which would "double-count" the patient’s systemic factors (like age or BMI).
* **Fixed Effect:** Timepoint (T0, T1, T2).

### 3. Reporting the VCSS Results

Unlike QOL, where only the total score is usually reported, VCSS is often reported as a **Total Score** accompanied by a mention of specific clinical improvements.

**Standard VCSS Results Table:**
| Timepoint | Mean VCSS ( SD) | Median (Range) | 95% CI of Mean | p-value (vs T0) |
| :--- | :---: | :---: | :---: | :---: |
| **T0 (Baseline)** |  |  | [7.9, 8.9] | — |
| **T1 (Post-Op)** |  |  | [2.8, 3.6] |  |
| **T2 (Follow-up)** |  |  | [1.6, 2.2] |  |

> **Note on Normality:** Even if VCSS isn't perfectly normal (especially at T2 where many scores cluster at 0-1), reporting the **Mean  SD** is the "pseudo-standard" in vascular journals to show the magnitude of change clearly.

### 4. Visualizing Evolution: Box-and-Whisker Plots

While QOL uses line graphs, VCSS evolution is best shown via **Boxplots**. This allows the reviewer to see the "floor effect" (the score dropping toward zero) and the reduction in variance after surgery.

---

### 5. Writing the "Results" Section for VCSS

When describing the evolution, use the "Delta" () to emphasize the surgical impact:

> "The mean VCSS showed a significant clinical improvement, decreasing from  at baseline to  at T2 (LMM estimate of change:  points; ). At the final follow-up,  of limbs achieved a reduction of  points, meeting the threshold for the Minimal Clinically Important Difference (MCID) in venous severity."

### 6. Special Handling of R and L Limbs

If you want to be extremely thorough for a high-impact journal, you can add a supplemental table or a brief sentence comparing the two sides:

* *"No significant difference was observed in VCSS improvement between right and left limbs ()."*
* This confirms to the reviewer that the RFA was equally effective regardless of laterality.

---

**Would you like to move to Step 3: Detailing the CEAP "Shift Table" and how to handle the categorical evolution (which requires a different statistical approach)?**