Since you are following the evolution of **VEINES-QOL** (Quality of Life) and **VEINES-Sym** (Symptom frequency), the standard reporting method focuses on the **T-score**.

Here is how to structure the reporting for a scientific publication using your Linear Mixed-Effects Model (LMM) approach.

---

## 1. The Metric: Why T-Scores?

The VEINES-QOL/Sym instruments are validated using **T-score transformation**.

* **Mean of 50:** The scores are normalized so that 50 represents the average of the study population (usually the baseline or a validated reference population).
* **Standard Deviation of 10:** A 10-point difference represents one standard deviation.
* **Directionality:** Higher scores always indicate **better** outcomes (better QOL or fewer symptoms).

## 2. Statistical Reporting (Step 1)

Using your model: `smf.mixedlm("iter_t ~ C(timepoint)", groups=df_modl["patient_id"])`, you are treating the T-score as a continuous dependent variable.

### In the "Methods" Section

You should specify that:

> "VEINES-QOL and VEINES-Sym scores were transformed into standardized T-scores (Mean=50, SD=10). Changes over time were analyzed using a **Linear Mixed-Effects Model**, treating timepoint as a fixed effect and patient ID as a random effect to account for the longitudinal nature of the data and potential missing values at follow-up (T1, T2)."

### In the "Results" Section

Do not just report p-values. Report the **Estimated Marginal Means (EMM)** or the **Coefficients**.

| Measure | T0 (Baseline) | T1 (Change ) | T2 (Change ) | p-value (Trend) |
| --- | --- | --- | --- | --- |
| **VEINES-QOL** |  |  |  |  |
| **VEINES-Sym** |  |  |  |  |

## 3. Interpreting "Clinical Significance"

Reviewers will look for the **MCID (Minimal Clinically Important Difference)**. For VEINES-QOL, a change of **4 to 5 points** in the T-score is generally cited as the threshold for a "clinically meaningful" improvement.

* **Tip:** In your discussion, emphasize that your model showed an improvement exceeding this 5-point threshold at T1 and T2.

---

## 4. Visualizing QOL Evolution

The best way to visualize this is a **Profile Plot (Line Graph)** showing the mean T-score at each timepoint with Error Bars (95% Confidence Intervals).

* **Why a line graph?** It clearly illustrates the "trajectory" of recovery. For RFA, you typically see a steep incline from T0 to T1, followed by a plateau or slight further increase at T2.

---

## 5. Handling R/L Limbs in QOL

Unlike VCSS or CEAP, **VEINES-QOL is patient-reported**, not limb-reported.

* **Crucial Distinction:** Even if you performed RFA on both limbs, the patient only fills out **one** questionnaire.
* **Standard Practice:** If a patient had bilateral surgery, you do not "double count" their QOL score. You link the QOL score to the `patient_id`.
* **Your Model:** Your use of `groups=df_modl["patient_id"]` correctly handles this. If you have bilateral data, ensure your dataframe treats the QOL score as a single entry per timepoint per patient.

**Would you like to move to Step 2: Detailing the VCSS "Limb-Specific" reporting?**