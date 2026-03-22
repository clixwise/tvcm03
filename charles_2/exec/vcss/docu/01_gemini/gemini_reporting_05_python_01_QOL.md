To report **VEINES-QOL** and **VEINES-Sym** for a scientific publication, you should focus on the transition from raw scores to standardized **T-scores**. This is the international "real standard" for this instrument.

Here is the detailed breakdown for Step 1.

---

### 1. Standardization: The T-Score Logic

The VEINES instrument is validated using a T-score transformation. You should report these as continuous variables.

* **The Scale:** The scores are typically normalized to a **Mean of 50** and a **Standard Deviation (SD) of 10**.
* **Interpretation:** A higher score always indicates **better outcomes** (higher QOL or fewer/less frequent symptoms).
* **The "5-Point Rule":** In your discussion, you can cite that a change of ** points** is widely accepted as the **Minimal Clinically Important Difference (MCID)** for the VEINES scale.

### 2. Statistical Approach: Linear Mixed-Effects (LMM)

As you noted, the `smf.mixedlm` model is the superior choice here. In your publication, you must explain *why* you used it:

* **Handling Missing Data:** It retains patients who might have missed T1 but returned for T2 (unlike ANOVA).
* **Patient-Level Analysis:** Since QOL is a "per-patient" metric (unlike the "per-limb" VCSS), your `groups=df_modl["patient_id"]` ensures the model tracks the individual's improvement over time.

### 3. Reporting Format (The "Standard" Table)

In the medical literature, VEINES-QOL/Sym are almost **universally reported as Mean  SD**, even if the distribution is slightly non-normal. This allows for direct comparison with other RFA trials (e.g., the ATTRACT or EVRA trials).

**Suggested Table Layout:**
| Metric | T0 (Baseline) | T1 (Short-term) | T2 (Long-term) |  (T2-T0) | p-value* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **VEINES-QOL** |  |  |  |  |  |
| **VEINES-Sym** |  |  |  |  |  |

**p-value represents the fixed effect of "Timepoint" from your Mixed Model.*

### 4. Visualization: The Profile Plot

For QOL, a **Line Graph (Profile Plot)** is the standard visual. It illustrates the "slope" of recovery.

* **X-axis:** T0, T1, T2.
* **Y-axis:** T-score (usually restricted to the 30–70 range to show detail).
* **Error Bars:** Use **95% Confidence Intervals (CI)** rather than SD for the graph, as it more accurately represents the precision of your Mixed Model estimates.

---

### 5. Writing the "Methods" Snippet

You can adapt this text for your manuscript:

> "Quality of Life was assessed using the VEINES-QOL/Sym questionnaire, with raw scores transformed into standardized T-scores (Mean=50, SD=10). To evaluate the evolution across timepoints (T0, T1, T2), a Linear Mixed-Effects Model was employed, treating the timepoint as a fixed effect and the patient ID as a random effect. This model was chosen for its robustness to missing data at follow-up. Following field-specific standards, results are reported as Mean  SD."

**Would you like to move to Step 2: Detailing the VCSS "per-limb" reporting and how to handle the R vs. L data in your Mixed Model?**