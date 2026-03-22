For **Step 3: CEAP Evolution**, we shift from numerical averages to **categorical redistribution**. CEAP is a classification (a "snapshot" of disease state), so reporting the "mean CEAP" is statistically incorrect. Instead, you must show how many limbs migrated from "severe" classes to "mild" classes.

Here is the detailed reporting standard for CEAP in a longitudinal study.

---

### 1. The Denominator and "Unit of Analysis"

Like VCSS, CEAP is **limb-specific**. In your publication, you must report based on the total number of limbs ().

* If a patient has a C2 on the Right and a C3 on the Left, they are counted twice in your CEAP distribution table.
* Ensure your results section states: *"CEAP clinical classification was assessed for all  limbs at each timepoint."*

### 2. The "Shift Table" (Transition Matrix)

This is the "Real Standard" for reporting CEAP. It allows a reviewer to see exactly what happened to the patients who started at a specific grade (e.g., C4).

**Example CEAP Shift Table (T0 to T2):**
| Baseline (T0) | → | T2: C0–C1 | T2: C2 | T2: C3 | T2: C4–C6 | Total |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **C2** | () | 25 | 5 | 0 | 0 | 30 |
| **C3** | () | 2 | 15 | 3 | 0 | 20 |
| **C4** | () | 0 | 4 | 2 | 4 | 10 |
| **Total** | | **27** | **24** | **5** | **4** | **60** |

* **Logic:** The "success" of RFA is visualized by the movement of numbers toward the left columns (lower C-classes).

### 3. Visualizing with Stacked Bar Charts

While VCSS uses boxplots, CEAP evolution is best shown with a **100% Stacked Bar Chart**. It provides a "color-coded" view of the population's recovery.

* **Y-axis:** Percentage of total limbs (0–100%).
* **X-axis:** T0, T1, and T2.
* **Segments:** Different colors for each C-class (e.g., Dark Red for C6, Light Blue for C1).

### 4. Statistical Analysis: Comparing Proportions

Because you cannot use a Linear Mixed Model for these categories, you should use non-parametric tests for "paired categorical data":

* **Wilcoxon Signed-Rank Test:** Often used to show that the *median* clinical class at T2 is significantly lower than at T0.
* **Stuart-Maxwell Test:** The gold standard for comparing the overall distribution of a  table (like the Shift Table above) to see if the change is significant.
* **Binary Simplification:** You can report the percentage of **"Clinical Downgraders"** (limbs that moved to any lower C-category).

### 5. The "C5/C6" Reporting Rule

In the medical literature, there is a specific nuance for venous ulcers:

* **C6 to C5:** This is considered a major surgical success (ulcer healing).
* **The C5 Floor:** Remember that a limb with a history of a healed ulcer (C5) remains C5 even if it looks normal after RFA. In your "Results," mention that for C5 patients, the goal was the **prevention of recurrence** rather than a shift to C0.

---

### 6. Writing the "Results" Section for CEAP

> "At baseline (T0), the majority of limbs were classified as C2 (50%) or C3 (33.3%). Following RFA, a significant downward shift in the CEAP clinical classification was observed at T2 (, Wilcoxon Signed-Rank). Specifically, 85% of limbs showed a clinical class reduction, with 45% of limbs reaching C0 or C1 status at the final follow-up."

---

### Summary of your 3-Step Reporting Strategy

Now that we have covered all three, your publication structure is complete:

1. **VEINES-QOL/Sym:** Mean  SD (T-scores) using **LMM**. (Visual: Line Graph).
2. **VCSS:** Mean  SD (Total Score) using **LMM** (Limb-level). (Visual: Boxplot).
3. **CEAP:** Frequencies/Percentages using **Shift Tables**. (Visual: Stacked Bar Chart).

**Would you like me to help you draft the final "Conclusion" or "Discussion" points regarding how these three scores correlate in your results?**