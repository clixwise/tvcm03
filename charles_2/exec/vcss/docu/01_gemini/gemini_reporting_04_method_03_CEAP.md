For **Step 3: CEAP Evolution**, we move away from "Means" and "T-Scores." Because CEAP is a classification system, the medical literature demands that you show the **migration** of limbs from more severe classes to less severe ones.

Here is how to detail the CEAP reporting for your publication.

---

### 1. The Denominator: Number of Limbs ()

Unlike QOL (which is per patient), CEAP must be reported **per limb**.

* If you have 50 patients and 20 of them had bilateral RFA, your CEAP reporting should be based on ** limbs**.
* In your table, you should list  for both **Right (R)** and **Left (L)** limbs to show symmetry in your study.

### 2. The Standard Visual: The "Stacked Bar Chart"

This is the most common way to visualize CEAP evolution. Each bar represents a timepoint (T0, T1, T2), and the segments of the bar represent the percentage of limbs in each "C" class.

* **What it shows:** You want to see the "warm" colors (C4, C5, C6) shrinking and the "cool" colors (C0, C1, C2) expanding as you move from T0 to T2.

### 3. The "Shift Table" (Transition Matrix)

Reviewers love this because it provides "granularity." It shows exactly what happened to the C4 patients, for example.

**Example Table for your Results:**
| T0 Baseline | → | T2 Follow-up: C0-C1 | T2 Follow-up: C2 | T2 Follow-up: C3 | T2 Follow-up: C4-C6 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **C2 (Varicose)** | () | 22 | 8 | 0 | 0 |
| **C3 (Edema)** | () | 5 | 12 | 3 | 0 |
| **C4 (Skin changes)**| () | 0 | 4 | 2 | 4 |

* **Interpretation:** In this example, out of 30 limbs that started at C2, 22 "downgraded" to C0 or C1 after RFA.

### 4. Statistical Analysis for CEAP

Since you cannot use your Linear Mixed Model for categorical data, you have two options for the "Statistics" section of your paper:

1. **Wilcoxon Signed-Rank Test:** While CEAP is categorical, the "C" classes are **ordinal** (C6 is worse than C2). You can use this test to compare the "median class" at T0 vs T2.
2. **Stuart-Maxwell Test:** This is the specific test for "change in proportions" in a paired  table. It tells you if the overall shift from T0 to T2 is statistically significant.
3. **Binary Simplification:** Report the **"Success Rate"** as the percentage of limbs that moved down at least one clinical class (e.g., *"85% of limbs showed a CEAP clinical class downgrade at T2"*).

### 5. Nuance: The C5 "Floor"

In your discussion, you must mention the **"Once a C5, always a C5"** rule.

* If a patient has a history of a healed ulcer (C5), even if their leg looks perfect after RFA, they remain C5.
* Therefore, you should report that for C5/C6 patients, "clinical improvement" is defined as **ulcer healing** or **lack of recurrence**, rather than a shift to C0.

---

### Final Publication Summary Table (Combined)

When you draft your final manuscript, your "Table 1" or "Results Overview" will likely look like this:

| Metric | T0 (Baseline) | T1 (Short-term) | T2 (Long-term) | Analysis Type |
| --- | --- | --- | --- | --- |
| **VEINES-QOL** |  |  |  | Mixed Model (T-score) |
| **VCSS** |  |  |  | Mixed Model (Limb score) |
| **CEAP (Class)** | Mostly C3-C4 | Mostly C2 | Mostly C1-C2 | Shift Table (Categorical) |

**Would you like me to help you draft the specific "Statistical Methods" paragraph that explains how you handled the different natures of these three scores?**