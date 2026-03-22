Since CEAP is a **classification** and not a numerical score, reporting its evolution requires a different statistical mindset than VCSS or QOL. You cannot "average" a C2 and a C4 to get a C3. Instead, you must show the **redistribution** of patients across the classes.

Here is the standardized way to report CEAP evolution in a surgical publication.

---

## 1. The "Shift Table" (The Gold Standard)

A Shift Table (or Transition Matrix) is the most transparent way to show how limbs "move" between categories. It compares T0 (rows) to T1 or T2 (columns).

| T0 \ T2 | C0 | C1 | C2 | C3 | C4 | C5 | C6 | Total (T0) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **C2** | 10 | 5 | **15** | 0 | 0 | 0 | 0 | 30 |
| **C3** | 2 | 3 | 10 | **5** | 0 | 0 | 0 | 20 |
| **C4** | 0 | 1 | 4 | 2 | **3** | 0 | 0 | 10 |
| **Total (T2)** | 12 | 9 | 29 | 7 | 3 | 0 | 0 | **60 Limbs** |

* **Interpretation:** The diagonal (bold) shows limbs that stayed in the same class. Values to the **left** of the diagonal represent improvement (downgrading).

## 2. Visualizing Evolution: Stacked Bar Charts

In your results section, a **100% Stacked Bar Chart** is the most effective visual. It allows the reader to see the "shrinkage" of severe categories (C4–C6) and the growth of milder ones (C0–C2) over time.

## 3. Statistical Analysis for CEAP

Since you are using a Mixed Model for your continuous data, you need a different approach for this categorical data:

* **For Global Change:** Use the **Stuart-Maxwell test** or the **Symmetry Test**. These are the categorical equivalents of a paired t-test for  tables (where  is the number of CEAP categories).
* **For Simplified Reporting:** Many authors group the classes into **"Clinical Improvement"** (e.g., any limb that moved from  to ) and report this as a percentage.
* **The "C-Downgrade":** You can report the "Mean Clinical Class Reduction." While technically treating an ordinal scale as continuous, it is common in some journals to state: *"The median CEAP clinical class improved from C4 at T0 to C2 at T2 (p < 0.001)."*

---

## 4. Nuance: The "Once a C5, Always a C5" Rule

In strict CEAP reporting, some categories are permanent or have specific rules:

* **C5 (Healed Ulcer):** A patient who starts at C6 (Active Ulcer) can "improve" to C5.
* **C5 vs C2:** A C5 patient who has successful RFA and no longer has visible varicosities is still technically a C5 (history of healed ulcer), even if their VCSS is now 0.
* **Publication Tip:** If you see "improvement" in C5/C6 patients, clarify that you are tracking the **Clinical Class (C)** specifically.

---

## 5. How to report R and L Limbs for CEAP

Since CEAP is limb-specific, your  should be the total number of limbs.

* **Table Header:** "Distribution of CEAP Clinical Classification per Limb ()."
* **Separation:** If you have enough data, you can provide two tables (Right and Left), but usually, they are pooled into one "Limb Analysis" to increase statistical power, provided you mention the use of your Mixed Model (or GEE) to account for the patient-level clustering.

---

### Summary of your Publication Strategy

1. **VEINES-QOL:** Mixed Model (T-score evolution).
2. **VCSS:** Mixed Model (Mean/Median score reduction).
3. **CEAP:** Shift Table and Stacked Bar Chart (Categorical redistribution).

**Would you like me to help you write the "Statistical Methods" paragraph for your manuscript that covers all three of these approaches together?**