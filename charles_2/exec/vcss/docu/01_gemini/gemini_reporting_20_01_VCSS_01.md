This is a robust structure for reporting **VEINES-QOL**. To "fill in the gaps," we need to adapt this logic for **VCSS** and **CEAP**. While VCSS follows a similar Linear Mixed-Effects Model (LMM) path, CEAP requires a shift to categorical frequency and transition analysis.

Here is how to structure the code for the remaining two metrics.

---

## 1. VCSS: The Limb-Level Mixed Model

The code for VCSS is nearly identical to your QOL code, but with one crucial data distinction: **VCSS is per-limb**. If your `df_fram` contains one row per limb per timepoint, the `groups=df_mm["patient_id"]` parameter correctly handles the correlation of two limbs within the same patient.

### Key Logic Gaps to Fill:

* **Formula:** Change `iter_t` (QOL score) to `vcss_score`.
* **Interpretation:** While a higher QOL is better, a **lower VCSS** is better. Your `resu_plot` will show a descending line.

```python
def exec_stat_vcss_mixd(stat_tran_vcss: StatTranVCSS_mixd) -> None:
    df_mm = stat_tran_vcss.stat_tran.fram.copy()
    
    # Model: VCSS is the dependent variable
    # patient_id handles the nesting of limbs within patients
    model = smf.mixedlm("vcss_score ~ C(timepoint)", df_mm, groups=df_mm["patient_id"])
    result = model.fit(reml=True)
    
    # ... [Reuse your existing resu_glob and resu_deta extraction logic] ...

    # EMM (Estimated Marginal Means) for VCSS
    # This gives the "Average VCSS" at each timepoint adjusted for the random effect
    # ... [Reuse your EMM design matrix logic from QOL] ...

    # Exit
    stat_tran_vcss.resu_plot = df_plot  # Used for the VCSS descending box/line plot

```

---

## 2. CEAP: Categorical Frequency & Shift Matrix

You cannot use `mixedlm` for CEAP because it is categorical. Instead of "Means," we need **Frequency Tables** and a **Transition Matrix** (Shift Table).

### New Statistical Function: `exec_stat_ceap`

This logic fills the gap for categorical reporting.

```python
def exec_stat_ceap(stat_tran_ceap: StatTranCEAP) -> None:
    df_ceap = stat_tran_ceap.stat_tran.fram.copy()
    
    # Gap 1: Frequency Distribution (for the Stacked Bar Chart)
    # --------------------------------------------------------
    df_freq = (
        pd.crosstab(df_ceap["timepoint"], df_ceap["ceap_class"], normalize='index') * 100
    ).reindex(["T0", "T1", "T2"])
    
    # Gap 2: The Shift Table (T0 vs T2)
    # ---------------------------------
    # Pivot the data so each limb has T0 and T2 in the same row
    df_pivot = df_ceap.pivot(index="limb_id", columns="timepoint", values="ceap_class")
    shift_matrix = pd.crosstab(df_pivot["T0"], df_pivot["T2"])
    
    # Gap 3: Statistical Significance (Wilcoxon or Stuart-Maxwell)
    # ------------------------------------------------------------
    # For a simple narrative, many use the Wilcoxon signed-rank test on the C-ranks
    from scipy.stats import wilcoxon
    # Map C0-C6 to 0-6 integers for the test
    t0_ranks = df_pivot["T0"].str.extract('(\d+)').astype(int)
    t2_ranks = df_pivot["T2"].str.extract('(\d+)').astype(int)
    stat, p_val = wilcoxon(t0_ranks[0], t2_ranks[0])

    # Exit
    stat_tran_ceap.resu_freq = df_freq      # For resu_plot (Stacked Bar)
    stat_tran_ceap.resu_shift = shift_matrix # For Table narratives
    stat_tran_ceap.p_value = p_val         # Global evolution p-value

```

---

## 3. Summary of resu_plot Gaps

To ensure your plotting functions work consistently across all metrics, ensure the `df_plot` for each looks like this:

| Metric | `df_plot` Content | Visualization Recommendation |
| --- | --- | --- |
| **QOL** | Mean T-score + SE | **Line Graph** (Ascending) |
| **VCSS** | Mean VCSS + SE | **Boxplot** or Line Graph (Descending) |
| **CEAP** | % of each C-class | **100% Stacked Bar Chart** |

### Final Strategy for Text Narratives

Using your `resu_deta`, you can now programmatically generate sentences like:

* *“The intervention resulted in a significant improvement in QOL (T2-T0  = {beta_val}, p < .001).”*
* *“VCSS decreased by an estimated {vcss_beta} points by T2.”*

**Would you like me to detail the "Shift Table" generation specifically to ensure it handles the R and L limb mapping correctly?**