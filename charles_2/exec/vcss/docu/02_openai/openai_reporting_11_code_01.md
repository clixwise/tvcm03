Great — this is the right moment to **lock structure before real data**.
Below I do **exactly 1️⃣ and 2️⃣**, strictly conforming to the **Final coherence map** and your note.

I will:

* Create **mock long data**
* Derive **table-ready dataframes** (formatted strings, copy/paste ready)
* Derive **plot-ready dataframes** (numeric, no formatting)
* Keep **IDs compatible with functional pipelines**

No statistics will contradict your Methods.

---

# 1️⃣ Mock *long* datasets (starting point)

## A. VEINES (patient-level, long)

```text
df_veines_long
```

| patient_id | timepoint | VEINES_QOL | VEINES_Sym |
| ---------- | --------- | ---------- | ---------- |
| P01        | T0        | 47.2       | 46.1       |
| P01        | T1        | 55.4       | 54.2       |
| P01        | T2        | 58.1       | 56.8       |
| …          | …         | …          | …          |
| P30        | T2        | 60.3       | 59.0       |

Assumptions (mock):

* n = 30 patients
* VEINES ≈ normal
* Monotonic improvement

---

## B. VCSS (limb-level, long)

```text
df_vcss_long
```

| patient_id | limb | timepoint | VCSS |
| ---------- | ---- | --------- | ---- |
| P01        | R    | T0        | 9    |
| P01        | R    | T1        | 4    |
| P01        | R    | T2        | 2    |
| P01        | L    | T0        | 7    |
| P01        | L    | T1        | 3    |
| P01        | L    | T2        | 1    |
| …          | …    | …         | …    |

Assumptions:

* Limb-level correlation
* Ordinal bounded values
* Improvement post-RFA

---

# 2️⃣ Table-ready dataframes (formatted strings)

These are **publication-ready**, not for plotting.

---

## **Table 2 – VEINES descriptive (Mean ± SD)**

```text
df_tab_veines_desc
```

| ID                     | Metric     | Timepoint | Value      |
| ---------------------- | ---------- | --------- | ---------- |
| StatTranVEINES_01_desc | VEINES_QOL | T0        | 48.6 ± 5.1 |
| StatTranVEINES_01_desc | VEINES_QOL | T1        | 55.9 ± 4.8 |
| StatTranVEINES_01_desc | VEINES_QOL | T2        | 58.7 ± 4.5 |
| StatTranVEINES_01_desc | VEINES_Sym | T0        | 47.9 ± 5.3 |
| StatTranVEINES_01_desc | VEINES_Sym | T1        | 54.8 ± 4.9 |
| StatTranVEINES_01_desc | VEINES_Sym | T2        | 57.6 ± 4.6 |

✔️ Mean ± SD
✔️ Matches Table 2 shell
✔️ No CI here

---

## **Table 3 – VEINES model-based (EMM + Δ vs T0)**

```text
df_tab_veines_model
```

| ID                      | Metric     | Timepoint | Estimated Mean (95% CI) | Mean Change vs T0 (95% CI) |
| ----------------------- | ---------- | --------- | ----------------------- | -------------------------- |
| StatTranVEINES_02_model | VEINES_QOL | T0        | 48.6 [46.8–50.4]        | —                          |
|                         | VEINES_QOL | T1        | 55.7 [53.9–57.5]        | +7.1 [5.4–8.8]             |
|                         | VEINES_QOL | T2        | 58.9 [57.1–60.7]        | +10.3 [8.6–12.0]           |
|                         | VEINES_Sym | T0        | 47.9 [46.0–49.8]        | —                          |
|                         | VEINES_Sym | T1        | 54.6 [52.8–56.4]        | +6.7 [5.0–8.4]             |
|                         | VEINES_Sym | T2        | 57.4 [55.6–59.2]        | +9.5 [7.8–11.2]            |

✔️ CI only
✔️ Baseline change = “—”
✔️ Matches Table 3 shell

---

## **Table 4 – VCSS descriptive (Median [IQR])**

```text
df_tab_vcss_desc
```

| ID                   | Metric | Timepoint | Value          |
| -------------------- | ------ | --------- | -------------- |
| StatTranVCSS_01_desc | VCSS_R | T0        | 9.0 [6.0–12.0] |
|                      | VCSS_L | T0        | 7.0 [5.0–11.0] |
|                      | VCSS_R | T1        | 4.0 [3.0–6.0]  |
|                      | VCSS_L | T1        | 3.0 [2.0–5.0]  |
|                      | VCSS_R | T2        | 2.0 [1.0–3.0]  |
|                      | VCSS_L | T2        | 1.0 [1.0–2.0]  |

✔️ Median + IQR
✔️ Limb-level
✔️ Exactly your preferred format (improved clarity)

---

## **Table 5 – VCSS change vs baseline (Mean change + CI)**

```text
df_tab_vcss_change
```

| ID                    | Metric | Timepoint | Mean Change vs T0 (95% CI) |
| --------------------- | ------ | --------- | -------------------------- |
| StatTranVCSS_02_model | VCSS_R | T1        | −4.8 [−5.6 to −4.0]        |
|                       | VCSS_R | T2        | −6.9 [−7.7 to −6.1]        |
|                       | VCSS_L | T1        | −4.2 [−5.0 to −3.4]        |
|                       | VCSS_L | T2        | −6.1 [−6.9 to −5.3]        |

✔️ Negative = improvement
✔️ CI only
✔️ Matches Table 5 shell

---

# 3️⃣ Plot-ready dataframes (numeric only)

These are what you feed directly into plotting code.

---

## **Figure 1 – VEINES evolution (mean ± CI)**

```text
df_plot_veines
```

| Metric     | Timepoint | Mean | CI_lower | CI_upper |
| ---------- | --------- | ---- | -------- | -------- |
| VEINES_QOL | T0        | 48.6 | 46.8     | 50.4     |
| VEINES_QOL | T1        | 55.7 | 53.9     | 57.5     |
| VEINES_QOL | T2        | 58.9 | 57.1     | 60.7     |
| VEINES_Sym | T0        | 47.9 | 46.0     | 49.8     |
| VEINES_Sym | T1        | 54.6 | 52.8     | 56.4     |
| VEINES_Sym | T2        | 57.4 | 55.6     | 59.2     |

✔️ Mean + CI
✔️ One row = one plotted point

---

## **Figure 2 – VCSS model-based evolution**

```text
df_plot_vcss
```

| Limb | Timepoint | Mean | CI_lower | CI_upper |
| ---- | --------- | ---- | -------- | -------- |
| R    | T0        | 8.9  | 8.1      | 9.7      |
| R    | T1        | 4.1  | 3.6      | 4.6      |
| R    | T2        | 2.0  | 1.6      | 2.4      |
| L    | T0        | 7.3  | 6.6      | 8.0      |
| L    | T1        | 3.1  | 2.6      | 3.6      |
| L    | T2        | 1.2  | 0.9      | 1.5      |

✔️ Limb preserved
✔️ CI only
✔️ Ready for Figure 2

---

## **Figure 3 (optional) – Raw VCSS median/IQR**

```text
df_plot_vcss_raw
```

| Limb | Timepoint | Median | IQR_low | IQR_high |
| ---- | --------- | ------ | ------- | -------- |
| R    | T0        | 9      | 6       | 12       |
| R    | T1        | 4      | 3       | 6        |
| R    | T2        | 2      | 1       | 3        |
| L    | T0        | 7      | 5       | 11       |
| L    | T1        | 3      | 2       | 5        |
| L    | T2        | 1      | 1       | 2        |

---

# ✅ Final integrity check

* ✔️ Tables = formatted, publication-ready
* ✔️ Plots = numeric, CI/IQR correct
* ✔️ Limb correlation only handled in model outputs
* ✔️ No SD appears in plots
* ✔️ No CI appears in raw descriptive tables

---

### Next step (clean transition)

If you want, next we can:

* Write **exact pandas code** to generate each df from real data
* Align **figure aesthetics** with journal defaults
* Draft **Discussion limitations** (VCSS ordinal, modelling choice)

Just tell me where to go next.
