This is an **excellent and very important question**, and reviewers *do* notice inconsistencies here. I’ll answer it in a **principled but pragmatic** way, aligned with medical publishing standards, and then give you **clear rules** you can apply consistently across **tables** and **figures**.

---

## Core principle (one sentence)

> **Use SD / IQR to describe variability in the sample; use CI to express precision of an estimate.**

Everything else follows from this.

---

## 1️⃣ Tables — what to report and why

### A. Descriptive tables (baseline & raw evolution)

These answer: *“What did we observe in this sample?”*

#### VEINES QOL / Sym

* Continuous, approximately normal
* Sample description matters

✅ **Report**:

* **Mean ± SD** (primary)
* Optionally: Median [IQR] if distribution is skewed

❌ Avoid CI here unless table is explicitly labelled as *model-based estimates*

**Example**:

```
VEINES-QOL score, mean ± SD
```

---

#### VCSS

* Ordinal, bounded, often skewed

✅ **Report**:

* **Median [IQR]** (primary)
* Mean ± SD may be added secondarily if commonly used in your field

❌ CI not required in descriptive tables

**Example**:

```
VCSS, median [IQR]
```

---

### B. Tables of change / comparison / inference

These answer: *“How precise is the estimated effect?”*

✅ **Report**:

* Mean change from baseline
* **95% CI of the mean change**
* p-value (if applicable)

This applies to **both VEINES and VCSS**, *if you model them*.

**Example**:

```
Mean change vs T0 (95% CI)
```

---

### 📌 Recommended table structure

| Purpose            | Statistic                                |
| ------------------ | ---------------------------------------- |
| Raw description    | Mean ± SD (VEINES) / Median [IQR] (VCSS) |
| Change vs baseline | Mean change (95% CI)                     |
| Model-based        | Estimated mean (95% CI)                  |

---

## 2️⃣ Plots — what to show and why

### A. Plots should almost always use **CI**, not SD

**Why?**

* SD reflects individual variability → visually misleading
* CI reflects uncertainty of the mean → interpretable

This is now standard in clinical journals.

---

### B. VEINES QOL / Sym plots

#### Raw evolution plot

* X-axis: timepoint
* Y-axis: mean VEINES score
* Error bars: **95% CI**

This communicates:

> “The estimated group mean and its precision over time”

✔️ This is appropriate even for “raw” summaries

---

#### Model-based plots

* Estimated marginal means
* **95% CI only**

❌ Never SD on model plots

---

### C. VCSS plots

#### Raw VCSS

Two acceptable approaches:

**Option 1 (most common)**

* Plot **median**
* Error bars: **IQR**

**Option 2 (if aligning with VEINES visuals)**

* Plot **mean**
* Error bars: **95% CI**
* Explicitly state this choice in Methods/figure legend

✔️ Both are acceptable if consistent and justified
❌ Mixing median with CI is not acceptable

---

### D. Delta plots (Δ vs T0)

For **both VEINES and VCSS**:

✅ Plot:

* Mean change from baseline
* **95% CI**

This is the cleanest way to compare instruments visually.

---

## 3️⃣ Consistency rules (copy-paste checklist)

### Tables

* **Descriptive tables**

  * VEINES → Mean ± SD
  * VCSS → Median [IQR]
* **Inferential / change tables**

  * Mean change (95% CI)

### Plots

* Never SD in plots unless explicitly showing variability
* Prefer **95% CI**
* Median → IQR
* Mean → CI
* Model-based → CI only

---

## 4️⃣ How reviewers typically react

| Practice                | Reviewer reaction                      |
| ----------------------- | -------------------------------------- |
| SD in plots             | “Why SD and not CI?”                   |
| CI in descriptive table | “Unclear what estimate this refers to” |
| Median + CI             | ❌ Methodological error                 |
| Mean + CI               | ✔️ Accepted                            |
| Median + IQR            | ✔️ Accepted                            |

---

## 5️⃣ Practical recommendation for *your* study

# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
To stay **maximally consistent** across QOL and VCSS:

### Tables

* Raw descriptive:

  * VEINES → Mean ± SD  [OK 2026-01-19]
  * VCSS → Median [IQR] [OK 2026-01-19]
* Change / model:

  * Mean change (95% CI) [OK 2026-01-19]

### Figures

* Evolution plots:

  * Mean ± 95% CI (both VEINES and VCSS) [OK 2026-01-19]
* Supplementary (optional):

  * Median ± IQR for VCSS [NOT DONE]

This keeps the **story aligned**, without violating statistical principles.
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
---

Next step, when you’re ready:
👉 we can build a **mock Results section** (text + figure legend) using this exact logic, before touching any code.

# ****
#
# ****