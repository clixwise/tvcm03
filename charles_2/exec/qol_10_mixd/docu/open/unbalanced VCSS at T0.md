# ****
#
# ****

I'd like to pinpoint this comment (prior) :

"Most vascular journals accept linear mixed models for VCSS"

How do authors cope with my following remark :

Clinically, we observe at _T0_ (the start of our longitudinal study where patients are undergoing surgery treatment), that patients might present either 'balanced' or 'unbalanced' VCSS scores ; f.i. a patient might have an ulcer (L) and no diseased leg at R . Another one might exhibit 'balanced' disease scores. VCSS scores range between 0 thru 30 and any 'continuous' R,L combination might present itself.    

So... does modeling apply in this context of cohorts with quite varying leg 'profiles'

# ****
#
# ****
This is a **very good** question — and exactly the kind of conceptual point sharp reviewers *do* think about but rarely articulate clearly. The short answer is: **yes, modelling applies**, and the way authors “cope” is by being explicit about **what is being modelled** and **what is not**.

Let me unpack this cleanly, in reviewer-grade language.

---

## 1️⃣ The core misunderstanding to defuse (clinically vs statistically)

Your clinical observation is absolutely correct:

* At **T0**, patients may have:

  * unilateral disease (e.g. ulcer on L, R ≈ 0),
  * bilateral but asymmetric disease,
  * bilateral and roughly symmetric disease.
* VCSS is **bounded (0–30)** and **semi-discrete**, with heterogeneous limb profiles.

The **key point** is:

> **Mixed-effects models do *not* assume clinical symmetry or homogeneity of limb profiles at baseline.**

They assume:

* a **mean structure** for change over time,
* and a **correlation structure** for repeated measurements.

Baseline heterogeneity is *expected*, not problematic.

---

## 2️⃣ What exactly is being modelled in your VCSS mixed model

Your model:

```text
VCSS ~ timepoint + limb + (1 | patient)
```

means:

### Fixed effects

* **Timepoint**: average change in VCSS over time *across all limbs*
* **Limb**: average systematic difference between L and R limbs (if any)

### Random effect

* **Patient intercept**: each patient has their own baseline VCSS “level”

This random intercept is *precisely* what absorbs:

* unilateral vs bilateral disease,
* severe vs mild baseline burden,
* asymmetric leg profiles.

So the model **does not require**:

* similar baseline VCSS across patients,
* similar L/R profiles,
* balanced disease at T0.

Instead, it **explicitly allows** all of that.

---

## 3️⃣ Why this is accepted in vascular journals

Most vascular and outcomes journals accept this because:

### 🔹 VCSS is treated as a *quasi-continuous clinical severity index*

* This is long-established (VCSS ≠ ordinal staging)
* Mean changes are clinically interpretable

### 🔹 The estimand is population-average change

You are not claiming:

> “Every limb improves equally.”

You are estimating:

> “On average, limbs show a reduction in VCSS over time.”

That distinction matters a lot.

### 🔹 Random effects reflect clinical reality

Patients are:

* heterogeneous,
* asymmetric,
* non-exchangeable across limbs.

Mixed models were designed for exactly this setting.

---

## 4️⃣ How authors *explicitly* address your concern in writing

Good papers do **not** ignore limb heterogeneity — they **name it**.

Here is the language typically used (and what reviewers expect).

### Methods-style clarification (recommended)

> VCSS was analysed at the limb level. Linear mixed-effects models were used to estimate longitudinal changes, with timepoint and limb as fixed effects and a patient-level random intercept to account for within-patient correlation and baseline heterogeneity, including asymmetric or unilateral disease.

This single sentence neutralises the concern.

---

## 5️⃣ What the model does **not** claim (important for you)

Your model does **not** claim:

* that left and right limbs are comparable at baseline,
* that improvement is symmetric,
* that baseline imbalance is noise.

It claims:

* that **time-related change** can be estimated *after accounting for* baseline heterogeneity.

That is the correct inferential target.

---

## 6️⃣ When modelling *would* become questionable

You’re right to ask — there *are* limits.

Modelling would be questionable if:

* > 70–80% of limbs were exactly zero at baseline (zero inflation),
* VCSS changes were extremely non-linear or jump-like,
* treatment targeted only one limb in most patients *and* contralateral limbs were analysed as equivalent outcomes.

In those cases, authors often:

* stratify by treated limb,
* or perform sensitivity analyses.

Your current mock (and likely real data) **do not** fall into that category.

---

## 7️⃣ How to pre-empt reviewer pushback (strong advice)

You already did the right thing by:

* making **raw descriptive statistics primary**, and
* using mixed models as **supportive inference**.

That combination is very defensible.

If challenged, your response is simply:

> “The mixed-effects model estimates population-level temporal trends while explicitly accounting for limb-level measurements nested within patients and substantial baseline heterogeneity.”

That sentence alone usually ends the discussion.

---

## Bottom line (clinically and statistically)

✔ Yes — modelling applies
✔ Baseline asymmetry is not a violation
✔ Mixed models are *designed* for this heterogeneity
✔ Your analytic strategy is conservative, not aggressive

If you want, next we can:

* draft a **reviewer response paragraph** to this exact concern, or
* discuss whether adding a **random slope for time** would ever be justified (often not).
