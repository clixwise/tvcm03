
https://chatgpt.com/c/693e6518-a904-8325-8023-774a660fcc0d
Let’s formalize a production-ready function that handles all pre-model VEINES-QOL/Sym assumptions:
A1 — completeness (pati_isok)
A2 — score distribution (histogram, Q–Q, skew/kurtosis)
C1 — ceiling/floor effects

The function will loop over all timepoints, be fully reproducible, and produce:
Summary tables (completeness + ceiling/floor)
Histogram + Q–Q plots per timepoint
Skew/kurtosis numbers
---
Run this before any mixed-effects modelling
Inspect plots for obvious skew / ceiling/floor
Report one sentence in Methods:
“Pre-model diagnostics included assessment of questionnaire completeness, score distributions, and ceiling/floor effects at each timepoint.”