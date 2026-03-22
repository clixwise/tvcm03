|   index | Component   | Assumption           | Metric                | Value           | Comment             |
|--------:|:------------|:---------------------|:----------------------|:----------------|:--------------------|
|       0 | Residuals   | Mean ≈ 0             | Mean                  | 0.0000          | ✓ Close to zero     |
|       1 | Residuals   | Dispersion           | Std Dev               | 3.1290          | ✓ Reasonable        |
|       2 | Residuals   | Normality            | Shapiro-Wilk W=0.9701 | p=0.0478        | ⚠ Non-normal        |
|       3 | Residuals   | Normality (Anderson) | A²=0.8700             | Critical=0.7540 | ⚠ Non-normal        |
|       4 | Residuals   | Shape                | Skewness              | 0.0785          | ✓ Symmetric         |
|       5 | Residuals   | Shape                | Kurtosis              | 1.4215          | ⚠ Heavy/light tails |