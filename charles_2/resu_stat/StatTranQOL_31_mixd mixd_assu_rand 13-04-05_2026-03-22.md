|   index | Component         | Assumption           | Metric                | Value           | Comment         |
|--------:|:------------------|:---------------------|:----------------------|:----------------|:----------------|
|       0 | Random Intercepts | Mean ≈ 0             | Mean                  | 0.0000          | ✓ Close to zero |
|       1 | Random Intercepts | Dispersion           | Std Dev               | 2.8802          | ✓ Reasonable    |
|       2 | Random Intercepts | Normality            | Shapiro-Wilk W=0.9715 | p=0.5799        | ✓ Normal        |
|       3 | Random Intercepts | Normality (Anderson) | A²=0.4034             | Critical=0.7120 | ✓ Normal        |
|       4 | Random Intercepts | Shape                | Skewness              | -0.2670         | ✓ Symmetric     |
|       5 | Random Intercepts | Shape                | Kurtosis              | -0.2199         | ✓ Mesokurtic    |