import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ****
#
# ****
def simulate_categorical_lmm(n_patients=50, timepoints=[0, 3, 6, 12]):
    data = []
    
    # Mapping for labels
    time_labels = {0: 'T0_Baseline', 3: 'T1_3Mo', 6: 'T2_6Mo', 12: 'T3_12Mo'}
    
    # Same generative parameters
    beta_0, beta_time, beta_bmi, beta_ceap = 60, 0.5, -0.7, -4.0
    sd_intercept, sd_slope = 5.0, 0.2

    for i in range(n_patients):
        bmi = np.random.normal(28, 5)
        ceap = np.random.randint(1, 7)
        u_0 = np.random.normal(0, sd_intercept)
        u_1 = np.random.normal(0, sd_slope)
        
        for t in timepoints:
            if t > 0 and np.random.random() < (t * 0.02):
                continue 

            error = np.random.normal(0, 2)
            # Math remains continuous for the generation, but labeling is categorical
            qol = (beta_0 + u_0) + (beta_time + u_1) * t + (beta_ceap * ceap) + (beta_bmi * bmi) + error
            
            data.append({
                'PatientID': f'P{i:03d}',
                'Visit': time_labels[t], # This is the key change
                'CEAP': f'C{ceap}',
                'BMI': round(bmi, 1),
                'VEINES_QOL': round(qol, 2)
            })

    return pd.DataFrame(data)

df = simulate_categorical_lmm()

# Fit the model using 'Visit' as a categorical factor

print(df)

# ****
#
# ****

# Define the model formula
# VEINES_QOL is the outcome
# Month, CEAP, and BMI are Fixed Effects
# (1 | PatientID) indicates a Random Intercept for each patient
model = smf.mixedlm("VEINES_QOL ~ Visit + CEAP + BMI", 
                    df, 
                    groups=df["PatientID"])

# Fit the model
result = model.fit()

# View the results
print(result.summary())

# ****
#
# ****


'''
    PatientID        Visit CEAP   BMI  VEINES_QOL
0        P000  T0_Baseline   C2  32.7       32.21
1        P000       T1_3Mo   C2  32.7       32.09
2        P000       T2_6Mo   C2  32.7       35.32
3        P000      T3_12Mo   C2  32.7       36.10
4        P001  T0_Baseline   C1  36.2       23.27
..        ...          ...  ...   ...         ...
172      P048       T1_3Mo   C3  20.8       37.92
173      P048      T3_12Mo   C3  20.8       45.50
174      P049  T0_Baseline   C1  35.2       27.55
175      P049       T1_3Mo   C1  35.2       25.07
176      P049      T3_12Mo   C1  35.2       29.24

[183 rows x 5 columns]

'''
'''
            Mixed Linear Model Regression Results
==============================================================
Model:              MixedLM   Dependent Variable:   VEINES_QOL
No. Observations:   182       Method:               REML      
No. Groups:         50        Scale:                5.5428    
Min. group size:    2         Log-Likelihood:       -469.7621 
Max. group size:    4         Converged:            Yes       
Mean group size:    3.6
--------------------------------------------------------------
                  Coef.  Std.Err.   z    P>|z|  [0.025  0.975]
--------------------------------------------------------------
Intercept         50.083    5.295  9.458 0.000  39.704  60.462
Visit[T.T1_3Mo]    1.912    0.474  4.030 0.000   0.982   2.841
Visit[T.T2_6Mo]    3.426    0.492  6.968 0.000   2.462   4.389
Visit[T.T3_12Mo]   6.466    0.513 12.604 0.000   5.461   7.472
CEAP[T.C2]        -5.197    2.187 -2.376 0.017  -9.484  -0.911
CEAP[T.C3]       -12.850    2.347 -5.476 0.000 -17.449  -8.251
CEAP[T.C4]       -14.520    2.688 -5.403 0.000 -19.788  -9.253
CEAP[T.C5]       -16.658    2.817 -5.913 0.000 -22.179 -11.137
CEAP[T.C6]       -24.744    2.598 -9.524 0.000 -29.836 -19.652
BMI               -0.384    0.177 -2.167 0.030  -0.732  -0.037
Group Var         22.915    2.592
==============================================================
'''
'''
            Mixed Linear Model Regression Results
==============================================================
Model:              MixedLM   Dependent Variable:   VEINES_QOL
No. Observations:   177       Method:               REML
No. Groups:         50        Scale:                4.4154
Min. group size:    2         Log-Likelihood:       -443.0625
Max. group size:    4         Converged:            Yes
Mean group size:    3.5
--------------------------------------------------------------
                  Coef.  Std.Err.   z    P>|z|  [0.025  0.975]
--------------------------------------------------------------
Intercept         66.552    5.388 12.351 0.000  55.992  77.113
Visit[T.T1_3Mo]    0.849    0.429  1.978 0.048   0.008   1.691
Visit[T.T2_6Mo]    2.355    0.447  5.268 0.000   1.479   3.231
Visit[T.T3_12Mo]   5.280    0.463 11.405 0.000   4.372   6.187
CEAP[T.C2]        -3.882    2.202 -1.763 0.078  -8.197   0.433
CEAP[T.C3]       -11.278    2.900 -3.889 0.000 -16.963  -5.594
CEAP[T.C4]       -14.475    2.388 -6.061 0.000 -19.155  -9.794
CEAP[T.C5]       -17.444    2.245 -7.769 0.000 -21.844 -13.043
CEAP[T.C6]       -21.424    2.368 -9.046 0.000 -26.066 -16.782
BMI               -1.035    0.167 -6.190 0.000  -1.363  -0.707
Group Var         22.705    2.864
==============================================================
'''