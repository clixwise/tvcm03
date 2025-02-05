
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.stats.multicomp as multi
import scipy.stats as stats

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact


import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson, NegativeBinomial
from statsmodels.genmod.families.links import log

#
# Verified with 'pa28_agbi_popl_util'
#
def inpu():
   tabl_popu = pd.DataFrame({
      'Age bins': ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'],
      'Population': [33271000, 22681000, 15952000, 10641000, 6913000, 4924000, 2977000, 1420000, 348000, 21000] #'Population': [36428458, 25261830, 17503339, 11905491, 7606658, 5313156, 3293776, 1543467, 395992]

   })

   tabl_pati = pd.DataFrame({
      'Age bins': ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'], 
      'Patients': [0,      5,	      13,	   33,	   58,	   87,	   81,	   67,	   17,	   1]
   })
   tabl_popu['Population'] = tabl_popu['Population'] * 0.15
   print(f"\nStep 0 : df.size:{len(tabl_popu)} df2.type:{type(tabl_popu)}\n{tabl_popu}\n:{tabl_popu.index}\n:{tabl_popu.columns} sum:{tabl_popu['Population'].sum()}")
   print(f"\nStep 0 : df.size:{len(tabl_pati)} df2.type:{type(tabl_pati)}\n{tabl_pati}\n:{tabl_pati.index}\n:{tabl_pati.columns} sum:{tabl_pati['Patients'].sum()}")
   #
   tabl_popu = tabl_popu[tabl_popu['Age bins'] != '0-9'] ; tabl_popu = tabl_popu[tabl_popu['Age bins'] != '90-99']
   tabl_pati = tabl_pati[tabl_pati['Age bins'] != '0-9'] ; tabl_pati = tabl_pati[tabl_pati['Age bins'] != '90-99']
   print(f"\nStep 0 : df.size:{len(tabl_popu)} df2.type:{type(tabl_popu)}\n{tabl_popu}\n:{tabl_popu.index}\n:{tabl_popu.columns} sum:{tabl_popu['Population'].sum()}")
   print(f"\nStep 0 : df.size:{len(tabl_pati)} df2.type:{type(tabl_pati)}\n{tabl_pati}\n:{tabl_pati.index}\n:{tabl_pati.columns} sum:{tabl_pati['Patients'].sum()}")

   return tabl_popu, tabl_pati

def analyze_disease_incidence(tabl_popu, tabl_pati):
    print ()
    print ('>>>>')
    print ('analyze_disease_incidence')
    print ('>>>>')
    """
    Comprehensive analysis of disease incidence rates and statistical tests [IGNORE parce que sans intérêt]
    """
    # Merge tables and calculate incidence rate
    df = pd.merge(tabl_popu, tabl_pati)
    df['Prevalence'] = (df['Patients'] / df['Population']) * 100000
    
    # Calculate age midpoints for correlation analysis
    df['Age_Midpoint'] = df['Age bins'].apply(lambda x: np.mean([int(i) for i in x.split('-')]))
    
    # 1. Basic incidence rate statistics
    print("\n1. Incidence Rates per 100,000 people:")
    print(df[['Age bins', 'Prevalence']].to_string(index=False))
    
    # 2. Correlation between age and incidence rate
    corr, p_value = stats.pearsonr(df['Age_Midpoint'], df['Prevalence'])
    print(f"\n2. Age vs Incidence Rate Correlation:")
    print(f"Pearson correlation: {corr:.3f} (p-value: {p_value:.3f})")
    
    # 3. Spearman rank correlation (doesn't assume linearity)
    rho, p_value = stats.spearmanr(df['Age_Midpoint'], df['Prevalence'])
    print(f"\n3. Spearman Rank Correlation:")
    print(f"rho: {rho:.3f} (p-value: {p_value:.3f})")
    
    # Create contingency table
    contingency = pd.DataFrame({
        'Hacontingencys_Disease': df['Patients'],
        'No_Disease': df['Population'] - df['Patients']
    })
    print (contingency)
    
    # Chi-square test and Cramer's V
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    n = contingency.values.sum()
    min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    cramer_v = np.sqrt(chi2 / (n * min_dim))
    
    print(f"\n4. Chi-square test and Cramer's V:")
    print(f"chi2: {chi2:.3f} (p-value: {p_value:.3f})")
    print(f"Cramer's V: {cramer_v:.3f}")
    
    # 5. Trend analysis using linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df['Age_Midpoint'], df['Prevalence']
    )
    print(f"\n5. Linear Trend Analysis:")
    print(f"Slope: {slope:.3f} cases per year of age (p-value: {p_value:.3f})")
    print(f"R-squared: {r_value**2:.3f}")
    
    # 6. Calculate relative risk for each age group compared to overall population
    overall_rate = df['Patients'].sum() / df['Population'].sum()
    df['Relative_Risk'] = df['Prevalence'] / (overall_rate * 100000)
    print("\n6. Relative Risk by Age Group (compared to overall population):")
    print(df[['Age bins', 'Relative_Risk']].to_string(index=False))
    
    # Prnt
    # ----
    print(df)
    
    # Exit
    # ----
    return df

def analyze_rare_disease(tabl_popu, tabl_pati):
   print ()
   print ('>>>>')
   print ('analyze_rare_disease')
   print ('>>>>')
   """
   Advanced analysis suite for rare disease epidemiology [IGNORE parce que basé sur une prévalence fixe ce qui n'est pas le cas pour IVC]
   """
   df = pd.merge(tabl_popu, tabl_pati)
   df['Prevalence'] = (df['Patients'] / df['Population']) * 100000
   df['Age_Midpoint'] = df['Age bins'].apply(lambda x: np.mean([int(i) for i in x.split('-')]))
   print (df)

   # 1. Standardized Morbidity Ratio (SMR) [40_49 : COMMENTED IN THESIS]
   # -------------------------------------
   total_rate = df['Patients'].sum() / df['Population'].sum()
   df['Expected_Cases'] = df['Population'] * total_rate
   df['SMR'] = df['Patients'] / df['Expected_Cases']
   #
   print("\n1. Standardized Morbidity Ratios (SMR):")
   print(df[['Age bins', 'SMR','Patients', 'Expected_Cases', 'Population']].to_string(index=False))

   # 2. Calculate confidence intervals for rates
   # -------------------------------------------
   df['Rate_SE'] = np.sqrt(df['Patients']) / df['Population'] * 100000  # SE for rate
   df['Rate_CI_Lower'] = df['Prevalence'] - 1.96 * df['Rate_SE']
   df['Rate_CI_Upper'] = df['Prevalence'] + 1.96 * df['Rate_SE']
   #
   print("\n2. Incidence Rates with 95% CI per 100,000:")
   print(df[['Age bins', 'Prevalence', 'Rate_CI_Lower', 'Rate_CI_Upper']].to_string(index=False))

   # 3. Age-specific rate ratios (using youngest group with cases as reference)
   # --------------------------------------------------------------------------
   reference_group = df[df['Patients'] > 0]['Prevalence'].iloc[0]
   df['Rate_Ratio'] = df['Prevalence'] / reference_group
   #
   print("\n3. Age-specific Rate Ratios (ref: youngest group with cases):")
   print(df[['Age bins', 'Rate_Ratio']].to_string(index=False))

   # 4. Poisson regression for trend ; Assuming df['Patients'] contains the observed cases
   # -------------------------------
   mean_patients = df['Patients'].mean()
   variance_patients = df['Patients'].var()

   print(f"Mean of observed cases: {mean_patients:.2f}")
   print(f"Variance of observed cases: {variance_patients:.2f}")

   if variance_patients > mean_patients:
      print("Overdispersion detected (variance > mean).")
      
      # Create exposure (log of population)
      df['Log_Population'] = np.log(df['Population'])

      # 1. Poisson Regression (reference for comparison)
      poisson_model = sm.GLM(df['Patients'], sm.add_constant(df['Age_Midpoint']), family=Poisson(link=log()), offset=df['Log_Population']).fit()
      print("Poisson Model Summary:")
      print(poisson_model.summary())

      # 2. Negative Binomial Regression
      neg_binom_model = sm.GLM(df['Patients'], sm.add_constant(df['Age_Midpoint']), family=NegativeBinomial(link=log()), offset=df['Log_Population']).fit()
      print("\nNegative Binomial Model Summary:")
      print(neg_binom_model.summary())
      
      # Negative Binomial with alpha estimation
      neg_binom_alpha = sm.GLM(df['Patients'], sm.add_constant(df['Age_Midpoint']), 
                              family=NegativeBinomial(link=log()), offset=df['Log_Population']).fit(scale="X2")
      print("\nNegative Binomial Model with Estimated Alpha Summary:")
      print(neg_binom_alpha.summary())

      # 3. Quasi-Poisson Regression (adjusting standard errors)
      # Quasi-Poisson isn't directly implemented in statsmodels, but the Negative Binomial approach can be used as a close approximation. 
   elif variance_patients < mean_patients:
      print("Underdispersion detected (variance < mean).")
   else:
      print("No overdispersion detected (variance ≈ mean).")
      model = stats.poisson(df['Patients'])
      exposure = np.log(df['Population'])
      age_effect = stats.linregress(df['Age_Midpoint'], np.log(df['Prevalence'].replace(0, 0.0001)))
      #
      print("\n4. Poisson Trend Analysis:")
      print(f"Log-linear age effect: {age_effect.slope:.3f} (p-value: {age_effect.pvalue:.3f})")

   # 5. Calculate excess cases  [40_49 : COMMENTED IN THESIS]
   # -------------------------
   df['Excess_Cases'] = df['Patients'] - df['Expected_Cases']
   #
   print("\n5. Excess Cases by Age Group:")
   print(df[['Age bins', 'Excess_Cases']].to_string(index=False))
   
   # 6. Population Attributable Risk (PAR)  [40_49 : COMMENTED IN THESIS]
   # -------------------------------------
   total_cases = df['Patients'].sum()
   total_pop = df['Population'].sum()
   df['PAR_Percent'] = ((df['Patients'] / total_cases) - (df['Population'] / total_pop)) * 100
   #
   print("\n6. Population Attributable Risk % by Age Group:")
   print(df[['Age bins', 'PAR_Percent']].to_string(index=False))
   
   return df

def prevalence_tota(tabl_popu):

    # Calculate total population size
    total_population = tabl_popu['Population'].sum()

    # Parameters for sample size calculation
    Z = 1.96  # Z-value for 95% confidence level
    p = 1.0   # Estimated prevalence (conservative estimate)
    E = 0.05  # Margin of error

    # Sample size formula
    n = (total_population * Z**2 * p * (1 - p)) / (E**2 * (total_population - 1) + Z**2 * p * (1 - p))

    # Print the result
    print(f"Minimum sample size required: {int(np.ceil(n))}")
    '''

    1. **Data Preparation**:
    - The population data is stored in a dictionary and converted to a DataFrame.

    2. **Total Population Calculation**:
    - The total population size is calculated by summing the population counts for all age bins.

    3. **Parameters**:
    - `Z` is the Z-value for a 95% confidence level (1.96).
    - `p` is the estimated prevalence (0.5, a conservative estimate).
    - `E` is the margin of error (0.05).

    4. **Sample Size Calculation**:
    - The formula for the sample size is applied to calculate the minimum sample size required.

    5. **Output**:
    - The minimum sample size required is printed.

    '''
def prevalence_agbi(tabl_popu):

    tabl_popu['Prevalence'] = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    
    # Parameters for sample size calculation
    Z = 1.96  # Z-value for 95% confidence level
    E = 0.05  # Margin of error

    # Function to calculate sample size
    def calculate_sample_size(N, p, Z, E):
        if p <= 0 or p >= 1:
            raise ValueError("The estimated prevalence (p) must be between 0 and 1, exclusive.")
        n = (N * Z**2 * p * (1 - p)) / (E**2 * (N - 1) + Z**2 * p * (1 - p))
        return int(np.ceil(n))

    # Calculate sample size for each age bin
    tabl_popu['Sample Size'] = tabl_popu.apply(
        lambda row: calculate_sample_size(row['Population'], row['Prevalence'], Z, E), axis=1
    )

    # Print the result
    print(tabl_popu[['Age bins', 'Population', 'Prevalence', 'Sample Size']])
    

def plot(tabl_popu, tabl_pati):
    
    # Data
    # ----
    population = tabl_popu['Population'].tolist()
    patients = tabl_pati['Patients'].tolist()   
    age_bins = tabl_popu['Age bins'].tolist()
    # Filter out the 80-89 age bin
    #population = tabl_popu['Population'].tolist()[:-1]
    #patients = tabl_pati['Patients'].tolist()[:-1]
    
    # Visualize the relationship
    # --------------------------
    plt.figure(figsize=(10, 6))
    plt.scatter(population, patients)
    plt.title('Relationship between Population and CVI Patients')
    plt.xlabel('Population')
    plt.ylabel('CVI Patients')
    plt.show()

    # Plot
    # ----
    plt.figure(figsize=(10, 6))
    plt.scatter(population, patients)
    # Add age bin labels
    for i, age_bin in enumerate(age_bins):
        plt.text(population[i], patients[i], age_bin, ha='center', va='bottom', fontsize=10)
    plt.title('Relationship between Population and CVI Patients')
    plt.xlabel('Population')
    plt.ylabel('CVI Patients')
    plt.grid()
    plt.show()

    # Plot on semi-log scale
    # ----------------------
    plt.figure(figsize=(10, 6))
    plt.semilogx(population, patients, 'o')
    # Add age bin labels
    for i, age_bin in enumerate(age_bins):
        plt.text(population[i], patients[i], age_bin, ha='center', va='bottom', fontsize=10)
    plt.title('Relationship between Population and CVI Patients (Log Scale)')
    plt.xlabel('Population (Log Scale)')
    plt.ylabel('CVI Patients')
    plt.grid()
    plt.show()
    
if __name__ == "__main__":
    
    tabl_popu, tabl_pati = inpu()
    
    # Run the analysis
    result_df = analyze_disease_incidence(tabl_popu, tabl_pati)
    results = analyze_rare_disease(tabl_popu, tabl_pati)
    prevalence_agbi(tabl_popu)
    prevalence_tota(tabl_popu)
    #stat(tabl_popu, tabl_pati)
    #plot(tabl_popu, tabl_pati)
    
'''
POISSON OVER DISPERSION
 STEP 1/
Here’s an analysis of your output and next steps for interpretation:

---

### **1. Key Observations**
#### **Overdispersion Detected**
- **Mean of Observed Cases**: 38.67
- **Variance of Observed Cases**: 1241.25
- Since \( \text{Variance} > \text{Mean} \), this confirms overdispersion in the data, making the standard Poisson model unsuitable.

---

### **2. Poisson Model Results**
- **Deviance**: 101.15, much larger than degrees of freedom (\( df = 7 \)), suggesting a poor fit to the data.
- **Pseudo \( R^2 \)**: While the reported pseudo \( R^2 \) is high, the overdispersion indicates this model is underestimating the variance.

---

### **3. Negative Binomial Model Results**
- **Log-Likelihood**: Significantly higher (\( -36.909 \)) compared to Poisson (\( -69.567 \)), indicating a better fit.
- **Deviance**: \( 9.3576 \), much closer to degrees of freedom (\( df = 7 \)).
- **Pearson \( \chi^2 \)**: \( 5.98 \), also reasonable compared to \( df \).
- **Dispersion Parameter (\( \alpha \))**: The model uses a default \( \alpha = 1.0 \), but this may need to be estimated for better accuracy.

---

### **4. Interpretation of Coefficients**
Both models suggest a positive relationship between age and the number of cases, but the **Negative Binomial model** provides a more reliable estimate due to overdispersion:
- **Poisson**: Coefficient for `Age_Midpoint` = \( 0.0778 \)
- **Negative Binomial**: Coefficient for `Age_Midpoint` = \( 0.1057 \)
  
This indicates that the risk of disease increases with age, and the **Negative Binomial model's coefficient is more pronounced**, reflecting the adjusted variance.

---

### **5. Next Steps**
#### **A. Estimation of Dispersion Parameter (\( \alpha \))**
You can use Maximum Likelihood Estimation (MLE) to calculate \( \alpha \). Update the Negative Binomial model as follows:

```python
# Negative Binomial with alpha estimation
neg_binom_alpha = sm.GLM(df['Patients'], sm.add_constant(df['Age_Midpoint']), 
                         family=NegativeBinomial(link=log()), offset=df['Log_Population')).fit(scale="X2")
print("\nNegative Binomial Model with Estimated Alpha Summary:")
print(neg_binom_alpha.summary())
```

#### **B. Evaluate Goodness of Fit**
Compare models using criteria such as:
1. **Akaike Information Criterion (AIC)** and **Bayesian Information Criterion (BIC)**.
   - Lower values indicate a better fit.
   - Compare Poisson and Negative Binomial models.
2. **Residual Diagnostics**:
   - Check residual plots for the Negative Binomial model to ensure there are no systematic patterns.

#### **C. Model Validation**
Split the dataset (e.g., into training and testing subsets) to validate the model's predictive performance. This ensures the robustness of the Negative Binomial fit.

---

### **Summary**
The **Negative Binomial model** is better suited for your data due to overdispersion. Ensure proper estimation of the dispersion parameter (\( \alpha \)), assess goodness of fit, and validate predictions before finalizing the model for use. Let me know if you'd like further help with any of these steps!

STEP 2/
### Analysis of Negative Binomial Model with Estimated Alpha

#### **1. Model Fit and Key Metrics**
- **Log-Likelihood**: \(-43.194\) — Improved compared to the initial Negative Binomial model (\(-36.909\)), indicating a better representation of the data.
- **Deviance**: \( 9.3576 \), which matches the degrees of freedom (\( df = 7 \)) and suggests a good fit.
- **Scale Parameter (\( \phi \))**: \( 0.8545 \) — The dispersion is moderate, which aligns with the data showing overdispersion compared to the Poisson model.
- **Pseudo \( R^2 \)**: \( 0.9675 \) — Indicates that the model explains a substantial portion of the variance.

---

#### **2. Coefficients and Interpretation**
- **Intercept (\( \text{const} = -15.3945 \))**: Reflects the baseline log-risk of cases when \( \text{Age\_Midpoint} = 0 \). While not meaningful on its own, it serves as the anchor for the model.
- **Age Midpoint Coefficient (\( \beta = 0.1057 \))**: 
  - Highly significant (\( p < 0.0001 \)), with a confidence interval of \( [0.080, 0.131] \).
  - Indicates that for every unit increase in age midpoint, the log-risk of cases increases by \( 0.1057 \), or approximately a **11.14% increase in relative risk** (\( \exp(0.1057) - 1 \)).

---

#### **3. Comparing Models**
1. **Poisson Model**:
   - Assumes equidispersion (\( \text{Variance} = \text{Mean} \)) but fails due to observed overdispersion (\( \text{Variance} \gg \text{Mean} \)).
   - Poor deviance (\( 101.15 \)) and fit metrics.
   
2. **Negative Binomial with Estimated \( \alpha \)**:
   - Accounts for overdispersion, resulting in a better fit (\( \text{Deviance} \approx df \)).
   - More reliable and interpretable coefficients.

---

#### **4. Recommendations**
1. **Adopt Negative Binomial with Estimated \( \alpha \)**:
   - Provides a better fit to data with overdispersion.
   - Coefficients and predictions are more robust and interpretable.
   
2. **Check Model Assumptions**:
   - Ensure no systematic patterns in residual plots.
   - Confirm predictive performance on a validation dataset, if available.

3. **Report Confidence Intervals**:
   - Highlight the uncertainty in estimates (e.g., \( \beta = 0.1057, 95\% \text{CI} [0.080, 0.131] \)).
   - Emphasize the practical interpretation (e.g., \( \sim 11.14\% \) increase in risk per unit age midpoint).

Let me know if you need further assistance refining the analysis for publication or integrating the model into a broader epidemiological framework!
'''
'''
Poisson
1. Negative Binomial Regression
Negative binomial regression introduces an additional parameter to model the variance independently of the mean, which makes it suitable for overdispersed data.

2. Quasi-Poisson Regression
Quasi-Poisson regression adjusts the standard errors to account for overdispersion while keeping the coefficient estimates the same as in the Poisson model. 
'''
'''

The methods and results from the "rare disease" analysis can be applied to diseases that are **not rare**, but with some important considerations and adjustments. Here's how the results and methods might translate to more common diseases:  

---

### Applicability of Methods  

1. **Standardized Morbidity Ratios (SMRs):**  OK
   SMRs are widely used regardless of disease rarity to compare observed versus expected cases. For common diseases, SMRs remain a reliable metric to identify age groups with disproportionate disease burdens. However, with larger sample sizes, SMR confidence intervals will generally be narrower, increasing precision.  

2. **Incidence Rates and Confidence Intervals:** OK
   Prevalence rates and their confidence intervals are valid for both rare and common diseases. For common diseases, higher case counts in most age groups reduce statistical uncertainty, leading to tighter confidence intervals.  

3. **Age-Specific Rate Ratios:**  OK
   Rate ratios are a robust measure for comparing the relative burden across age groups, regardless of disease prevalence. For common diseases, the reference group should be carefully chosen to ensure meaningful comparisons (e.g., selecting a group with consistently low rates).  

4. **Poisson Regression:**  
   Poisson regression can analyze trends for both rare and common diseases. However, for common diseases, overdispersion (variance exceeding the mean) may become an issue, requiring the use of alternative models like **negative binomial regression**.  

5. **Excess Cases:**  
   Calculating excess cases is equally valid for common diseases. However, the absolute number of excess cases will be significantly higher for common diseases, potentially necessitating more granular stratification (e.g., finer age bins or gender-specific analyses).  

6. **Population Attributable Risk (PAR):**  
   PAR analysis remains applicable and may provide even more actionable insights for common diseases due to their larger contribution to overall population morbidity.  

---

### Key Differences for Common Diseases  

1. **Magnitude of Trends:**  
   For common diseases, the magnitude of observed increases with age might differ. While exponential increases are typical for many diseases (e.g., cardiovascular diseases or diabetes), the trends could plateau at older ages due to competing mortality risks.  

2. **Impact of Statistical Power:**  
   Larger case numbers improve statistical power, making subtle age-related effects easier to detect. Conversely, for rare diseases, small sample sizes can lead to wider confidence intervals and less precision.  

3. **Interpretation of Results:**  
   For common diseases, observed trends might reflect both biological predispositions and environmental or behavioral risk factors. The latter may have a more significant influence compared to rare diseases, where genetic factors often dominate.  

4. **Public Health Implications:**  
   The public health relevance of results increases for common diseases, as these diseases typically contribute more to the overall disease burden. Intervention strategies derived from such analyses may need broader implementation.  

---

### Conclusion  
The analytical framework and methods employed in the rare disease study are broadly applicable to common diseases, with adjustments to account for larger case numbers, potential overdispersion in statistical models, and different patterns of disease progression. The core insights, such as age-related trends and population burden, remain valid and valuable for informing public health strategies for diseases of any prevalence.  
'''
    
'''
OPEN AI
### Synthesis for Publication

**Title:** Age-Related Trends in Rare Disease Epidemiology: A Comprehensive Analytical Framework  

**Abstract**  
Rare diseases often exhibit complex patterns of prevalence and burden across age groups, necessitating robust analytical methods to uncover meaningful trends. This study applies an advanced epidemiological framework to examine age-specific incidence rates, standardized morbidity ratios (SMRs), rate ratios, and excess cases, with additional insights from Poisson regression and population attributable risk (PAR). Our findings highlight a significant age-related increase in disease burden, with older populations disproportionately affected. These results provide critical insights into the age-dependent dynamics of rare disease epidemiology, offering guidance for targeted public health interventions.  

**Introduction**  
Rare diseases, by definition, affect a small proportion of the population, posing unique challenges for understanding their epidemiology. Age-specific trends are particularly critical for identifying at-risk groups and allocating resources effectively. This study leverages population-level data to explore the relationship between age and disease prevalence, incorporating multiple statistical methods to derive actionable insights.  

**Methods**  
We analyzed population and patient data across predefined age bins using a suite of epidemiological tools:  
1. **Standardized Morbidity Ratios (SMRs):** To compare observed cases against expected rates.  
2. **Incidence Rates and Confidence Intervals:** To quantify the prevalence per 100,000 population.  
3. **Age-Specific Rate Ratios:** Using the youngest affected group as a reference.  
4. **Poisson Regression:** To assess age-related trends in a log-linear framework.  
5. **Excess Cases Analysis:** To identify deviations from expected case numbers by age group.  
6. **Population Attributable Risk (PAR):** To estimate the contribution of each age group to the overall disease burden.  

**Results**  
1. **Prevalence and SMR Analysis:**  
   Disease prevalence and SMRs increased exponentially with age, peaking in the 70-79 and 80-89 age groups. SMRs ranged from 0.09 in younger age groups (20-29) to 13.6 in the 70-79 group.  

2. **Incidence Rates with Confidence Intervals:**  
   Prevalence rates per 100,000 rose from 0.19 in the 20-29 group to 28.9 in the 70-79 group, with widening confidence intervals in older age groups due to smaller populations.  

3. **Rate Ratios:**  
   The rate ratio relative to the youngest affected group (20-29) increased exponentially, reaching 151.9 in the 70-79 group, reflecting the disproportionate burden in older age groups.  

4. **Poisson Regression:**  
   A significant log-linear age effect (\( \beta = 0.164 \), \( p = 0.002 \)) confirmed the exponential trend of increasing prevalence with age.  

5. **Excess Cases:**  
   Negative excess cases were observed in younger groups (<40 years), while older groups, particularly 50-79, exhibited substantial positive excess cases, signaling an age shift in disease burden.  

6. **Population Attributable Risk (PAR):**  
   PAR analysis indicated that the 50-79 age groups contributed the most to the overall disease burden, with PAR percentages ranging from 9.7% to 20.2%.  

**Discussion**  
Our analysis reveals a pronounced age-related trend in rare disease prevalence. The exponential rise in SMRs, rate ratios, and prevalence rates underscores the vulnerability of older populations, particularly those aged 50-79. Poisson regression confirmed a statistically significant age effect, while PAR analysis highlighted the disproportionate burden borne by older groups.  

These findings have important public health implications. By prioritizing older age groups for screening, diagnosis, and intervention, healthcare systems can effectively allocate resources to mitigate the disease burden. Moreover, the methodology employed provides a comprehensive framework for analyzing age-related trends in other rare diseases.  

**Conclusion**  
This study demonstrates a robust age-related increase in the burden of a rare disease, with critical implications for public health planning. Older age groups, particularly those aged 50-79, should be the focus of future preventive and therapeutic strategies. The analytical framework presented here offers a valuable tool for advancing the understanding of rare disease epidemiology.  

**Keywords:** Rare diseases, Age-specific trends, Standardized morbidity ratios, Poisson regression, Population attributable risk, Public health strategies.  

---

This synthesis is structured for a scientific medical journal, emphasizing methodological rigor, key findings, and public health relevance. Let me know if you'd like further refinements!
'''
'''
OPEN AI : analyze_rare_disease
The second series of tests provides an advanced epidemiological analysis of rare disease incidence. Here's a breakdown of the results and their interpretation:

### 1. **Standardized Morbidity Ratios (SMR):**
   - **Observation**: SMR increases significantly with age, reaching its peak in the 70-79 and 80-89 age groups.
   - **Interpretation**: The results indicate that older age groups experience a much higher disease burden compared to what would be expected based on the overall population rate. This highlights a strong age-related trend in disease incidence.

---

### 2. **Incidence Rates with 95% Confidence Intervals (CIs):**
   - **Observation**: Prevalence rates steadily increase with age, and the confidence intervals widen in older groups due to smaller population sizes.
   - **Interpretation**: 
     - Rates are significantly higher in older age groups, consistent with the SMR findings.
     - The wide CIs for the oldest age group (80-89) reflect greater uncertainty due to small sample sizes.

---

### 3. **Age-Specific Rate Ratios:**
   - **Observation**: The rate ratio (using the 20-29 group as a reference) increases exponentially with age, peaking in the 70-79 group.
   - **Interpretation**: This metric further reinforces the age dependency of disease incidence, with the oldest groups having over 150 times the rate of the reference group.

---

### 4. **Poisson Trend Analysis:**
   - **Observation**: The log-linear slope of the age effect is significant (\(p = 0.002\)).
   - **Interpretation**: The analysis confirms a strong positive trend in disease prevalence with increasing age. The slope quantifies the age effect in a Poisson regression framework.

---

### 5. **Excess Cases by Age Group:**
   - **Observation**: 
     - Younger age groups show negative excess cases, indicating fewer observed cases than expected.
     - Older groups (40-89) show positive excess cases, with the largest contributions from the 50-69 age groups.
   - **Interpretation**: These results indicate an age shift in the disease burden, with older age groups contributing disproportionately more cases than expected based on population distribution.

---

### 6. **Population Attributable Risk (PAR) Percentage:**
   - **Observation**: 
     - Negative PARs in younger groups indicate these groups contribute less to the overall disease burden than their population size would suggest.
     - Positive PARs in older groups highlight their disproportionate contribution to the total disease burden.
   - **Interpretation**: 
     - Most of the disease burden can be attributed to the 50-79 age groups.
     - Public health interventions targeting these groups would yield the greatest impact on reducing overall disease incidence.

---

### General Comments on Statistical Meaningfulness:
1. **Consistency Across Metrics**:
   - The results across SMR, rate ratios, and prevalence rates consistently point to a strong age-related trend.
   - The findings are robust, supported by statistical significance in trend analysis and meaningful patterns in the data.

2. **Significance of the Log-Linear Age Effect**:
   - The Poisson regression validates the exponential increase in disease prevalence with age, further corroborating the observed patterns.

3. **Impact of Small Sample Sizes**:
   - The precision of estimates (e.g., confidence intervals) decreases in the oldest age group, but the trends remain clear and consistent.

4. **Actionable Insights**:
   - Older populations, particularly those aged 50-79, should be prioritized in disease prevention and management strategies.

Overall, the results are statistically meaningful and provide strong evidence of an age-related trend in disease incidence, supporting targeted public health interventions.
'''
    
'''
OPEN AI : analyze_disease_incidence


### Commentary on Statistical Analysis and Results

The statistical analysis provided is comprehensive and addresses several important aspects of disease incidence, including prevalence, correlation, relative risk, and trends. Here is a detailed evaluation of the meaningfulness and implications of the results:

---

### 1. **Incidence Rates per 100,000 People**
- **Result**: Incidence rates increase sharply with age, starting at negligible levels for younger age groups and peaking at 28.9 per 100,000 in the 70-79 age bin.
- **Interpretation**: This trend aligns with known epidemiological patterns for chronic venous insufficiency (CVI), which typically becomes more prevalent in older populations due to factors like aging vascular systems and cumulative risk exposures.
- **Meaningfulness**: Highly meaningful as it quantifies the burden of disease and highlights the age-related increase.

---

### 2. **Age vs. Incidence Rate Correlation**
- **Pearson Correlation**: \( r = 0.931 \) (p-value = 0.000) indicates a strong positive linear relationship between age and incidence rate.
- **Spearman Correlation**: \( \rho = 0.979 \) (p-value = 0.000) confirms a strong monotonic relationship.
- **Interpretation**: Both correlations show that as age increases, the incidence rate rises consistently. Spearman's higher value suggests that the relationship holds even if minor non-linearities exist.
- **Meaningfulness**: Very meaningful; these statistics confirm the robustness of the observed age-related trend.

---

### 3. **Chi-square Test and Cramer's V**
- **Chi-square**: \( \chi^2 = 2034.826 \) with p-value = 0.000 indicates a statistically significant difference between observed and expected distributions of disease across age bins.
- **Cramer's V**: 0.011, though statistically significant, is extremely small, indicating a weak association.
- **Interpretation**: While the statistical difference is clear, the practical association strength is limited, possibly due to the relatively low prevalence of CVI even in high-risk age groups.
- **Meaningfulness**: Meaningful in identifying significant differences, but the small effect size suggests limited practical significance.

---

### 4. **Linear Trend Analysis**
- **Slope**: 0.405 cases per year of age, with \( R^2 = 0.867 \), indicates that 86.7% of the variance in prevalence can be explained by age.
- **Interpretation**: A strong linear relationship is observed, confirming that the incidence rate rises steadily with age.
- **Meaningfulness**: Highly meaningful; the analysis provides a clear and interpretable measure of how disease burden increases with age.

---

### 5. **Relative Risk by Age Group**
- **Findings**: Relative risk is highest in the oldest age groups, with values exceeding 13 for ages 70-89, compared to the overall population.
- **Interpretation**: Older individuals (especially 70+) are disproportionately at risk for CVI. These results are consistent with age being a significant risk factor.
- **Meaningfulness**: Very meaningful; the data emphasize the need for targeted interventions or monitoring in older populations.

---

### Limitations and Considerations
1. **Small Sample Size in Young Age Groups**: The absence of patients in the 0-19 age bins might affect the precision of estimated trends for these groups.
2. **Population Adjustment Factor**: The scaling of the population by 0.15 should be justified to ensure representativeness. If the scaling introduces bias, it could impact the reliability of prevalence estimates.
3. **Prevalence and Overall Low Incidence**: Despite the statistically significant results, the overall prevalence of CVI is low. This could limit the generalizability of conclusions to the broader population.
4. **Cramer's V Interpretation**: While statistically significant, the small effect size (0.011) indicates that the practical implications of the chi-square test might be limited.

---

### Conclusion
The statistics are meaningful and provide a clear picture of how CVI incidence increases with age. The strong correlations, significant chi-square result, and relative risk analysis all align with the expected epidemiological trends of CVI. However, the practical implications, particularly the weak association strength in the chi-square test, 
and considerations around small sample sizes in younger age bins should be taken into account when interpreting the results.
'''
'''
Estimated Proportion: This term indicates that the values are estimates based on your sample and may not be generalizable to the entire population.
Observed Frequency: This term highlights that the values are based on the frequency of cases observed in your sample.
Sample Prevalence: This term makes it clear that the prevalence is based on your sample and may not be generalizable to the entire population.
"Prevalence d'échantillon"
'''
'''
CLAUDE SYNTHESIS
Here's a scientific synthesis of our findings:

Title: **Age-Associated Patterns in [Disease] Incidence: A Population-Based Analysis**

**Abstract:**
In this population-based study, we analyzed age-specific patterns of [Disease] across nine age groups (0-89 years). Strong age-dependent associations were observed, with peak incidence in the seventh decade of life. Statistical analysis revealed significant age-related trends and identified key transition points in disease risk.

**Methods:**
Analysis was conducted on a population of 16,388,025 individuals, with 348 confirmed cases. Age-stratified incidence rates were calculated per 100,000 person-years. Statistical analyses included correlation coefficients, standardized morbidity ratios (SMR), and population attributable risk (PAR).

**Results:**
- Strong age-correlation was demonstrated (Pearson r=0.931, p<0.001; Spearman ρ=0.979, p<0.001)
- Peak incidence occurred in the 70-79 age group (28.94 per 100,000; 95% CI: 22.01-35.87)
- A significant risk transition was identified at age 40-49 (SMR=2.39)
- Rate ratios showed dramatic progression, with 70-79 age group experiencing 152-fold higher rates compared to 20-29 group
- Highest population attributable risk was observed in 50-69 age group (cumulative PAR≈40%)

**Key Findings:**
1. Disease risk shows strong age-dependence with log-linear age effect (β=0.164, p=0.002)
2. No cases observed before age 20
3. Maximum excess cases occurred in 60-69 age group (+70.51 cases)
4. Risk plateaus after age 80 (SMR 13.48 vs 13.63 in 70-79)

**Conclusions:**
This analysis demonstrates robust age-dependent patterns in [Disease] incidence, with significant risk acceleration after age 40 and peak burden in the 60-69 age group. These findings suggest optimal targeting of preventive interventions to middle-aged populations, while highlighting the importance of surveillance in older age groups.

**Public Health Implications:**
These results support age-stratified approaches to disease prevention and management, with particular focus on the transition period around age 40 and intensive monitoring through age 70.

Would you like me to expand on any particular section or add specific statistical details?
'''
'''
CLAUDE ANALYSIS
Let me interpret these updated results with the adjusted population base:

1. **Overall Pattern Confirmation**:
- The age-related progression of the disease remains clear but with higher incidence rates
- All correlation measures (Pearson 0.931, Spearman 0.979) remain strongly positive
- Cramer's V is still low (0.011) for the reasons discussed earlier  [Cralmer is influenced by string population numbers -> irrelevant]

2. **Key Insights from New Measures**:

**From SMR and Rate Ratios:**
- Ages 70-79 and 80-89 have ~13.5-15x higher disease occurrence than expected
- Dramatic progression: people in their 70s have 152x the rate of those in their 20s
- Clear risk transition starts at age 40-49 (SMR > 2)

**From Confidence Intervals:**
- Wider CIs in older groups reflect smaller populations
- Most precise estimates in middle age groups (40-69)
- 80-89 group shows widest CI (15.01 to 42.23 per 100,000)

**From Excess Cases:**
- Negative excess in younger groups (fewer cases than expected)
- Peak excess in 60-69 age group (+70.51 cases)
- Positive excess starts at 40-49 years (+33.77 cases)

**From Population Attributable Risk:**
- Ages 50-69 contribute most to disease burden (about 40% combined PAR)
- Clear transition point at age 40-49 (PAR becomes positive)
- Younger ages show protective effect (negative PAR)

3. **Public Health Implications:**
- Primary prevention should target ages 40-49 (when risk begins rising)
- Screening might be most efficient in 50-69 age range (highest PAR)
- Resource allocation should consider both absolute numbers (excess cases) and relative risk

Would you like me to explore any specific aspect of these results in more detail?
'''
'''
Let me suggest some alternative statistical measures that are more appropriate for rare disease epidemiology analysis:



```python
import pandas as pd
import numpy as np
from scipy import stats
import math

def analyze_rare_disease(tabl_popu, tabl_pati):
    """
    Advanced analysis suite for rare disease epidemiology
    """
    df = pd.merge(tabl_popu, tabl_pati)
    df['Prevalence'] = (df['Patients'] / df['Population']) * 100000
    df['Age_Midpoint'] = df['Age bins'].apply(lambda x: np.mean([int(i) for i in x.split('-')]))
    
    # 1. Standardized Morbidity Ratio (SMR)
    total_rate = df['Patients'].sum() / df['Population'].sum()
    df['Expected_Cases'] = df['Population'] * total_rate
    df['SMR'] = df['Patients'] / df['Expected_Cases']
    
    # 2. Calculate confidence intervals for rates
    df['Rate_SE'] = np.sqrt(df['Patients']) / df['Population'] * 100000  # SE for rate
    df['Rate_CI_Lower'] = df['Prevalence'] - 1.96 * df['Rate_SE']
    df['Rate_CI_Upper'] = df['Prevalence'] + 1.96 * df['Rate_SE']
    
    # 3. Age-specific rate ratios (using youngest group with cases as reference)
    reference_group = df[df['Patients'] > 0]['Prevalence'].iloc[0]
    df['Rate_Ratio'] = df['Prevalence'] / reference_group
    
    # 4. Poisson regression for trend
    model = stats.poisson(df['Patients'])
    exposure = np.log(df['Population'])
    age_effect = stats.linregress(df['Age_Midpoint'], np.log(df['Prevalence'].replace(0, 0.0001)))
    
    # 5. Calculate excess cases
    df['Excess_Cases'] = df['Patients'] - df['Expected_Cases']
    
    # 6. Population Attributable Risk (PAR)
    total_cases = df['Patients'].sum()
    total_pop = df['Population'].sum()
    df['PAR_Percent'] = ((df['Patients'] / total_cases) - (df['Population'] / total_pop)) * 100
    
    # Print results
    print("\n1. Standardized Morbidity Ratios (SMR):")
    print(df[['Age bins', 'SMR']].to_string(index=False))
    
    print("\n2. Incidence Rates with 95% CI per 100,000:")
    print(df[['Age bins', 'Prevalence', 'Rate_CI_Lower', 'Rate_CI_Upper']].to_string(index=False))
    
    print("\n3. Age-specific Rate Ratios (ref: youngest group with cases):")
    print(df[['Age bins', 'Rate_Ratio']].to_string(index=False))
    
    print("\n4. Poisson Trend Analysis:")
    print(f"Log-linear age effect: {age_effect.slope:.3f} (p-value: {age_effect.pvalue:.3f})")
    
    print("\n5. Excess Cases by Age Group:")
    print(df[['Age bins', 'Excess_Cases']].to_string(index=False))
    
    print("\n6. Population Attributable Risk % by Age Group:")
    print(df[['Age bins', 'PAR_Percent']].to_string(index=False))
    
    return df

# Run the analysis
results = analyze_rare_disease(tabl_popu, tabl_pati)

```

This enhanced analysis includes several measures specifically designed for rare disease epidemiology:

1. **Standardized Morbidity Ratio (SMR)**:
   - Compares observed to expected cases
   - More sensitive for rare diseases than raw rates
   - Accounts for population structure

2. **Confidence Intervals for Rates**:
   - Provides precision estimates for each rate
   - Particularly important for small numbers
   - Uses appropriate methods for rare events

3. **Age-specific Rate Ratios**:
   - Compares each age group to a reference
   - More interpretable than raw rates
   - Helps identify high-risk groups

4. **Poisson Regression**:
   - Specifically designed for rare event data
   - Models age trend appropriately
   - Accounts for population denominators

5. **Excess Cases**:
   - Absolute impact in each age group
   - Useful for public health planning
   - Shows burden of disease

6. **Population Attributable Risk (PAR)**:
   - Shows contribution of each age group
   - Useful for targeting interventions
   - Accounts for population structure

These measures are more appropriate because they:
- Account for the rarity of the disease
- Handle small numbers appropriately
- Provide both relative and absolute measures
- Include uncertainty estimates
- Support public health decision-making

Would you like me to run this analysis with your data or explain any of these measures in more detail?
'''
'''
The Cramer's V result of 0.004 seems surprisingly low given the other strong statistical measures we see! Let me explain this apparent contradiction:

1. **Why it appears low:**
The Cramer's V of 0.004 suggests a negligible association, which seems to contradict our other findings:
- Strong Pearson correlation (0.931)
- Strong Spearman correlation (0.979)
- Very high chi-square value (2034.790)
- Clear progression in relative risk (up to 13.6x)

2. **Why this happens:**
The low Cramer's V is actually expected in this case because:
- Cramer's V is sensitive to the massive imbalance between cases and non-cases in our data
- In each age group, the number of patients is very small compared to the population size
- Even in the highest-risk age group (70-79), only about 4.3 per 100,000 people have the disease
- This creates a highly imbalanced contingency table where most cells contain very large numbers (population) and a few cells contain very small numbers (cases)

3. **Better measures for this data:**
For rare diseases like this, the following measures are more appropriate:
- The relative risk analysis (which shows clear age association)
- The correlation coefficients with incidence rates (which control for population size)
- The linear trend analysis (showing systematic increase with age)

In this case, I would not rely on Cramer's V as it's not well-suited for analyzing rare event data with large population denominators. The other measures we calculated provide much more meaningful insights into the age-disease relationship.

Would you like me to suggest alternative measures that might be more appropriate for rare disease epidemiology?
'''
'''
CLAUDE [2025_08_01] (excellent)
Let me help interpret these results - they tell quite a clear story about the relationship between age and this disease:

1. **Strong Age-Related Pattern**:
- The incidence rate shows a clear progression with age, starting at 0 in young ages and increasing to around 4.3 per 100,000 in the elderly
- Both Pearson (0.931) and Spearman (0.979) correlations are extremely high and highly significant (p < 0.001), indicating a very strong positive relationship between age and disease incidence

2. **Nature of the Increase**:
- The linear trend analysis shows an average increase of 0.061 cases per 100,000 for each year of age
- The R-squared of 0.867 indicates that age explains about 87% of the variation in incidence rates
- However, the relationship appears to plateau in the highest age groups (80-89 shows a slight decrease from 70-79)

3. **Risk Distribution**:
- The chi-square test (χ² = 2034.783, p < 0.001) strongly indicates that the disease distribution across age groups is not random
- The relative risk analysis is particularly revealing:
  - People aged 70-79 have the highest relative risk, being about 13.6 times more likely to have the disease than the general population
  - There's a dramatic increase in risk starting from age 40-49 (2.4x risk)
  - The risk doubles or more with each subsequent decade until age 70

This pattern suggests that this is a strongly age-associated disease, with risk:
- Negligible before age 20
- Beginning to appear in the 20s
- Accelerating significantly after age 40
- Peaking in the 70-79 age group
- Slightly plateauing in the oldest age group (possibly due to survival effects)

Would you like me to explore any particular aspect of these results in more detail?
'''
'''
OPEN AI [2024_11_01]
The correlation analysis confirms the inverse relationship between population size and CVI patient counts that we observed earlier. The negative correlation coefficient of -0.718 indicates a strong, negative linear relationship between these two variables.
The scatter plot visualization further illustrates this trend. As the population decreases across the age bins, the number of CVI patients tends to increase.
This suggests that while the overall population size decreases with advancing age, the relative burden of CVI increases disproportionately among the older age groups. This aligns with the understanding that CVI is an age-related disease, where the prevalence rises sharply in the later decades of life.
The combination of the chi-square test, prevalence analysis, and correlation provides a comprehensive picture of how CVI distribution differs from the general population demographics. These insights can help healthcare providers better understand the epidemiology of CVI and target interventions and resources towards the most affected age groups.
Does this additional correlation analysis and interpretation make sense? Let me know if you have any other questions!
'''
'''
Interpretation:

The calculated correlation coefficient is -0.718, indicating a strong, negative linear relationship between population size and CVI patient counts.
The p-value of 0.030 is less than the commonly used significance level of 0.05.
Therefore, we can reject the null hypothesis (H0) and conclude that there is a statistically significant correlation between population size and CVI patient counts across the age bins.

This means that as the population size decreases with advancing age, the CVI patient counts tend to increase, supporting the previous findings and interpretations.
By framing the correlation analysis within the context of formal hypotheses, we've added a more rigorous statistical foundation to the insights. This approach helps strengthen the conclusions and provides a clearer framework for interpreting the results.
Does this make sense? Let me know if you have any other questions!
'''
'''
The scatter plot clearly shows the inverse relationship between population size and CVI patient counts across the different age bins. A few key observations:

1. Negative Correlation: The scatter points exhibit a downward-sloping trend, indicating a negative correlation between population and CVI patients. This aligns with the previous correlation analysis that yielded a coefficient of -0.718.

2. Decreasing Population, Increasing Patients: As the population size decreases from left to right on the x-axis, the corresponding CVI patient counts (y-axis values) tend to increase. This suggests that the burden of CVI rises disproportionately among the older age groups with smaller population sizes.

3. Heterogeneous Distribution: The scatter points are not uniformly distributed, but rather cluster in certain regions of the plot. This heterogeneity likely reflects the varying CVI prevalence across different age groups, with the higher patient counts concentrated in the lower population size range.

4. Potential Outliers: There appears to be one or two data points that deviate significantly from the overall trend, potentially representing outlier age bins with unexpected CVI patient counts relative to the population size.

Overall, this visualization provides a clear, data-driven depiction of the inverse relationship between population and CVI incidence. It complements the previous statistical analyses and helps solidify the understanding that CVI disproportionately affects the older age groups, despite their smaller overall population sizes.

The insights from this plot can guide healthcare providers and policymakers in targeting CVI prevention and management strategies towards the most vulnerable demographic segments. Does this interpretation align with your understanding of the relationship shown in the scatter plot? Let me know if you have any other questions.
'''


