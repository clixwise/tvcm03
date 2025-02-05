import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.oneway import anova_oneway
from util_file_mngr import write
from scipy import stats  # We'll use scipy's levene test instead

# -------------------------------
# ANOVA Test of Independence
# -------------------------------

def anov(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df1):
    
    # Trac
    trac = False
        
    # Prec
    indx_cate_nam1 = indx_cate_list[0]
    indx_cate_nam2 = indx_cate_list[1] 
    df2 = df1.sort_values(by=indx_name_stra) # note : same 'stat, pval' whether sorted or not  
    indx_list_stra = df1[indx_name_stra]# df1['Gender_num'] = df1['Gender'].map({'Male': 0, 'Female': 1})
    colu_list_ordi = df1[colu_name_ordi]
    if trac:
        print(f"\nStep 0 : indx_list_stra.size:{len(indx_list_stra)} df1.type:{type(indx_list_stra)}\n{indx_list_stra}\n:{indx_list_stra.index}")
        write(f"\nStep 0 : indx_list_stra.size:{len(indx_list_stra)} df1.type:{type(indx_list_stra)}\n{indx_list_stra}\n:{indx_list_stra.index}")
        print(f"\nStep 0 : colu_list_ordi.size:{len(colu_list_ordi)} df1.type:{type(colu_list_ordi)}\n{colu_list_ordi}\n:{colu_list_ordi.index}")
        write(f"\nStep 0 : colu_list_ordi.size:{len(colu_list_ordi)} df1.type:{type(colu_list_ordi)}\n{colu_list_ordi}\n:{colu_list_ordi.index}")

    # Exec Calculate the grand mean
    gran_mean = colu_list_ordi.mean()
    gran_mean_form = f"{gran_mean:.3e}" if gran_mean < 0.001 else f"{gran_mean:.3f}"
    # Calculate deviations from the grand mean for each gender
    for gender in [indx_cate_nam1, indx_cate_nam2]: # gender in ['Male', 'Female']
        gender_mean = df1[df1[indx_name] == gender][colu_name_ordi].mean() # df[df['Gender'] == gender]['Age_Numeric'].mean()
        devi = gender_mean - gran_mean
        devi_form = f"{devi:.3e}" if devi < 0.001 else f"{devi:.3f}"
        print(f"{indx_name}:{gender} - {colu_name} mean:{gran_mean_form}, deviation:{devi_form}")

    # Perform Levene's test
    male_ages = df1[df1[indx_name] == indx_cate_nam1][colu_name_ordi] # df[df['Gender'] == 'Male']['Age_Numeric']
    female_ages = df1[df1[indx_name] == indx_cate_nam2][colu_name_ordi] # df[df['Gender'] == 'Female']['Age_Numeric']
    stat_leve, pval_leve = stats.levene(male_ages, female_ages)
    if pval_leve < 0.05:
        use_var = 'unequal' # The assumption of homogeneity of variances is violated ; Using Welch's ANOVA
    else:
        use_var = 'equal' # The assumption of homogeneity of variances is met ; Using standard ANOVA

    # Perform ANOVA (standard or Welch's depending on Levene's test result)
    anova_results = anova_oneway(df1[colu_name_ordi], df1[indx_name], use_var=use_var) # df['Age_Numeric'], df['Gender']
    stat = anova_results.statistic
    pval = anova_results.pvalue

    # Effect size (Eta-squared)
    def eta_squared(f_statistic, df_between, df_within):
        return (f_statistic * df_between) / (f_statistic * df_between + df_within)
    df_between = len(df.index) - 1  # Number of groups - 1
    df_within = len(df1) - len(df.index)  # Total sample size - Number of groups
    effe_size = eta_squared(stat, df_between, df_within)
    effe_size_intp = f"{'small' if effe_size < 0.06 else 'medium' if effe_size < 0.14 else 'large'} effect"

    # Resu
    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    effe_size_form = f"{effe_size:.3e}" if effe_size < 0.001 else f"{effe_size:.3f}"
    pval_leve_form = f"{pval_leve:.3e}" if pval_leve < 0.001 else f"{pval_leve:.3f}"
    print(f"\nData : {what}\nANOVA : Stat:{stat_form} Effect size (Eta-squared):{effe_size_form} ; {effe_size_intp}")
    print(f"Pval Levene:{pval_leve_form} (assumption of homogeneity of variances ie 'pval>0.05') ; if 'Pval Levene < 0.05' then consider 'Pval Welch's F'")
    write(f"\nData : {what}\nANOVA : Stat:{stat_form} Effect size (Eta-squared):{effe_size_form} ; {effe_size_intp}")
    write(f"Pval Levene:{pval_leve_form} (assumption of homogeneity of variances ie 'pval>0.05') ; if 'Pval Levene < 0.05' then consider 'Pval Welch's F'")
   
    # Intp
    # Mistral
    # H0 = "H0 : there is no difference in the means of the severity scores for the left and right sides across different categories."
    # Ha = "Ha : there is a difference in the means of the severity scores for the left and right sides across different categories."
    H0 = f"H0 : There is no difference in '{colu_name}' between '{indx_name}' groups\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"Ha : There is a difference in '{colu_name}' between '{indx_name}' groups\n({colu_cate_list}) vs ({indx_cate_list})"
    alpha = 0.05
    if pval < alpha:
        print(f"ANOVA : Reject the null hypothesis:\n{Ha}")
        write(f"ANOVA : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"ANOVA : Fail to reject the null hypothesis:\n{H0}")
        write(f"ANOVA : Fail to reject the null hypothesis:\n{H0}")
    pass
