import os
import sys
import pandas as pd
from util_file_mngr import write
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_effectsize, proportion_confint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import logit
from sklearn.metrics import roc_curve, auc

# ----
# Stat
# ----
def desc(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line):
    
    # Prec   
    memG_name= indx_cate_list[0] ; memD_name= indx_cate_list[1] 
    veNA_name= colu_cate_list[0] ; veVI_name= colu_cate_list[1] 
    
    # Exec 
    tota_memb = len(df_line)
    memG = len(df_line[df_line[indx_name] == memG_name]) # len(df_line[df_line['mbre'] == 'G'])
    memD = len(df_line[df_line[indx_name] == memD_name]) # len(df_line[df_line['mbre'] == 'D'])
    veNA = len(df_line[df_line[colu_name] == veNA_name]) # len(df_line[df_line['vein'] == 'NA'])
    veVI = len(df_line[df_line[colu_name] == veVI_name]) # len(df_line[df_line['vein'] == 'VI'])
    
    print(f"\nData : {what}")
    print(f"Desc : {indx_name}: tota:{tota_memb} - {memG_name}:{memG} ({memG/tota_memb:.2%}) - {memD_name}:{memD} ({memD/tota_memb:.2%})")
    print(f"Desc : {colu_name}: tota:{tota_memb} - {veNA_name}:{veNA} ({veNA/tota_memb:.2%}) - {veVI_name}:{veVI} ({veVI/tota_memb:.2%})")
    write(f"\nData : {what}")
    write(f"Desc : {indx_name}: tota:{tota_memb} - {memG_name}:{memG} ({memG/tota_memb:.2%}) - {memD_name}:{memD} ({memD/tota_memb:.2%})")
    write(f"Desc : {colu_name}: tota:{tota_memb} - {veNA_name}:{veNA} ({veNA/tota_memb:.2%}) - {veVI_name}:{veVI} ({veVI/tota_memb:.2%})")

def chi2(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line):

    # Exec
    chi2, pval, dof, expected = stats.chi2_contingency(df_tabl)

    # Resu
    stat_form = f"{chi2:.3e}" if chi2 < 0.001 else f"{chi2:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\nChi-Square : Stat:{stat_form} Pval:{pval_form} Dof:{dof}")
    write(f"\nData : {what}\nChi-Square : Stat:{stat_form} Pval:{pval_form} Dof:{dof}")  

def phic(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line):

    # Exec
    chi2, pval, dof, expected = stats.chi2_contingency(df_tabl)
    n = df_tabl.sum().sum()
    phic = (chi2 / n) ** 0.5

    # Resu
    phic_form = f"{phic:.3e}" if phic < 0.001 else f"{phic:.3f}"
    print(f"\nData : {what}\nPHI Coefficient : Stat:{phic_form}")
    write(f"\nData : {what}\nPHI Coefficient : Stat:{phic_form}")  

def cram(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line):     
    # Exec
    # PHI coef
    chi2, pval, dof, expected = stats.chi2_contingency(df_tabl)
    n = df_tabl.sum().sum()
    phic = (chi2 / n) ** 0.5
    # Cramer V
    min_dim = min(df_tabl.shape) - 1
    cramer_v = stat = phic / (min_dim ** 0.5)
    
    # Resu
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    print(f"\nData : {what}\nCramer V : Stat:{stat_form}")
    write(f"\nData : {what}\nCramer V : Stat:{stat_form}")
    
def fish(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line):
   
    # Extract the contingency table
    table = df_tabl.values
    print(f"df_tabl:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}\n:{df_tabl.columns}\nSum:{df_tabl.sum().sum()}")
    print(f"table  :{table}")
    
    # Calculate Odds Ratio -> 'odds_ratio_manu' == 'odds_ratio' by 'stats.fisher_exact'
    a, b = table[0]
    c, d = table[1]
    odds_ratio_manu = (a * d) / (b * c)

    # Exec
    odds_ratio, pval = stats.fisher_exact(df_tabl)
    stat = odds_ratio
    if odds_ratio != odds_ratio_manu:
        raise Exception()
    
    # Resu
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\nFisher-Exact : Stat:{stat_form} Pval:{pval_form}")
    write(f"\nData : {what}\nFisher-Exact : Stat:{stat_form} Pval:{pval_form}") 

def risk(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line):
   
    # Prec   
    memG_name= indx_cate_list[0] ; memD_name= indx_cate_list[1] 
    veNA_name= colu_cate_list[0] ; veVI_name= colu_cate_list[1] 
    
    # Exec 
    memG_risk = df_tabl.loc[memG_name, veVI_name] / df_tabl.loc[memG_name].sum()
    memD_risk = df_tabl.loc[memD_name, veVI_name] / df_tabl.loc[memD_name].sum()
    risk_divi = memG_risk / memD_risk
    risk_diff = memG_risk - memD_risk
    # Number to traet
    number_needed_to_treat = 1 / abs(risk_diff)
    
    # Resu
    risk_divi_form = f"{risk_divi:.3e}" if risk_divi < 0.001 else f"{risk_divi:.3f}"
    risk_diff_form = f"{risk_diff:.3e}" if risk_diff < 0.001 else f"{risk_diff:.3f}"
    need_form = f"{number_needed_to_treat:.3e}" if number_needed_to_treat < 0.001 else f"{number_needed_to_treat:.3f}"
    print(f"\nData : {what}\nRelative risk(/):{risk_divi_form} risk(-):{risk_diff_form} number needed to treat:{need_form}")
    write(f"\nData : {what}\nRelative risk(/):{risk_divi_form} risk(-):{risk_diff_form} number needed to treat:{need_form}")

def prop(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line):
   
    # Exec   
    memG_name= indx_cate_list[0] ; memD_name= indx_cate_list[1] 
    veNA_name= colu_cate_list[0] ; veVI_name= colu_cate_list[1]  
    memG_veVI = df_tabl.loc[memG_name, veVI_name]
    memG_tota = df_tabl.loc[memG_name].sum()
    male_proportion = memG_veVI / memG_tota
    ci_memG = proportion_effectsize(male_proportion, 0.5)  # 0.5 is a reference proportion, adjust as needed
    memD_veVI = df_tabl.loc[memD_name, veVI_name]
    memD_tota = df_tabl.loc[memD_name].sum()
    female_proportion = memD_veVI / memD_tota
    ci_memD = proportion_effectsize(female_proportion, 0.5)  # 0.5 is a reference proportion, adjust as needed
    
    # Resu
    ci_memG_form = f"{ci_memG:.3e}" if ci_memG < 0.001 else f"{ci_memG:.3f}"
    ci_memD_form = f"{ci_memD:.3e}" if ci_memD < 0.001 else f"{ci_memD:.3f}"
    print(f"\nData : {what}\nProportion Effect Size : CI_{memG_name}:{ci_memG_form} CI_{memD_name}:{ci_memD_form}")
    write(f"\nData : {what}\nProportion Effect Size : CI_{memG_name}:{ci_memG_form} CI_{memD_name}:{ci_memD_form}")
    
    # Exec
    alpha = 0.05  # 95% confidence interval
    for sexe in indx_cate_list: # for gender in ['M', 'F']:
        sexe_veVI = df_tabl.loc[sexe, veVI_name]
        sexe_tota = df_tabl.loc[sexe].sum()
        ci_lowr, ci_uppr = proportion_confint(sexe_veVI, sexe_tota, alpha=alpha, method='wilson')
        ci_lowr_form = f"{ci_lowr:.3e}" if ci_lowr < 0.001 else f"{ci_lowr:.3f}"
        ci_uppr_form = f"{ci_uppr:.3e}" if ci_uppr < 0.001 else f"{ci_uppr:.3f}"
        print(f"95%CI : {sexe}: {ci_lowr_form} - {ci_uppr_form}")
        write(f"95%CI : {sexe}: {ci_lowr_form} - {ci_uppr_form}")

def logi(what, df_tabl, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line):
    
    # Prec   
    memG_name= indx_cate_list[0] ; memD_name= indx_cate_list[1] 
    veNA_name= colu_cate_list[0] ; veVI_name= colu_cate_list[1] 
    
    # Exec
    X = pd.get_dummies(df_line[indx_name], drop_first=True) # X = pd.get_dummies(df1['mbre'], drop_first=True)
    print (X)
    y = df_line[colu_name].map({veNA_name: 0, veVI_name: 1}) # y = df1['Ill'].map({'N': 0, 'Y': 1})
    print (y)
    X = X.astype(int)
    X = sm.add_constant(X)
    modl = sm.Logit(y, X).fit()
    
    # Resu
    print(f"\nData : {what}\nLogistic regression :\n{modl.summary()}")
    write(f"\nData : {what}\nLogistic regression :\n{modl.summary()}")
    pass

def plot(what, dg1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line):

    # Assuming df1 is your original DataFrame with 724 rows
    # If 'vein' is categorical, convert it to binary
    df_line['vein_binary'] = (df_line[colu_name] == 'VI').astype(int)

    # Fit logistic regression model
    model = logit(f"vein_binary ~ {indx_name}", data=df_line).fit()

    # Print summary
    print(model.summary())

    # Get predictions
    df_line['predicted_prob'] = model.predict(df_line)
    print (df_line)
    # Visualizations
    plt.figure(figsize=(12, 6))

    # 1. Histogram of predicted probabilities by gender
    plt.subplot(121)
    sns.histplot(data=df_line, x='predicted_prob', hue=indx_name, element='step', stat='density', common_norm=False)
    plt.title('Predicted Probabilities by Gender')
    plt.xlabel('Predicted Probability of Vein Insufficiency')

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(df_line['vein_binary'], df_line['predicted_prob'])
    roc_auc = auc(fpr, tpr)

    plt.subplot(122)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    # 3. Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(df_line['vein_binary'], (df_line['predicted_prob'] > 0.5).astype(int))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    pass

def stat_glob(what, dg1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line, yate):
    if False:
        plot(what, dg1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    desc(what, dg1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    chi2(what, dg1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    fish(what, dg1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    phic(what, dg1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    cram(what, dg1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    risk(what, dg1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    prop(what, dg1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)
    logi(what, dg1, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df_line)               
    pass