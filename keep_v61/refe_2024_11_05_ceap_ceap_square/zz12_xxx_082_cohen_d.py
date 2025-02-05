import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from scipy.stats import wilcoxon, spearmanr, skew
import matplotlib.pyplot as plt

def ceap_to_numeric(ceap_list):
    return max([i for i, x in enumerate(ceap_list) if x == 1])

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def interpret_cohens_d(d):
    if abs(d) < 0.2:
        return "negligible"
    elif abs(d) < 0.5:
        return "small"
    elif abs(d) < 0.8:
        return "medium"
    else:
        return "large"
        
# -------------------------------
# Cohen d Test of Independence
# -------------------------------

def cohe(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df1):
    
    # Trac
    trac = False
        
    # Prec
    df2 = df1.sort_values(by=indx_name_stra) # note : same 'stat, pval' whether sorted or not   
    indx_list_stra = df1[indx_name_stra].apply(ceap_to_numeric) # 'ceaL'
    colu_list_ordi = df1[colu_name_ordi].apply(ceap_to_numeric) # 'ceaR'
    if trac:
        print(f"\nStep 1 : indx_list_stra.size:{len(indx_list_stra)} df2.type:{type(indx_list_stra)}\n{indx_list_stra}\n:{indx_list_stra.index}")
        write(f"\nStep 1 : indx_list_stra.size:{len(indx_list_stra)} df2.type:{type(indx_list_stra)}\n{indx_list_stra}\n:{indx_list_stra.index}")
        print(f"\nStep 2 : colu_list_ordi.size:{len(colu_list_ordi)} df2.type:{type(colu_list_ordi)}\n{colu_list_ordi}\n:{colu_list_ordi.index}")
        write(f"\nStep 2 : colu_list_ordi.size:{len(colu_list_ordi)} df2.type:{type(colu_list_ordi)}\n{colu_list_ordi}\n:{colu_list_ordi.index}")

    # Exec
    stat = cohens_d(indx_list_stra, colu_list_ordi)
    effe_size = interpret_cohens_d(stat)
    # Calculate mean difference and confidence interval
    diff = indx_list_stra - colu_list_ordi
    mean_diff = np.mean(diff)
    sem = stats.sem(diff)
    df = len(diff) - 1
    ci = stats.t.interval(confidence=0.95, df=df, loc=mean_diff, scale=sem)

    # Resu
    if np.isnan(stat):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    if stat > 0:
        com1_form = f"The positive Cohen's d value indicates that, on average, {indx_name_stra} tends to have higher severity."
    else:
        com1_form = f"The positive Cohen's d value indicates that, on average, {colu_name_ordi} tends to have higher severity."
    mean_diff_form = f"{mean_diff:.3e}" if mean_diff < 0.001 else f"{mean_diff:.3f}"
    mean_diff_abso_form = f"{abs(mean_diff):.3e}" if mean_diff < 0.001 else f"{abs(mean_diff):.3f}"
    cilo_form = f"{ci[0]:.3e}" if ci[0] < 0.001 else f"{ci[0]:.3f}"
    ciup_form = f"{ci[1]:.3e}" if ci[1] < 0.001 else f"{ci[1]:.3f}"
    com2_form = "The test provides a standardized measure of the difference between the two groups."
    com3_form = "The test interprets the magnitude of the effect, regardless of the scale of the original variables."
    print(f"\nData : {what}\nCohen D : Stat:{stat_form} Effe:{effe_size}")
    print(f"Cohen D : {com1_form}")
    print(f"Cohen D : The average severity in the left leg is {mean_diff_abso_form} points {'higher' if mean_diff > 0 else 'lower'} than in the right leg.")
    print(f"Cohen D : Mean difference ({indx_name_stra} - {colu_name_ordi}): {mean_diff_form} 95%CI between {cilo_form} and {ciup_form}.")
    print(f"Cohen D : The average severity in {indx_name_stra} is {mean_diff_abso_form} points {'higher' if mean_diff > 0 else 'lower'} than in {colu_name_ordi}.")
    print(f"Cohen D : {com2_form}")
    print(f"Cohen D : {com3_form}")
    write(f"\nData : {what}\nCohen D : Stat:{stat_form} Effe:{effe_size}")
    write(f"Cohen D : {com1_form}")
    write(f"Cohen D : The average severity in the left leg is {mean_diff_abso_form} points {'higher' if mean_diff > 0 else 'lower'} than in the right leg.")
    write(f"Cohen D : Mean difference ({indx_name_stra} - {colu_name_ordi}): {mean_diff_form} 95%CI between {cilo_form} and {ciup_form}.")
    write(f"Cohen D : The average severity in {indx_name_stra} is {mean_diff_abso_form} points {'higher' if mean_diff > 0 else 'lower'} than in {colu_name_ordi}.")
    write(f"Cohen D : {com2_form}")
    write(f"Cohen D : {com3_form}")

    pass
