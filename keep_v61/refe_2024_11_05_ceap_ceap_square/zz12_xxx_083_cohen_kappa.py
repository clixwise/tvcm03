import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from scipy.stats import wilcoxon, spearmanr, skew

from sklearn.metrics import cohen_kappa_score
import numpy as np
from scipy import stats

def ceap_to_numeric(ceap_list):
    return max([i for i, x in enumerate(ceap_list) if x == 1])

def interpret_kappa(k):
    if k < 0:
        return "Poor agreement"
    elif k < 0.20:
        return "Slight agreement"
    elif k < 0.40:
        return "Fair agreement"
    elif k < 0.60:
        return "Moderate agreement"
    elif k < 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"
    
# -------------------------------
# Cohen kappa Test of Independence
# -------------------------------

def cohe_kapp(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df1):
    
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
    stat = kappa = cohen_kappa_score(indx_list_stra, colu_list_ordi)

    # Calculate the observed agreement
    observed_agreement = np.mean(indx_list_stra == colu_list_ordi)
    # Calculate the expected agreement
    n = len(indx_list_stra)
    n_classes = len(np.unique(np.concatenate([indx_list_stra, colu_list_ordi])))
    expected_agreement = 1 / n_classes
    # Calculate the standard error
    se = np.sqrt((observed_agreement * (1 - observed_agreement)) / (n * (1 - expected_agreement)**2))
    # Calculate z-score
    z = kappa / se
    # Calculate p-value (two-tailed test)
    pval = 2 * (1 - stats.norm.cdf(abs(z)))

    # Resu
    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    intp_form = f"{interpret_kappa(kappa)} between {indx_name_stra} and {colu_name_ordi} severity classifications."
    com2_form = f"The test measures the agreement between two raters (in this case, {indx_name_stra} and {colu_name_ordi} classifications)."
    com3_form = f"The test takes into account the agreement that would be expected by chance. Values closer to 1 indicate stronger agreement."
    print(f"\nData : {what}\nCohen Kappa : Stat:{stat_form} Pval:{pval_form} Intp:{intp_form}") 
    print(f"Cohen Kappa : {com2_form}")
    print(f"Cohen Kappa : {com3_form}")
    write(f"\nData : {what}\nCohen Kappa : Stat:{stat_form} Pval:{pval_form} Intp:{intp_form}") 
    write(f"Cohen Kappa : {com2_form}")
    write(f"Cohen Kappa : {com3_form}")

    # Intp
    H0 = f"H0 : The agreement between '{indx_name_stra}' and '{colu_name_ordi}' severity classifications is not statistically significant."
    Ha = f"Ha : The agreement between '{indx_name_stra}' and '{colu_name_ordi}' severity classifications is statistically significant."
    alpha = 0.05
    if pval < alpha:
        print(f"Cohen Kappa : Reject the null hypothesis:\n{Ha}")
        write(f"Cohen Kappa : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Cohen Kappa : Fail to reject the null hypothesis:\n{H0}")
        write(f"Cohen Kappa : Fail to reject the null hypothesis:\n{H0}")
    pass

   

