import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from scipy.stats import spearmanr

# -------------------------------
# Spearman's Rank Test of Independence
# -------------------------------

def spea(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df1):
    
    # Trac
    trac = True
        
    # Prec
    df2 = df1.sort_values(by=indx_name_stra) # note : same 'stat, pval' whether sorted or not   
    indx_list_stra = df1[indx_name_stra]# df2['Gender_num'] = df2['Gender'].map({'Male': 0, 'Female': 1})
    colu_list_ordi = df1[colu_name_ordi]
    if trac:
        print(f"\nStep 1 : indx_list_stra.size:{len(indx_list_stra)} df2.type:{type(indx_list_stra)}\n{indx_list_stra}\n:{indx_list_stra.index}")
        write(f"\nStep 1 : indx_list_stra.size:{len(indx_list_stra)} df2.type:{type(indx_list_stra)}\n{indx_list_stra}\n:{indx_list_stra.index}")
        print(f"\nStep 2 : colu_list_ordi.size:{len(colu_list_ordi)} df2.type:{type(colu_list_ordi)}\n{colu_list_ordi}\n:{colu_list_ordi.index}")
        write(f"\nStep 2 : colu_list_ordi.size:{len(colu_list_ordi)} df2.type:{type(colu_list_ordi)}\n{colu_list_ordi}\n:{colu_list_ordi.index}")
    # Exec
    stat, pval = spearmanr(indx_list_stra, colu_list_ordi)
    
    # Resu
    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    asso_form = "positive LE 1" if stat > 0 else "negative GE -1" if stat < 0 else "none"
    print(f"\nData : {what}\nSpearman's Rank : Stat:{stat_form} Pval:{pval_form} Asso:{asso_form}")
    write(f"\nData : {what}\nSpearman's Rank : Stat:{stat_form} Pval:{pval_form} Asso:{asso_form}")  
   
    # Intp
    # Mistral
    # H0 = "H0 : there is no association between the severity scores for the left and right sides."
    # Ha = "Ha : there is an association between the severity scores for the left and right sides."
    H0 = f"H0 : There is no monotonic relationship between the two variables '{indx_name_stra}' and '{colu_name_ordi}' : Rho EQ 0."
    Ha = f"Ha : There is a monotonic relationship between the two variables '{indx_name_stra}' and '{colu_name_ordi}' : Rho NE 0."
    alpha = 0.05
    if pval < alpha:
        print(f"Spearman's Rank : Reject the null hypothesis:\n{Ha}")
        write(f"Spearman's Rank : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Spearman's Rank : Fail to reject the null hypothesis:\n{H0}")
        write(f"Spearman's Rank : Fail to reject the null hypothesis:\n{H0}")
    pass
'''
Yes, the Wilcoxon Rank Test pertains to tests for two ordinal variables. Specifically, the Wilcoxon Signed-Rank Test is designed for comparing two related samples of ordinal data[3][4]. This test is appropriate when:

1. The data is measured at the ordinal level[5].
2. The samples are paired or related, such as before and after measurements on the same subjects[3][6].
3. The differences between pairs can be ranked[7].

The test evaluates whether there are significant differences between two sets of ordinal measurements taken on the same subjects or under matched conditions[3][6]. It's particularly useful for analyzing data from questionnaires with ordered scales or when comparing results across a period of time[3].

However, it's important to note that while the Wilcoxon Signed-Rank Test can be used with ordinal data, it requires the ability to determine the magnitude and direction of differences between pairs[7]. This can sometimes be challenging with purely ordinal data, and in such cases, alternatives like the sign test might be more appropriate[7].

Citations:
[1] https://www.uv.es/visualstats/vista-frames/help/lecturenotes/lecture09/lec9part4.html
[2] https://pubmed.ncbi.nlm.nih.gov/8731005/
[3] https://mark-me.github.io/statistical-tests-ordinal/
[4] https://peterstatistics.com/CrashCourse/4-TwoVarPair/OrdOrd/OrdOrdPair3a.html
[5] https://statistics.laerd.com/spss-tutorials/wilcoxon-signed-rank-test-using-spss-statistics.php
[6] https://instruct.uwo.ca/geog/201/wilcoxon.htm
[7] https://stats.stackexchange.com/questions/47168/is-ordinal-or-interval-data-required-for-the-wilcoxon-signed-rank-test
[8] https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/wilcoxon-sign-test/
'''