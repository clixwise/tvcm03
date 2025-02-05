import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats

def ceap_to_numeric(ceap_list):
    return max([i for i, x in enumerate(ceap_list) if x == 1])

# -------------------------------
# Kolmogorov-Smirnov
# -------------------------------
def kolmsmir(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df1):

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
    stat, pval = stats.ks_2samp(indx_list_stra, colu_list_ordi)

    # Resu
    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\nKolmogorov-Smirnov : Stat:{stat_form} Pval:{pval_form} Note : Stat is the maximum distance between the cumulative distributions")
    write(f"\nData : {what}\nKolmogorov-Smirnov : Stat:{stat_form} Pval:{pval_form}")  
    print(f"Kolmogorov-Smirnov : test is sensitive to both the location and shape of the distributions.")
    write(f"Kolmogorov-Smirnov : test is sensitive to both the location and shape of the distributions.")

    # Intp
    # Mistral
    # H0 : the distributions of the severity scores for the left and right sides are the same.
    # Ha : the distributions of the severity scores for the left and right sides are different
    H0 = f"H0 : There is no difference in '{colu_name}' distribution between '{indx_name}' groups\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"Ha : There is a difference in '{colu_name}' distribution between '{indx_name}' groups\n({colu_cate_list}) vs ({indx_cate_list})"
    alpha = 0.05
    if pval < alpha:
        print(f"Kolmogorov-Smirnov : Reject the null hypothesis:\n{Ha}")
        write(f"Kolmogorov-Smirnov : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Kolmogorov-Smirnov : Fail to reject the null hypothesis:\n{H0}")
        write(f"Kolmogorov-Smirnov : Fail to reject the null hypothesis:\n{H0}")
    pass
'''
The Kolmogorov-Smirnov test is not appropriate for comparing two ordinal variables. While it can be used for ordinal data in some contexts, it has limitations when applied to two ordinal variables:

1. The test is designed for continuous distributions[3]. Ordinal data is discrete by nature, which violates this assumption.

2. For ordinal variables, the intervals between categories are arbitrary, making it meaningless to test for normality or compare distributions in the way the Kolmogorov-Smirnov test does[2].

3. The distribution of the Kolmogorov-Smirnov statistic relies on continuity and will be highly conservative when used on discrete data[2].

4. The test's power is reduced when applied to non-continuous data[2].

For comparing two ordinal variables, other tests are more appropriate, such as:

1. Wilcoxon Rank-Sum test (for independent samples)
2. Wilcoxon Signed-Rank test (for paired samples)
3. Mann-Whitney U test

These tests are specifically designed to handle ordinal data and provide more reliable results in this context[4][5].

Citations:
[1] https://davidmlane.com/hyperstat/viswanathan/Kolmogorov.html
[2] https://stats.stackexchange.com/questions/95153/how-to-interpret-ks-test-or-shapiro-wilk-test-for-ordinal-criterion-variable
[3] https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
[4] https://www.quantitativeskills.com/sisa/statistics/ordhlp.htm
[5] https://mark-me.github.io/statistical-tests-ordinal/
[6] https://www.statisticssolutions.com/kolmogorov-smrinovs-one-sample-test/
[7] https://arize.com/blog-course/kolmogorov-smirnov-test/
[8] https://www.spss-tutorials.com/spss-kolmogorov-smirnov-test-for-normality/comment-page-2/
'''