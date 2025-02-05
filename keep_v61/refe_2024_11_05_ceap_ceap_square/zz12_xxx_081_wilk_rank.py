import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from scipy.stats import wilcoxon, spearmanr, skew
import matplotlib.pyplot as plt

def ceap_to_numeric(ceap_list):
    return max([i for i, x in enumerate(ceap_list) if x == 1])

# -------------------------------
# Wilkoxon Rank Test of Independence
# -------------------------------
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

def wilk_rank(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df1):
    
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

    # Test for symmetry
    differences = indx_list_stra - colu_list_ordi
    skewness = skew(differences)
    skew_threshold = 0.5  # A common threshold for approximate symmetry

    if abs(skewness) >= skew_threshold:
        skewness_form = f"{skewness:.3e}" if skewness < 0.001 else f"{skewness:.3f}"
        sym1_form = f"Symmetry assumption is not met (The test assumes the differences between pairs are symmetrically distributed)"
        sym2_form = f"This test is appropriate when the data doesn't follow a normal distribution."
        print(f"\nData : {what}\nWilkoxon Rank : Skewness:{skewness_form} hence : {sym1_form}")
        print(f"Wilkoxon Rank : {sym2_form}")
        write(f"\nData : {what}\nWilkoxon Rank : Skewness:{skewness_form} hence : {sym1_form}")
        write(f"Wilkoxon Rank : {sym2_form}")
        return

    # Visualize the distribution of differences
    visu = False
    if visu:
        plt.figure(figsize=(10, 6))
        plt.hist(differences, bins=20, edgecolor='black')
        plt.title('Distribution of Differences (Left - Right Severity)')
        plt.xlabel('Difference')
        plt.ylabel('Frequency')
        plt.axvline(x=0, color='red', linestyle='--', label='Zero difference')
        plt.legend()
        plt.show()

    # Exec
    stat, pval = wilcoxon(indx_list_stra, colu_list_ordi)
    seve = len(indx_list_stra) * (len(indx_list_stra) + 1) / 4
    
    # Resu
    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    skewness_form = f"{skewness:.3e}" if skewness < 0.001 else f"{skewness:.3f}"
    seve_form = f"Severity in '{indx_name_stra}' > '{colu_name_ordi}'" if stat < seve else f"Severity in '{indx_name_stra}' < '{colu_name_ordi}'" if stat > seve else f"Severity in '{indx_name_stra}' = '{colu_name_ordi}'"
    sym1_form = f"Symmetry assumption is met (The test assumes the differences between pairs are symmetrically distributed)"
    sym2_form = f"This test is appropriate when the data doesn't follow a normal distribution."
    print(f"\nData : {what}\nWilkoxon Rank : Stat:{stat_form} Pval:{pval_form} Skewness:{skewness_form}")
    print(f"Wilkoxon Rank : {seve_form}")
    print(f"Wilkoxon Rank : Skewness:{skewness_form} hence : {sym1_form}")
    print(f"Wilkoxon Rank : {sym2_form}")
    write(f"\nData : {what}\nWilkoxon Rank : Stat:{stat_form} Pval:{pval_form} Skewness:{skewness_form}")
    write(f"Wilkoxon Rank : {seve_form}")
    write(f"Wilkoxon Rank : Skewness:{skewness_form} hence : {sym1_form}")
    write(f"Wilkoxon Rank : {sym2_form}")

    # Intp
    H0 = f"H0 : The difference between '{indx_name_stra}' and '{colu_name_ordi}' severity is not statistically significant."
    Ha = f"Ha : The difference between '{indx_name_stra}' and '{colu_name_ordi}' severity is statistically significant."
    alpha = 0.05
    if pval < alpha:
        print(f"Wilkoxon Rank : Reject the null hypothesis:\n{Ha}")
        write(f"Wilkoxon Rank : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Wilkoxon Rank : Fail to reject the null hypothesis:\n{H0}")
        write(f"Wilkoxon Rank : Fail to reject the null hypothesis:\n{H0}")
    pass
