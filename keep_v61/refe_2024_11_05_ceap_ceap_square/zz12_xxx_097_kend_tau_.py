import numpy as np
import pandas as pd
from util_file_mngr import write
import numpy as np
from scipy import stats

# -------------------------------
# Kendall Tau
# -------------------------------
def kend_tau_(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):

    # Trac
    # ----
    trac = False
    
    # Prec : flatten the contingency table to create pairs of observations
    # ----
    ceap_severity = []
    age_bins = []
    df = df.astype(int)
    for ceap in df.index:
        for age_bin in df.columns:
            count = df.loc[ceap, age_bin]
            ceap_severity.extend([ceap] * count)
            age_bins.extend([age_bin] * count)

    # Convert to ordinal ranks
    ceap_ranks = pd.Categorical(ceap_severity, categories=df.index, ordered=True).codes
    age_ranks = pd.Categorical(age_bins, categories=df.columns, ordered=True).codes

    # Exec intp
    # ----
    def interpret_statistic(value):
        sign = "+" if value >= 0 else "-"
        if value == 0:
            return "0/5: No association"
        elif abs(value) <= 0.2:
            return f"{sign}1/5"
        elif abs(value) <= 0.4:
            return f"{sign}2/5"
        elif abs(value) <= 0.6:
            return f"{sign}3/5"
        elif abs(value) <= 0.8:
            return f"{sign}4/5"
        else:
            return f"{sign}5/5"
    
    # Exec
    # ----
    tau_b, pval = stats.kendalltau(ceap_ranks, age_ranks)
    stat = tau_b

    # Intp
    # ----
    intp = interpret_statistic(tau_b)

    # Resu
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\nKendall Tau [2025_01_19] : Stat:{stat_form} Intp: {intp} Pval:{pval_form}")
    write(f"\nData : {what}\nKendall Tau [2025_01_19] : Stat:{stat_form} Intp: {intp} Pval:{pval_form}")

    # Intp
    H0 = f"H0 : there is no association between the categorical '{colu_name}' and the group '{indx_name}' variables"
    H0 += f"\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"Ha : there is a association between the categorical '{colu_name}' and the group '{indx_name}' variables"
    Ha += f"\n({colu_cate_list}) vs ({indx_cate_list})"
    alpha = 0.05
    if pval < alpha:
        print(f"Kendall Tau [2025_01_19] : Reject the null hypothesis:\n{Ha}")
        write(f"Kendall Tau [2025_01_19] : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Kendall Tau [2025_01_19] : Fail to reject the null hypothesis:\n{H0}")
        write(f"Kendall Tau [2025_01_19] : Fail to reject the null hypothesis:\n{H0}")
    pass
'''
You're absolutely correct. Spearman's rank correlation, Goodman and Kruskal's gamma, and Kendall's tau are specifically designed for ordinal data or continuous data that can be ranked. These tests rely on the ability to rank the data in a meaningful order. If one of the variables is not ordinal (for example, if it's nominal), these tests would not be appropriate. Here's a breakdown:

1. Spearman's rank correlation:
   - Requires both variables to be ordinal or continuous
   - Measures monotonic relationships

2. Goodman and Kruskal's gamma:
   - Designed for two ordinal variables
   - Measures the strength of association between two ordinal variables

3. Kendall's tau:
   - Requires both variables to be ordinal or continuous
   - Measures the strength of dependence between two variables

If one of your variables is not ordinal, you would need to consider other statistical methods appropriate for the data types you're working with. For example:

- Chi-square test of independence: For two categorical variables (including nominal)
- Point-biserial correlation: For one dichotomous variable and one continuous variable
- One-way ANOVA: For one categorical independent variable and one continuous dependent variable

Always ensure that the statistical test you choose is appropriate for the types of variables you're analyzing.
'''
'''
Yes, the Wilcoxon Rank Test pertains to tests for two ordinal variables. 
Specifically, the Wilcoxon Signed-Rank Test is designed for comparing two related samples of ordinal data[3][4]. This test is appropriate when:

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