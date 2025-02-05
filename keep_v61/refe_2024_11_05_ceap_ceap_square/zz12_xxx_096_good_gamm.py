import numpy as np
import pandas as pd
from util_file_mngr import write
import numpy as np
from scipy import stats
    
# -------------------------------
# Goodman and Kruskal's Gamma
# -------------------------------
def good_gamm(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):

    # Trac
    # ----
    trac = False
    
    # Prec : flatten the contingency table to create pairs of observations
    # ----
    def goodman_kruskal_gamma_fram_rank_prec(df): # Open ai
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
        
        return ceap_ranks, age_ranks 
   
    # Exec Stat : Goodman and Kruskal's Gamma [frame approach]
    # ----
    def goodman_kruskal_gamma_fram_stat(df): # Open ai
        concordant = 0
        discordant = 0

        # Iterate over all row and column pairs
        for i, ceap_i in enumerate(df.index):
            for j, ceap_j in enumerate(df.index):
                if i >= j:  # Only consider upper triangular part for rows
                    continue
                for k, age_k in enumerate(df.columns):
                    for l, age_l in enumerate(df.columns):
                        if k >= l:  # Only consider upper triangular part for columns
                            continue
                        
                        # Count pairs based on the contingency table
                        count_ij_kl = df.loc[ceap_i, age_k] * df.loc[ceap_j, age_l]
                        count_ij_lk = df.loc[ceap_i, age_l] * df.loc[ceap_j, age_k]

                        # Check concordant and discordant conditions
                        if k < l:  # Concordant pair
                            concordant += count_ij_kl
                        if k > l:  # Discordant pair
                            discordant += count_ij_lk

        # Calculate Gamma
        gamma = (concordant - discordant) / (concordant + discordant) if (concordant + discordant) > 0 else 0
        return gamma
    
    # Exec Stat : Goodman and Kruskal's Gamma [rank 1 approach]
    # ----
    def goodman_kruskal_gamma_rank_sta1(ceap_ranks, age_ranks): # Open ai
        n = len(ceap_ranks)
        concordant = 0
        discordant = 0
        # Iterate over all pairs of observations
        for i in range(n):
            for j in range(i + 1, n):  # Consider each pair (i, j) where i < j
                if (ceap_ranks[i] < ceap_ranks[j] and age_ranks[i] < age_ranks[j]) or \
                (ceap_ranks[i] > ceap_ranks[j] and age_ranks[i] > age_ranks[j]):
                    concordant += 1
                elif (ceap_ranks[i] < ceap_ranks[j] and age_ranks[i] > age_ranks[j]) or \
                    (ceap_ranks[i] > ceap_ranks[j] and age_ranks[i] < age_ranks[j]):
                    discordant += 1
        gamma = (concordant - discordant) / (concordant + discordant) if (concordant + discordant) > 0 else 0
        return gamma
    
    # Exec Stat : Goodman and Kruskal's Gamma [rank 2 approach]
    # ----
    def goodman_kruskal_gamma_rank_sta2(x, y): # Mistral
        n = len(x)
        n_concordant = np.sum((x[:, np.newaxis] > x) & (y[:, np.newaxis] > y))
        n_discordant = np.sum((x[:, np.newaxis] > x) & (y[:, np.newaxis] < y))
        gamma = (n_concordant - n_discordant) / (n_concordant + n_discordant)
        return gamma

    # Exec Pval : Goodman and Kruskal's Gamma
    # ----
    '''
    To compute the p-value for Goodman and Kruskal's Gamma, you typically need to perform a permutation test. 
    This involves repeatedly shuffling one of the variables and recomputing the Gamma statistic to build 
    a distribution of Gamma values under the null hypothesis (no association). 
    The p-value is then the proportion of these permuted Gamma values that are as extreme as or more extreme than the observed Gamma value.
    '''  
    def goodman_kruskal_gamma_fram_perm(df, n_permutations=1000):
        np.random.seed(42)
        observed_gamma = goodman_kruskal_gamma_fram_stat(df)
        permuted_gammas = []

        for _ in range(n_permutations):
            # Shuffle rows and columns independently
            shuffled_df = df.sample(frac=1, axis=0).sample(frac=1, axis=1).reset_index(drop=True)
            permuted_gamma = goodman_kruskal_gamma_fram_stat(shuffled_df)
            permuted_gammas.append(permuted_gamma)

        # Two-sided p-value
        permuted_gammas = np.array(permuted_gammas)
        p_value = np.mean(np.abs(permuted_gammas) >= np.abs(observed_gamma))
        return p_value # permuted_gammas
    
    def goodman_kruskal_gamma_rank_perm(x, y, statistic_func, num_permutations=1000):
        np.random.seed(42)
        observed_stat = statistic_func(x, y)
        permuted_stats = []

        for _ in range(num_permutations):
            y_permuted = np.random.permutation(y)
            permuted_stat = statistic_func(x, y_permuted)
            permuted_stats.append(permuted_stat)

        permuted_stats = np.array(permuted_stats)
        p_value = np.mean(np.abs(permuted_stats) >= np.abs(observed_stat))
        return p_value
    
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
    perm_coun = 2000
    fram_bool = False # !!! DOES NOT WORK !!!
    if fram_bool:
        gamm = gamm_fram = goodman_kruskal_gamma_fram_stat(df)
        pval = goodman_kruskal_gamma_fram_perm(df, perm_coun)
    rank_1_bool = False
    if rank_1_bool: # WORKS OK
        ceap_ranks, age_ranks = goodman_kruskal_gamma_fram_rank_prec(df)
        gamm = gamm_rank_1 = goodman_kruskal_gamma_rank_sta1(np.array(ceap_ranks), np.array(age_ranks))
        pval = goodman_kruskal_gamma_rank_perm(np.array(ceap_ranks), np.array(age_ranks), goodman_kruskal_gamma_rank_sta1, perm_coun)
    rank_2_bool = True
    if rank_2_bool: # WORKS OK
        ceap_ranks, age_ranks = goodman_kruskal_gamma_fram_rank_prec(df)
        gamm = gamm_rank_2 = goodman_kruskal_gamma_rank_sta2(np.array(ceap_ranks), np.array(age_ranks))
        pval = goodman_kruskal_gamma_rank_perm(np.array(ceap_ranks), np.array(age_ranks), goodman_kruskal_gamma_rank_sta2, perm_coun)

    # Intp
    # ----
    gamm_intp = interpret_statistic(gamm)

    # Resu
    stat_form = f"{gamm:.3e}" if gamm < 0.001 else f"{gamm:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\nGoodman and Kruskal's Gamma [2025_01_19] : Stat:{stat_form} Intp: {gamm_intp} Pval:{pval_form}")
    write(f"\nData : {what}\nGoodman and Kruskal's Gamma [2025_01_19] : Stat:{stat_form} Intp: {gamm_intp} Pval:{pval_form}")

    # Intp
    H0 = f"H0 : there is no association between the categorical '{colu_name}' and the group '{indx_name}' variables"
    H0 += f"\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"Ha : there is a association between the categorical '{colu_name}' and the group '{indx_name}' variables"
    Ha += f"\n({colu_cate_list}) vs ({indx_cate_list})"
    alpha = 0.05
    if pval < alpha:
        print(f"Goodman and Kruskal's Gamma [2025_01_19] : Reject the null hypothesis:\n{Ha}")
        write(f"Goodman and Kruskal's Gamma [2025_01_19] : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Goodman and Kruskal's Gamma [2025_01_19] : Fail to reject the null hypothesis:\n{H0}")
        write(f"Goodman and Kruskal's Gamma [2025_01_19] : Fail to reject the null hypothesis:\n{H0}")
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