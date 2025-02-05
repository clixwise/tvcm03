import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from scipy.stats import wilcoxon, spearmanr, skew
from scipy.stats import chi2
from sklearn.metrics import cohen_kappa_score
import numpy as np
from scipy import stats
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils import resample

def jaccard_similarity(list1, list2):
    set1 = set([i for i, x in enumerate(list1) if x == 1])
    set2 = set([i for i, x in enumerate(list2) if x == 1])
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 1  # Return 1 if both sets are empty
    
# -------------------------------
# Jaccard similarity Test of Independence
# -------------------------------

def jacc(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df1):
    
    # Exec : Calculate Jaccard similarity for each patient
    df1['jaccard_similarity'] = df1.apply(lambda row: jaccard_similarity(row[indx_name_stra], row[colu_name_ordi]), axis=1)
    # Calculate average Jaccard similarity
    stat = average_similarity = df1['jaccard_similarity'].mean()

    # 1. Wilcoxon signed-rank test
    wilcoxon_statistic, wilcoxon_p = stats.wilcoxon(df1['jaccard_similarity'] - 0.5)
    wilc_stat = wilcoxon_statistic
    wilc_pval = wilcoxon_p

    # 2. Bootstrap confidence interval
    n_bootstrap = 10000
    bootstrap_means = [resample(df1['jaccard_similarity']).mean() for _ in range(n_bootstrap)]
    ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])
    boot_mean = bootstrap_means
    boot_cilo = ci_lower
    boot_ciup = ci_upper

    # 3. Permutation test
    n_permutations = 10000
    observed_mean = df1['jaccard_similarity'].mean()
    permutation_means = [np.random.permutation(df1['jaccard_similarity']).mean() for _ in range(n_permutations)]
    perm_pval = p_value_permutation = sum(permutation_mean >= observed_mean for permutation_mean in permutation_means) / n_permutations

    # Resu
    if np.isnan(stat):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    wilc_stat_form = f"{wilc_stat:.3e}" if wilc_stat < 0.001 else f"{wilc_stat:.3f}"
    wilc_pval_form = f"{wilc_pval:.3e}" if wilc_pval < 0.001 else f"{wilc_pval:.3f}"
    wilc_comm = "Tests if the median Jaccard similarity is significantly different from 0.5. A small p-value suggests a significant difference."
    boot_cilo_form = f"{boot_cilo:.3e}" if boot_cilo < 0.001 else f"{boot_cilo:.3f}"
    boot_ciup_form = f"{boot_ciup:.3e}" if boot_ciup < 0.001 else f"{boot_ciup:.3f}"
    boot_comm = "Provides a range of plausible CI values for the true population mean Jaccard similarity. If 0.5 is not in this interval, it suggests a significant difference from 0.5."
    perm_pval_form = f"{perm_pval:.3e}" if perm_pval < 0.001 else f"{perm_pval:.3f}"
    perm_comm = "Tests if the observed mean Jaccard similarity is significantly different from what would be expected by chance. A small p-value suggests a significant difference."
    print(f"\nData : {what}\nJaccard similarity : Stat:{stat_form} average Jaccard Similarity between {indx_name_stra} and {colu_name_ordi} CEAP classifications")
    print(f"Jaccard similarity : Wilcoxon    Stat :{wilc_stat_form} Pval:{wilc_pval_form}")
    print(f"Jaccard similarity : Bootstrap   95%CI:{boot_cilo_form} - {boot_ciup_form}")      
    print(f"Jaccard similarity : Permutation Pval :{perm_pval_form}")
    print(f"Jaccard similarity : Wilcoxon   : {wilc_comm}")
    print(f"Jaccard similarity : Bootstrap  : {boot_comm}")
    print(f"Jaccard similarity : Permutation: {perm_comm}")
    write(f"\nData : {what}\nJaccard similarity : Stat:{stat_form} average Jaccard Similarity between {indx_name_stra} and {colu_name_ordi} CEAP classifications")
    write(f"Jaccard similarity : Wilcoxon    Stat :{wilc_stat_form} Pval:{wilc_pval_form}")
    write(f"Jaccard similarity : Bootstrap   95%CI:{boot_cilo_form} - {boot_ciup_form}")         
    write(f"Jaccard similarity : Permutation Pval :{perm_pval_form}")
    write(f"Jaccard similarity : Wilcoxon   : {wilc_comm}")
    write(f"Jaccard similarity : Bootstrap  : {boot_comm}")
    write(f"Jaccard similarity : Permutation: {perm_comm}")
    pass
