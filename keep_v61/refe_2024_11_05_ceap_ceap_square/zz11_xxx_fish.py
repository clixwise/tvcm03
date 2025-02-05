import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
from scipy.stats import hypergeom

def calculate_statistic(table):
    row_sums = table.sum(axis=1)[:, np.newaxis]
    col_sums = table.sum(axis=0)
    total = table.sum()
    expected = np.outer(row_sums, col_sums) / total
    
    with np.errstate(divide='ignore', invalid='ignore'):
        log_ratio = np.where(table != 0, np.log(table / expected), 0)
    
    return 2 * np.sum(table * log_ratio)

def generate_random_table(row_sums, col_sums):
    table = np.zeros((len(row_sums), len(col_sums)), dtype=int)
    total = sum(row_sums)
    for i, row_sum in enumerate(row_sums[:-1]):
        for j, col_sum in enumerate(col_sums[:-1]):
            table[i, j] = np.random.hypergeometric(col_sum, total - col_sum, row_sum)
            row_sum -= table[i, j]
            col_sums[j] -= table[i, j]
            total -= table[i, j]
    table[-1, :] = col_sums
    table[:, -1] = row_sums
    return table

def fisher_freeman_halton(table, num_simulations=10000):
    
    table = np.array(table, dtype=float)
    rows, cols = table.shape
    
    if rows == 2 and cols == 2:
        _, p_value = fisher_exact(table)
    else: # Monte-Carlo
        observed_stat = calculate_statistic(table)
        count = 0
        row_sums = table.sum(axis=1)
        col_sums = table.sum(axis=0)
        
        for _ in range(num_simulations):
            simulated_table = generate_random_table(row_sums, col_sums)
            simulated_stat = calculate_statistic(simulated_table)
            if simulated_stat <= observed_stat:
                count += 1
        
        p_value = count / num_simulations
    
    return p_value

# -------------------------------
# Fisher Halton Test of Independence
# -------------------------------
def fishfreehalt_1(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    
    # Exec
    tabl = df.values.tolist()
    print(tabl)
    perm = 5000
    pval = fisher_freeman_halton(tabl, perm)

    # Resu
    stat_form = f"-"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\nFisher Halton : Stat:{stat_form} Pval:{pval_form}")
    write(f"\nData : {what}\nFisher Halton : Stat:{stat_form} Pval:{pval_form}")  

    # Intp
    H0 = f"H0 : there is no association between the categorical '{colu_name}' and the group '{indx_name}' variables\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"Ha : there is an association between the categorical '{colu_name}' and the group '{indx_name}' variables\n({colu_cate_list}) vs ({indx_cate_list})"
    alpha = 0.05
    if pval < alpha:
        print(f"Fisher Halton : Reject the null hypothesis:\n{Ha}")
        write(f"Fisher Halton : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Fisher Halton : Fail to reject the null hypothesis:\n{H0}")
        write(f"Fisher Halton : Fail to reject the null hypothesis:\n{H0}")
    pass

def fishfreehalt_2(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    
    def freeman_halton_test(table):
        # Calculate the row and column sums
        row_sums = np.sum(table, axis=1)
        col_sums = np.sum(table, axis=0)
        total = np.sum(table)

        # Calculate the expected frequencies
        expected = np.outer(row_sums, col_sums) / total

        # Calculate the p-value using the hypergeometric distribution
        p_value = 1.0
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                p_value *= hypergeom.pmf(table[i, j], total, col_sums[j], row_sums[i])
        return p_value        
    
    # Exec
    # ----
    tabl = df.to_numpy()
    p_value = freeman_halton_test(tabl)
    pass
def fishfreehalt(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    fishfreehalt_1(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name)
    fishfreehalt_2(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name)
if __name__ == "__main__":
    # Example usage
    table = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    print(table)

    # Create the DataFrame
    df = pd.DataFrame({
        'Column1': [1, 4, 7],
        'Column2': [2, 5, 8],
        'Column3': [3, 6, 9]
    })
    table = df.values.tolist()
    print(table)

    result = fisher_freeman_halton(table)
    print(f"Pval: {result}")