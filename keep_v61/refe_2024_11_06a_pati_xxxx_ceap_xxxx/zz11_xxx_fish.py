import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy.stats import fisher_exact

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

def fisher_freeman_halton(df, num_simulations=10000, seed=42):
    
    if seed is not None:
        np.random.seed(seed)

    trac = True

    # Prec
    if trac: print(df)
    tabl1 = df.values.tolist()
    if trac: print(tabl1)
    tabl2 = np.array(tabl1, dtype=float)
    if trac: print(tabl2)
    rows, cols = tabl2.shape
    if trac: print(rows, cols)

    # Exec
    if rows == 2 and cols == 2:
        _, p_value = fisher_exact(tabl2) # Fisher Exact
    else:
        observed_stat = calculate_statistic(tabl2) # Monte-Carlo
        count = 0
        row_sums = tabl2.sum(axis=1)
        col_sums = tabl2.sum(axis=0)
        for _ in range(num_simulations):
            simulated_tabl = generate_random_table(row_sums, col_sums)
            simulated_stat = calculate_statistic(simulated_tabl)
            if simulated_stat <= observed_stat:
                count += 1
        p_value = count / num_simulations

    # Exit
    if trac: print(p_value)
    return p_value

# -------------------------------
# Fisher Exact Test of Independence
# -------------------------------
def fishfreehalt(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    
    # Exec
    perm = 10000
    seed = 42
    pval = fisher_freeman_halton(df, perm, seed)

    # Resu
    stat_form = f"-"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\nFisher Exact : Stat:{stat_form} Pval:{pval_form}")
    write(f"\nData : {what}\nFisher Exact : Stat:{stat_form} Pval:{pval_form}")  

    # Intp
    H0 = f"H0 : there is no association between the categorical '{colu_name}' and the group '{indx_name}' variables\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"Ha : there is an association between the categorical '{colu_name}' and the group '{indx_name}' variables\n({colu_cate_list}) vs ({indx_cate_list})"
    alpha = 0.05
    if pval < alpha:
        print(f"Fisher Exact : Reject the null hypothesis:\n{Ha}")
        write(f"Fisher Exact : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"Fisher Exact : Fail to reject the null hypothesis:\n{H0}")
        write(f"Fisher Exact : Fail to reject the null hypothesis:\n{H0}")
    pass

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