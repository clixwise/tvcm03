import numpy as np
import scipy.stats as stats
from scipy.optimize import linprog
import itertools

def contingency_table_probability(table, row_sums, col_sums, total):
    """ Computes the hypergeometric probability of a given contingency table. """
    prob = 1.0
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            prob *= stats.hypergeom.pmf(table[i, j], total, col_sums[j], row_sums[i])
    return prob

def generate_feasible_tables(row_sums, col_sums, total, max_tables=5000):
    """
    Generates feasible contingency tables using integer programming.
    Limits to max_tables for computational efficiency.
    """
    num_rows = len(row_sums)
    num_cols = len(col_sums)
    
    # Store generated tables
    feasible_tables = []
    
    # Generate permutations of row sums
    for perm in itertools.permutations(row_sums):
        table = np.zeros((num_rows, num_cols), dtype=int)
        
        for i in range(num_rows):
            remaining = perm[i]
            for j in range(num_cols - 1):
                upper_bound = min(remaining, col_sums[j])
                table[i, j] = np.random.randint(0, upper_bound + 1)
                remaining -= table[i, j]
            table[i, -1] = remaining  # Ensure row sum matches
        
        if np.all(np.sum(table, axis=0) == col_sums):
            feasible_tables.append(table)
        
        if len(feasible_tables) >= max_tables:
            break
    
    return feasible_tables

def freeman_halton_network_flow(table):
    """
    Computes the Freeman-Halton p-value using network-flow optimization.
    """
    table = np.array(table)  # Ensure table is a NumPy array
    row_sums = np.sum(table, axis=1)
    col_sums = np.sum(table, axis=0)
    total = np.sum(table)

    observed_prob = contingency_table_probability(table, row_sums, col_sums, total)

    # Generate valid contingency tables
    valid_tables = generate_feasible_tables(row_sums, col_sums, total)

    count_extreme = sum(
        contingency_table_probability(t, row_sums, col_sums, total) <= observed_prob
        for t in valid_tables
    )

    p_value = count_extreme / len(valid_tables) if valid_tables else 1.0
    return p_value

# Input contingency table
table = np.array([
    [0,  0,  0,  2, 18,  5,  0, 22],
    [0,  0,  0,  1,  8,  2,  2, 14],
    [0,  0,  1,  0,  3,  2,  1,  0],
    [3,  1,  1, 10, 14,  3,  3, 18],
    [20, 9,  0, 15, 80, 16, 3, 14],
    [9,  2,  1,  3, 17, 23, 7,  5],
    [7,  5,  1,  5,  4,  2, 6,  8],
    [39, 29, 3, 33, 9,  5,  7, 21]
])

# Run the Freeman-Halton test using network-flow optimization
p_value = freeman_halton_network_flow(table)
print(f"Network-Flow Estimated P-value: {p_value}")
