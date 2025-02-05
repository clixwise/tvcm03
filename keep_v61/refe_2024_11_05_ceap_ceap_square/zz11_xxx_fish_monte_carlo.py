import numpy as np
import scipy.stats as stats
from itertools import permutations

def generate_random_table(row_sums, col_sums, total):
    """ Generates a random table with fixed row/col margins using iterative proportional fitting. """
    table = np.zeros((len(row_sums), len(col_sums)), dtype=int)
    
    for i in range(len(row_sums)):
        row_remaining = row_sums[i]
        for j in range(len(col_sums) - 1):
            upper_bound = min(row_remaining, col_sums[j])
            if upper_bound > 0:
                table[i, j] = np.random.randint(0, upper_bound + 1)
            row_remaining -= table[i, j]
        table[i, -1] = row_remaining  # Ensure row sum matches
    
    return table

def contingency_table_probability(table, row_sums, col_sums, total):
    """ Computes the hypergeometric probability of a given contingency table. """
    prob = 1.0
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            prob *= stats.hypergeom.pmf(table[i, j], total, col_sums[j], row_sums[i])
    return prob

def freeman_halton_monte_carlo(table, num_simulations=10000):
    """ Monte Carlo estimation of Freeman-Halton test p-value. """
    row_sums = np.sum(table, axis=1)
    col_sums = np.sum(table, axis=0)
    total = np.sum(table)

    observed_prob = contingency_table_probability(table, row_sums, col_sums, total)

    count_extreme = 0
    for _ in range(num_simulations):
        simulated_table = generate_random_table(row_sums, col_sums, total)
        simulated_prob = contingency_table_probability(simulated_table, row_sums, col_sums, total)
        if simulated_prob <= observed_prob:
            count_extreme += 1

    p_value = count_extreme / num_simulations
    return p_value

# Input table
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

# Run the Freeman-Halton Monte Carlo test
p_value = freeman_halton_monte_carlo(table)
print(f"Monte Carlo Estimated P-value: {p_value}")
