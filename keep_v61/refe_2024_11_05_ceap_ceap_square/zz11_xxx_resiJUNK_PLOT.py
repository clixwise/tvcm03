import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df_tabl is your contingency table DataFrame
# If not, recreate it:
data = [
     [0, 0, 0, 2, 18, 5, 0, 22],
     [0, 0, 0, 1, 8, 2, 2, 14],
     [0, 0, 1, 0, 3, 2, 1, 0],
     [3, 1, 1, 10, 14, 3, 3, 18],
     [20, 9, 0, 15, 80, 16, 3, 14],
     [9, 2, 1, 3, 17, 23, 7, 5],
     [7, 5, 1, 5, 4, 2, 6, 8],
     [39, 29, 3, 33, 9, 5, 7, 21]
 ]
index = pd.Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], name='Lcea')
columns = pd.Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], name='Rcea')
df_tabl = pd.DataFrame(data, index=index, columns=columns)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df_tabl is your contingency table DataFrame
# If not, recreate it as shown in the previous example

# Calculate row and column totals
row_totals = df_tabl.sum(axis=1)
col_totals = df_tabl.sum(axis=0)

# Calculate the difference (Lcea - Rcea)
diff_totals = row_totals - col_totals

# Create a figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))

# Function to add value labels on top of bars
def add_value_labels(ax, spacing=5):
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        label = f"{y_value:.0f}"
        va = 'bottom' if y_value >= 0 else 'top'
        y_pos = y_value + spacing if y_value >= 0 else y_value - spacing
        ax.annotate(label, (x_value, y_pos), ha='center', va=va)

# Function to set y-axis limits with extra space for labels (for positive values only)
def set_y_axis_limit_positive(ax, values):
    max_val = max(values)
    ax.set_ylim(0, max_val * 1.25)  # Add 15% extra space on top

# Function to set y-axis limits with extra space for labels (for positive and negative values)
def set_y_axis_limit_both(ax, values):
    max_val = max(abs(max(values)), abs(min(values)))
    ax.set_ylim(-max_val * 1.15, max_val * 1.40)  # Add 15% extra space on both sides

# Plot row totals (Lcea)
sns.barplot(x=row_totals.index, y=row_totals.values, ax=ax1)
ax1.set_title('Row Marginal Distribution (Lcea)')
ax1.set_xlabel('Lcea')
ax1.set_ylabel('Total')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
set_y_axis_limit_positive(ax1, row_totals)
add_value_labels(ax1)

# Plot column totals (Rcea)
sns.barplot(x=col_totals.index, y=col_totals.values, ax=ax2)
ax2.set_title('Column Marginal Distribution (Rcea)')
ax2.set_xlabel('Rcea')
ax2.set_ylabel('Total')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
set_y_axis_limit_positive(ax2, col_totals)
add_value_labels(ax2)

# Plot difference (Lcea - Rcea)
sns.barplot(x=diff_totals.index, y=diff_totals.values, ax=ax3)
ax3.set_title('Difference (Lcea - Rcea)')
ax3.set_xlabel('Category')
ax3.set_ylabel('Difference')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(axis='y', linestyle='--', alpha=0.7)
set_y_axis_limit_both(ax3, diff_totals)
add_value_labels(ax3)

# Add a horizontal line at y=0 for the difference plot
ax3.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
plt.suptitle(f'Overall Title\nD\nD')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Calculate and print coefficient of variation for row and column totals
row_cv = row_totals.std() / row_totals.mean()
col_cv = col_totals.std() / col_totals.mean()

print(f"Coefficient of Variation for Row Totals (Lcea): {row_cv:.4f}")
print(f"Coefficient of Variation for Column Totals (Rcea): {col_cv:.4f}")



