    
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_qol_32_grph_plot_ import PlotTranQOL_32_assu_resi
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
from statsmodels.nonparametric.smoothers_lowess import lowess

# ----
# Plot 
# https://copilot.microsoft.com/shares/7rZLRYu1wYqmA3sQABBX4
# https://gemini.google.com/app/c1bf73ba66b18b6b
# ----

def plot_assu_resi_cook(plot_tran_mixd: PlotTranQOL_32_assu_resi) -> None:
    # from qol_30.c02_qol_32_grph_plot_ import PlotTranQOL_32_assu_resi

    trac = True

    # Data
    # ---- 
    result = plot_tran_mixd.fram_dict["orig_fram"]
    print(result)
     
    # ----
    # Grph
    # ---- 
    plt.style.use('default') # fig, ax = plt.subplots(figsize=(11,8)) # width, height

    # Figu
    # ----
    fig = plot_tran_mixd.figu
    ax = plot_tran_mixd.axis
    ax.clear()
     
    # Grph
    # ----
    line_widt = plot_tran_mixd.line_widt
    line_styl = plot_tran_mixd.line_styl
    line_colo = plot_tran_mixd.line_colo
    line_alph = plot_tran_mixd.line_alph
    line_labl = plot_tran_mixd.line_labl
    mark_size = plot_tran_mixd.mark_size
    mark_widt = plot_tran_mixd.mark_widt
    mark_colo = plot_tran_mixd.mark_colo
    capp_size = plot_tran_mixd.capp_size
    erro_colo = plot_tran_mixd.erro_colo

    # Exec
    # ----
    resid = result.resid
    # 1. Calculate the values
    std_resid = resid / np.sqrt(result.scale)
    p = len(result.params)
    cooks_d = (std_resid**2) / p

    # 1. Calculate stats
    threshold = 4 / len(resid)
    influential_indices = np.where(cooks_d > threshold)[0]
    n_influential = len(influential_indices)
    pct_influential = (n_influential / len(resid)) * 100

    # 2. Plotting with ax
    markerline, stemlines, baseline = ax.stem(
        cooks_d.index, 
        cooks_d, 
        markerfmt='o'
    )

    # Apply your skyblue/navy theme
    plt.setp(markerline, markerfacecolor='skyblue', markeredgecolor='navy')
    plt.setp(stemlines, color='lightgrey', alpha=0.7)

    # 3. Add threshold line with dynamic legend label
    # Using f-string for: count + percentage
    legend_label = f'Threshold ({threshold:.2f}): {n_influential} ({pct_influential:.1f}%)'
    ax.axhline(threshold, color='red', linestyle='--', linewidth=1, label=legend_label)

    # 4. Formatting
    ax.set_title("Cook's Distance (Approximation)")
    ax.set_xlabel("Observation Index")
    ax.set_ylabel("Cook's D")
    
    ax.grid(True, alpha=0.3)

    print(f"Potentially influential observations: {influential_indices}")
    
    # Lgnd
    # ---- 
    ax.legend(loc='upper right', frameon=True)
         
    # Titl
    # ----
    titl = plot_tran_mixd.titl # "VEINES-QOL over time"
    ax.set_title(titl)
    # fig.tight_layout(): See 'FiguTran' initiaization : self.fig = plt.figure(layout="constrained")
    # fig.show()
    pass

# 1. Residuals vs. Fitted Plot
# This checks linearity and homoscedasticity.
def plot_assu_resi_fitt(plot_tran_mixd: PlotTranQOL_32_assu_resi) -> None:
    # from qol_30.c02_qol_32_grph_plot_ import PlotTranQOL_32_assu_resi

    trac = True

    # Data
    # ---- 
    result = plot_tran_mixd.fram_dict["orig_fram"]
    print(result)
  
    # ----
    # Grph
    # ---- 
    plt.style.use('default') # fig, ax = plt.subplots(figsize=(11,8)) # width, height

    # Figu
    # ----
    fig = plot_tran_mixd.figu
    ax = plot_tran_mixd.axis
    ax.clear()
     
    # Grph
    # ----
    line_widt = plot_tran_mixd.line_widt
    line_styl = plot_tran_mixd.line_styl
    line_colo = plot_tran_mixd.line_colo
    line_alph = plot_tran_mixd.line_alph
    line_labl = plot_tran_mixd.line_labl
    mark_size = plot_tran_mixd.mark_size
    mark_widt = plot_tran_mixd.mark_widt
    mark_colo = plot_tran_mixd.mark_colo
    capp_size = plot_tran_mixd.capp_size
    erro_colo = plot_tran_mixd.erro_colo
    
    '''
    A flat red line at zero: Perfect. Your model assumptions are well-met.

    A wavy or curved red line: This suggests a non-linear relationship 
    that your current model isn't capturing (you might need to log-transform a variable or add a quadratic term).
    '''

    # Exec
    # ----
    resid = result.resid
    fitted = result.fittedvalues

    # 1. Plot the scatter points (consistent theme)
    ax.scatter(fitted, resid, alpha=0.6, color='skyblue', edgecolor='navy', label='Residuals')

    # 2. Add the horizontal zero line
    ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.8)

    # 3. Calculate Lowess smoothing
    # Returns: [x_sorted, y_fitted]
    smoothed = lowess(resid, fitted, frac=0.3) 

    # 4. Plot the Lowess trend line
    ax.plot(smoothed[:, 0], smoothed[:, 1], color='red', linewidth=2, label='Lowess Trend')

    # 5. Formatting
    ax.set_title("Residuals vs. Fitted (with Trend Line)", fontsize=13)
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.grid(True, alpha=0.3)
    

    # Optional: Symmetrize y-axis to make the zero line the center
    limit = max(abs(resid.min()), abs(resid.max())) * 1.2
    ax.set_ylim(-limit, limit)

    
    # Lgnd
    # ---- 
    ax.legend(loc='best', fontsize='small')
         
    # Titl
    # ----
    titl = plot_tran_mixd.titl # "VEINES-QOL over time"
    ax.set_title(titl)
    # fig.tight_layout(): See 'FiguTran' initiaization : self.fig = plt.figure(layout="constrained")
    # fig.show()
    pass

#2. Scale–Location Plot (a.k.a. Spread–Location Plot)
# This checks variance homogeneity more sensitively.
def plot_assu_resi_scal(plot_tran_mixd: PlotTranQOL_32_assu_resi) -> None:
    # from qol_30.c02_qol_32_grph_plot_ import PlotTranQOL_32_assu_resi
    trac = True

    # Data
    # ---- 
    result = plot_tran_mixd.fram_dict["orig_fram"]
    print(result)
  
    # ----
    # Grph
    # ---- 
    plt.style.use('default') # fig, ax = plt.subplots(figsize=(11,8)) # width, height

    # Figu
    # ----
    fig = plot_tran_mixd.figu
    ax = plot_tran_mixd.axis
    ax.clear()
     
    # Grph
    # ----
    line_widt = plot_tran_mixd.line_widt
    line_styl = plot_tran_mixd.line_styl
    line_colo = plot_tran_mixd.line_colo
    line_alph = plot_tran_mixd.line_alph
    line_labl = plot_tran_mixd.line_labl
    mark_size = plot_tran_mixd.mark_size
    mark_widt = plot_tran_mixd.mark_widt
    mark_colo = plot_tran_mixd.mark_colo
    capp_size = plot_tran_mixd.capp_size
    erro_colo = plot_tran_mixd.erro_colo
    
    '''
    A flat red line at zero: Perfect. Your model assumptions are well-met.

    A wavy or curved red line: This suggests a non-linear relationship 
    that your current model isn't capturing (you might need to log-transform a variable or add a quadratic term).
    '''

    # Exec
    # ----
    resid = result.resid
    fitted = result.fittedvalues


    # 1. Calculate values
    sqrt_abs_resid = np.sqrt(np.abs(resid))

    # 2. Create the scatter plot
    ax.scatter(
        fitted, 
        sqrt_abs_resid, 
        alpha=0.6, 
        color='skyblue', 
        edgecolor='navy', 
        label='Residuals'
    )

    # 3. Add Lowess trend line
    # (Better than a mean line for detecting variance changes)
    smoothed_sl = lowess(sqrt_abs_resid, fitted, frac=0.3)
    ax.plot(smoothed_sl[:, 0], smoothed_sl[:, 1], color='red', linewidth=2, label='Trend')

    # 4. Formatting using the 'ax' style
    ax.set_title("Scale–Location Plot", fontsize=13)
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("$\sqrt{|Standardized Residuals|}$")

    # 5. Styling
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')
    
    # Lgnd
    # ---- 
    ax.legend()     
         
    # Titl
    # ----
    titl = plot_tran_mixd.titl # "VEINES-QOL over time"
    ax.set_title(titl)
    # fig.tight_layout(): See 'FiguTran' initiaization : self.fig = plt.figure(layout="constrained")
    # fig.show()
    pass
    
def print_yes(df, labl=None):
    print (f"\n----\nFram labl : {labl}\n----")
    with pd.option_context(
            'display.max_columns', None,       # Show all columns
            # 'display.max_rows', None,          # Show more rows before truncating
            'display.max_colwidth', None,      # Don't cut off long text in 'info'
            'display.width', 1000,             # Prevent the table from wrapping to a new line
            'display.precision', 2,            # Round floats to 2 decimal places
            'display.colheader_justify', 'left' # Align headers for better readability
        ):
        if isinstance(df, pd.DataFrame):
            print(f"df:{len(df)} type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
            print(df.info())
        elif isinstance(df, pd.Series):
            print(f"df:{len(df)} type:{type(df)}\n{df}\n:{df.index}")
            print(df.info())
    pass