    
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

# ----
# Plot https://copilot.microsoft.com/shares/7rZLRYu1wYqmA3sQABBX4
# ----

def plot_assu_resi_ququ(plot_tran_mixd: PlotTranQOL_32_assu_resi) -> None:
    # from qol_30.c02_qol_32_grph_plot_ import PlotTranQOL_32_assu_resi

    trac = True

    # Data
    # ---- 
    df_assu_resi_plot = plot_tran_mixd.fram_dict["orig_fram"]
    print_yes(df_assu_resi_plot)
    '''
    '''
    df_plot = df_assu_resi_plot.copy()
    if trac:
        print_yes(df_plot, labl="df_plot")
  
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
    # --- RIGHT PLOT: Q–Q Plot ---
    data = df_plot['Residuals']
    # stats.probplot allows you to pass a specific axis object via 'plot='
    stats.probplot(data, dist="norm", plot=ax)

    # Customizing ax2 (the Q-Q plot)
    ax.get_lines()[0].set_markerfacecolor('skyblue') # Points
    ax.get_lines()[0].set_markeredgecolor('navy')
    ax.get_lines()[1].set_color('red')             # Reference line

    # Formatting ax2
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.grid(True, alpha=0.3)
    
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

def plot_assu_resi_hist(plot_tran_mixd: PlotTranQOL_32_assu_resi) -> None:
    # from qol_30.c02_qol_32_grph_plot_ import PlotTranQOL_32_assu_resi

    trac = True

    # Data
    # ---- 
    df_assu_resi_plot = plot_tran_mixd.fram_dict["orig_fram"]
    print_yes(df_assu_resi_plot)

    '''
    patient_id       Group
    PT_2024_02_00078 -3.72
    PT_2024_02_08277 -1.81
    PT_2024_02_10578  0.49
    PT_2024_02_11301  2.52
    '''
    df_plot = df_assu_resi_plot.copy()
    if trac:
        print_yes(df_plot, labl="df_plot")
  
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
    # 2. Get stats from your data
    data = df_plot['Residuals']
    mu, std = data.mean(), data.std()

    # 3. Plot the Histogram (Note: density=True is required for the curve to align)
    ax.hist(data, bins=10, density=True, alpha=0.6, color='skyblue', edgecolor='white', label='Actual Data')

    # 4. Create the Normal Curve (PDF)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)

    # 5. Overlay the curve
    ax.plot(x, p, 'r', linewidth=2, label=f'Normal Dist. ($\mu={mu:.2f}, \sigma={std:.2f}$)')

    # 6. Formatting
    # ax.set_title('Random Effects with Normal Curve Overlay')
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability Density')
    
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