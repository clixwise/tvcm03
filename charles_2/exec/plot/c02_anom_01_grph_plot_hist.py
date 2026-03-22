from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_anom_01_grph_plot_ import PlotTranANOM_01_hist
    from c02_anom_01_grph_plot_ import PlotConfANOM_01_hist
    
import matplotlib.pyplot as plt

# Third-Party Imports
import numpy as np
import pandas as pd
import seaborn as sns

# SciPy Imports
from scipy import stats
from scipy.interpolate import CubicSpline
from scipy.stats import (
    gaussian_kde,
    norm,
    skew,
    median_abs_deviation,
    mode,
    kurtosis
)

'''
Default Z-Values: Matplotlib has default zorder values for different types of elements. Knowing these defaults helps you decide what values to choose:
Images: 0
Patches (e.g., bar charts, filled areas): 1
Lines (ax.plot(), ax.axvline(), grid lines): 2
Text (e.g., labels, titles): 3
Legend: 5
'''  
# ----
# Timepoint outcomes
# ----
def exec_plot_hist(plot_tran_pure: PlotTranANOM_01_hist) -> None:
    # from c02_qol_01_grph_plot_ import PlotTranANOM_01_hist
    
    trac = True
    plot_tran_conf:PlotConfANOM_01_hist = plot_tran_pure.conf

    # Data
    # ---- 
    df_fram = plot_tran_conf.fram # workbook, date1, date2, timepoint, patient_id, name, none_m, none_z, none_t, ...
    if trac:
        print_yes(df_fram, labl="df_fram")
    #
    colu = plot_tran_conf.colu
    data_seri = plot_tran_conf.fram[colu]
    # data_seri = pd.to_numeric(data_seri, errors='coerce').astype('Int64') # optional
    data_seri = np.sort(data_seri) # required for CDF
     
    # ----
    # Grph
    # ---- 
    plt.style.use('default') # fig, ax = plt.subplots(figsize=(11,8)) # width, height

    # Vars
    # ----
    fig = plot_tran_conf.figu
    ax = plot_tran_conf.axis
    ax.clear()
    binn_size = plot_tran_conf.binn_size
    binn_list = np.arange(min(data_seri) // binn_size * binn_size, max(data_seri) // binn_size * binn_size + binn_size, binn_size)
    
    # Plot histogram
    # --------------
    hist_widt = plot_tran_conf.hist_widt
    hist_styl = plot_tran_conf.hist_styl
    hist_colo = plot_tran_conf.hist_colo
    hist_alph = plot_tran_conf.hist_alph
    hist_plot = plot_tran_conf.stra_hist
    if hist_plot:
        sns.histplot(data_seri, bins=binn_list, kde=False, color=hist_colo, edgecolor="darkgray", alpha=0.2, ax=ax, linewidth=hist_widt, linestyle=hist_styl)

    # Kernel Density Estimation (smoothed distribution)
    # -------------------------------------------------
    kdee_plot = plot_tran_conf.stra_kdee
    if kdee_plot:
        def kdee_plot_exec(data_seri):
            kde = gaussian_kde(data_seri)
            x_vals = np.linspace(data_seri.min(), data_seri.max(), 1000)
            kde_vals = kde(x_vals)
            ax.plot(x_vals, kde_vals * len(data_seri) * (data_seri.max() - data_seri.min()) / len(binn_list), label="Smoothed distribution_T", color="indigo", linestyle="--")
        kdee_plot_exec(data_seri)

    # Normal distribution
    # -------------------
    norm_widt = plot_tran_conf.norm_widt
    norm_styl = plot_tran_conf.norm_styl
    norm_colo = plot_tran_conf.norm_colo
    norm_plot = plot_tran_conf.stra_norm 
    if norm_plot:
        def norm_plot_exec(data_seri):
            mu, sigma = norm.fit(data_seri)
            x_vals = np.linspace(data_seri.min(), data_seri.max(), 1000)
            y_vals = norm.pdf(x_vals, mu, sigma) * len(data_seri) * (data_seri.max() - data_seri.min()) / len(binn_list)
            ax.plot(x_vals, y_vals, label="Normal distribution", color=norm_colo, linestyle=norm_styl, linewidth=norm_widt)
        norm_plot_exec(data_seri)
    
    # Cumulative Distribution Function (CDF)
    # --------------------------------------
    cdff_widt = plot_tran_conf.cdff_widt
    cdff_styl = plot_tran_conf.cdff_styl
    cdff_colo = plot_tran_conf.cdff_colo
    cdff_plot = plot_tran_conf.stra_cdff
    if cdff_plot:
        ax2 = ax.twinx()
        def cdff_plot_exec(data_seri, ax2):
            
            # ax2.set_ylabel('Cumulative Distribution (normalized)')
            ax2.set_ylim(0, 1)
        
            # 1. The KDE function is a smoothed version of the PDF (Probability Density Function)
            kernel = gaussian_kde(data_seri)
            # 2. Define a smooth range of x-values for plotting. Use a range slightly wider than the data to show the tails cleanly
            x_range = np.linspace(data_seri.min() - 2, data_seri.max() + 2, 200)
            # 3. Calculate the Smooth CDF
            # The smooth CDF is the cumulative integral of the KDE (PDF). Use a numerical integration method (like cumsum of the KDE values times the step size)
            smooth_pdf = kernel.evaluate(x_range)
            # Calculate the step size
            dx = x_range[1] - x_range[0] 
            # Calculate the cumulative sum and normalize to ensure the max is 1
            smooth_cdf = np.cumsum(smooth_pdf) * dx
            # Normalize exactly to 1.0 at the end, accounting for integration error
            smooth_cdf /= smooth_cdf[-1] 
            # Original (Step-Function) CDF Plot
            cdf_vals_step = np.arange(1, len(data_seri) + 1) / len(data_seri)
            ax2.plot(data_seri, cdf_vals_step, color='gray', linestyle=':', label='Step CDF (ECDF)', alpha=0.6)
            # Smoothed CDF Plot
            ax2.plot(x_range, smooth_cdf, color=cdff_colo, linestyle=cdff_styl, linewidth=cdff_widt, label='Smooth CDF (KDE-based)')
        cdff_plot_exec(data_seri, ax2)
        
    # Quartiles and Outliers
    # ----------------------
    quar_plot = plot_tran_conf.stra_quar
    if quar_plot:
        def quar_plot_exec(ages_):
            q1_, q3_ = np.percentile(ages_, [25, 75])
            below_q1_ = len(ages_[ages_ < q1_])
            above_q3_ = len(ages_[ages_ > q3_])
            within_iqr_ = len(ages_) - below_q1_ - above_q3_
            pati_belo_ = below_q1_ / len(ages_) * 100
            pati_with_ = within_iqr_ / len(ages_) * 100
            pati_abov_ = above_q3_ / len(ages_) * 100
            #
            ax.axvline(q1_, color="indigo", linestyle="--", label=f"Q1: {q1_:.0f} y.", linewidth=1)
            ax.axvline(q3_, color="indigo", linestyle="--", label=f"Q3: {q3_:.0f} y.", linewidth=1)
            return q1_, q3_,below_q1_, above_q3_, within_iqr_, pati_belo_, pati_with_, pati_abov_
        q1_T, q3_T,below_q1_T, above_q3_T, within_iqr_T, pati_belo_T, pati_with_T, pati_abov_T = quar_plot_exec(data_seri)
    
    # Median
    # ------
    medi_plot = plot_tran_conf.stra_medi
    if medi_plot:
        
        def medi_plot_exec(data_seri):
            median_age_T = np.median(data_seri)
            ax.axvline(median_age_T, color="purple", linestyle="--", label=f"Median: {round(median_age_T)} y.", linewidth=1)
            mark = False
            if mark:
                # Draw the vertical line for median age
                ax.axvline(median_age_T, color="purple", linestyle="-", label=f"Median: {round(median_age_T)} y.", linewidth=1)
                # Add a point marker at the median age
                ax.plot(median_age_T, 0.5, marker='.', color="purple", markersize=10, transform=ax.get_xaxis_transform())
        medi_plot_exec(data_seri)
        
    # MAD range
    # ---------
    madd_plot = plot_tran_conf.stra_madd
    if madd_plot:
        mad_T = stats.median_abs_deviation(data_seri)
        def madd_plot_exec(data_seri):
            median_age_T = np.median(data_seri)
            mad_T = stats.median_abs_deviation(data_seri)
            mad_T_L = median_age_T - mad_T
            mad_T_R = median_age_T + mad_T
            mad_disp = True
            if mad_disp:
                ax.axvspan(mad_T_L, mad_T_R, alpha=0.2, color='gray', label=f'±1 MAD: {mad_T:.1f}', linewidth=1)
                # ax.text(0.05, 0.95, f'MAD: {mad_T:.1f} years', transform=ax.transAxes, verticalalignment='top', fontsize=9, color=colo_indi)
        madd_plot_exec(data_seri)

    # Mean
    # ----
    mean_plot = plot_tran_conf.stra_mean
    if mean_plot:
        def mean_plot_exec(data_seri):
            mean_age_T = data_seri.mean()
            ax.axvline(mean_age_T, color="indigo", linestyle="-.", label=f"Mean: {round(mean_age_T)} y.", linewidth=1)
        mean_plot_exec(data_seri)
    
    # Mode
    # ----
    mode_plot = plot_tran_conf.stra_mode
    if mode_plot:
        def mode_plot_exec(data_seri):
            mode_result_T = stats.mode(data_seri) ; mode_age_T = mode_result_T.mode
            if np.isscalar(mode_age_T): 
                mode_age_T = float(mode_age_T) 
            else: 
                mode_age_T = mode_age_T[0]
            ax.axvline(mode_age_T, color="green", linestyle="--", label=f"Mode: {mode_age_T:.1f} years", linewidth=1)
        mode_plot_exec(data_seri)
    
    # Skeness
    # -------
    skew_T = skew(data_seri)

    # Gray backgrounds for Q1 and Q3 regions
    # --------------------------------------
    bool_quar = False
    if bool_quar:
        ax.axvspan(data_seri.min(), q1_T, color="gray", alpha=0.1, label="Below Q1")
        ax.axvspan(q3_T, data_seri.max(), color="gray", alpha=0.1, label="Above Q3")

    # Labels
    # ------    
    xlab = plot_tran_conf.xlab # parm_dict['plot'][what].get('xlab', None)
    if xlab is not None:
        ax.set_xlabel(xlab, fontsize=12)
    ylab = plot_tran_conf.yla1 # parm_dict['plot'][what].get('yla1', None)
    if ylab is not None:
        ax.set_ylabel(ylab, fontsize=12)
    ylab = plot_tran_conf.yla2
    if ylab is not None:
        ax2.set_ylabel(ylab, fontsize=12)
    '''
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # --- Reset/Set on an existing Axes object ---
    ax.set_title('New Title', fontsize=20)
    ax.set_xlabel('New X-Label', fontsize=18)
    ax.set_ylabel('New Y-Label', fontsize=18)
    # You must set the tick label size on the tick objects themselves
    ax.tick_params(axis='both', which='major', labelsize=15)

    ax.plot([0, 1], [0, 1])
    plt.show()
    '''
    # Titles
    # ------
    titl = plot_tran_conf.titl
    ax.set_title(titl)
    fig.tight_layout()
    # fig.show()

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
        print(f"df:{len(df)} type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
        print(df.info())
    pass