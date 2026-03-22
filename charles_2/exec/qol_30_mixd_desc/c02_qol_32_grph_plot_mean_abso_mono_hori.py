    
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_qol_32_grph_plot_ import PlotTranQOL_32_mean_abso_mono_hori
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# https://chatgpt.com/c/69667f25-5780-8330-88c0-fdb7649b4a96
'''
import matplotlib.pyplot as plt
import seaborn as sns
    def plot_qol_spaghetti(df):
        plt.figure(figsize=(10, 6))

        # 1. Plot individual patient lines (The "Spaghetti")
        # We use a low alpha (transparency) to see the density
        sns.lineplot(
            data=df, 
            x='timepoint', 
            y='VEINES_QOL_t', 
            units='patient_id', 
            estimator=None, 
            color='gray', 
            alpha=0.3, 
            linewidth=1
        )

        # 2. Plot the Group Mean (The "Signal")
        # This shows the actual trend your Mixed Model captured
        sns.lineplot(
            data=df, 
            x='timepoint', 
            y='VEINES_QOL_t', 
            color='red', 
            linewidth=3, 
            marker='o', 
            label='Group Mean'
        )

        plt.title('Patient-Level Evolution of VEINES QOL t-scores', fontsize=14)
        plt.ylabel('VEINES QOL t-score', fontsize=12)
        plt.xlabel('Timepoint', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.legend()
        
        plt.savefig('qol_spaghetti_plot.png')
        plt.show()
'''

def plot_mean_abso_mono_hori(plot_tran_mixd: PlotTranQOL_32_mean_abso_mono_hori) -> None:
    # from qol_30.c02_qol_32_grph_plot_ import import PlotTranQOL_32_mean_abso_mono_hori

    trac = True

    # Data
    # ---- 
    '''
    ----
    Fram labl : resu_fram
    ----
        workbook                                                                 patient_id        timepoint  iter_t  pati_isok  Age  BMI  CEAP_P
    0               2025-11-01 2025-11-01 T0 PT_2024_02_00078 MAKOLA ODETTE V03  PT_2024_02_00078  T0         46.8   True       59   34.7  C4
    1      2025-11-01 2025-11-01 T0 PT_2024_02_08277 KAVIRA ZAWADI CLAUDINE V03  PT_2024_02_08277  T0        147.2   True       45   38.5  C3
    ----
    Fram labl : resu_plot
    ----
        timepoint  n   mean   ci_lower  ci_upper  sd     se    modl_mean  modl_ci_lower  modl_ci_upper
    0  T0        30  53.34  46.87     59.80     18.06  3.30  53.34      43.43          63.25
    1  T1        30  50.00  48.75     51.26      3.50  0.64  50.00      40.09          59.91
    2  T2        30  63.34  47.48     79.19     44.30  8.09  63.34      53.43          73.25
    '''
    #df_orig = plot_tran_mixd.fram_dict["open_resu_fram"] # workbook, date1, date2, timepoint, patient_id, name, none_m, none_z, none_t, ...
    #df_fram = plot_tran_mixd.fram_dict["open_resu_plot"] # "timepoint", "n", "mean", "ci_lower", "ci_upper", "modl_mean", "modl_ci_lower", "modl_ci_upper", ...
    df_orig = plot_tran_mixd.fram_dict["orig_fram"] # workbook, date1, date2, timepoint, patient_id, name, none_m, none_z, none_t, ...
    df_fram = plot_tran_mixd.fram_dict["mean_fram"] # "timepoint", "n", "mean", "ci_lower", "ci_upper", "modl_mean", "modl_ci_lower", "modl_ci_upper", ...
    df_orig = df_orig.copy()
    df_fram = df_fram.copy()
    if trac:
        print_yes(df_orig, labl="df_orig")
        print_yes(df_fram, labl="df_fram")

    # Mock
    # ----
    exec_mock = False
    if exec_mock:
        
        # df_orig
        # -------
        # Mock raw scores for 5 patients
        data = []
        np.random.seed(42)
        time_list = ["T0", "T1", "T2"]
        for pid in range(1, 6):
            mean_base = np.random.normal(50, 5)
            t1 = mean_base + np.random.normal(2, 3)
            t2 = mean_base + np.random.normal(5, 5)
            data.extend([
                {"patient_id": pid, "timepoint": tp, "raw_score": val}
                for tp, val in zip(time_list, [mean_base, t1, t2])
            ])
        df_spag = pd.DataFrame(data)
        df_spag["timepoint"] = pd.Categorical(df_spag["timepoint"], categories=time_list, ordered=True)
        df_orig = df_spag  
        
        # df_fram
        # -------
        df_mock = pd.DataFrame({
            # Timepoints and counts
            "timepoint": pd.Categorical(["T0","T1","T2"], ordered=True),
            "n": [30, 30, 30],
            # Observed values
            "mean": [53.34, 50.00, 63.34],
            "sd": [18.06, 3.50, 44.30],
            "se": [3.30, 0.64, 8.09],
            "ci_lower": [46.87, 48.75, 47.48],
            "ci_upper": [59.80, 51.26, 79.19]
        })
        df_fram = df_mock
    
    # Exec
    # ----
    cols_requ = {"timepoint", "n", "mean", "ci_lower", "ci_upper"}
    if not cols_requ.issubset(df_fram.columns):
        raise ValueError(f"Missing columns: {cols_requ - set(df_fram.columns)}")
    #
    time_list = df_fram["timepoint"].cat.categories # T0, T1, ... categories https://gemini.google.com/app/f11fdb0ca1fc6a70
    x = np.arange(len(time_list))
    df_fram = df_fram.set_index("timepoint", drop=False) # -> column is not dropped
    mean_base = df_fram.loc[time_list[0], "mean"] # note : requires 'timepint' to be df_fram 'index'
    #
    if trac:
        print_yes(df_fram, labl="df_fram")    
      
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

    # Stra
    # ----
    # plot_tran_mixd.stra = 'a'
    match (plot_tran_mixd.stra):
        
        # ====
        case "a": # Veines QOL Score at T0,T1,T2 : Mean ± 95% CI [at timepoints]
        # ====

            # Plot
            # ----
            def _comp_yerr(df):   
                return np.vstack([
                    df["mean"] - df["ci_lower"],
                    df["ci_upper"] - df["mean"],
                ])
            yerr = _comp_yerr(df_fram)
            ax.errorbar(
                df_fram["timepoint"], 
                df_fram["mean"], 
                yerr=yerr, 
                fmt=line_styl, 
                color=line_colo, 
                alpha=line_alph, 
                linewidth=line_widt, 
                markersize=mark_size, 
                markerfacecolor=mark_colo, 
                markeredgewidth=mark_widt, 
                capsize=capp_size, 
                ecolor=erro_colo, 
                label=line_labl)
            #
            ax.fill_between(
                x,
                df_fram["ci_lower"],
                df_fram["ci_upper"],
                color=erro_colo,
                alpha=0.25,
                linewidth=0,
                zorder=1,
            )
                     
            # Axis, Grid
            # ---- 
            def _annotate_n(ax, x, df, y_range, dy=0.04):
                """Annotate sample sizes below CI."""
                offset = dy * y_range
                for xi, (_, row) in zip(x, df.iterrows()):
                    ax.text(
                        x=xi,
                        y=row["ci_lower"] - offset,
                        s=f"n = {int(row['n'])}",
                        ha="center",
                        va="top",
                        fontsize=9,
                        color="0.35",
                    )   
            ymin = df_fram["ci_lower"].min() - 2
            ymax = df_fram["ci_upper"].max() + 2
            yrng = ymax - ymin
            ax.set_ylim(ymin - 0.15 * yrng, ymax + 0.15 * yrng)
            _annotate_n(ax, x, df_fram, yrng)
            #plt.show()
            pass
         
        # ====
        case "b":  # Veines QOL Score at T0,T1,T2 : Mean ± 95% CI [between timepoints]
        # ====
        
            # Plot
            # ----
            ax.fill_between(
                x,
                df_fram["ci_lower"],
                df_fram["ci_upper"],
                color=erro_colo,
                alpha=0.25,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                x,
                df_fram["mean"],
                color=line_colo,
                linewidth=line_widt,
                marker="o",
                markersize=mark_size,
                markerfacecolor=mark_colo,
                label=line_labl,
                zorder=3,
            )
                     
            # Axis, Grid
            # ---- 
            def _annotate_n(ax, x, df, y_range, dy=0.04):
                """Annotate sample sizes below CI."""
                offset = dy * y_range
                for xi, (_, row) in zip(x, df.iterrows()):
                    ax.text(
                        x=xi,
                        y=row["ci_lower"] - offset,
                        s=f"n = {int(row['n'])}",
                        ha="center",
                        va="top",
                        fontsize=9,
                        color="0.35",
                    )   
            ymin = df_fram["ci_lower"].min() - 2
            ymax = df_fram["ci_upper"].max() + 2
            yrng = ymax - ymin
            ax.set_ylim(ymin - 0.15 * yrng, ymax + 0.15 * yrng)
            _annotate_n(ax, x, df_fram, yrng)  
            #plt.show()
            pass           

        # ====
        case "c":  # Veines QOL Score change from T0 to T1,T2
        # ======
            
            # Comp Δ and Error Distances
            # ----
            df_delt = df_fram["mean"] - mean_base
            if trac:
                print (mean_base)
                print (df_fram["mean"] )
                print_yes(df_delt, labl="df_delt")
            # Distances must be positive for ax.errorbar
            # (mean - lower) and (upper - mean)
            low_dist = (df_fram["mean"] - df_fram["ci_lower"]).values
            upp_dist = (df_fram["ci_upper"] - df_fram["mean"]).values
            yerr = np.vstack([low_dist, upp_dist])

            # Plot Δ from mean_base
            # ----
            ax.errorbar(
                x,
                df_delt,
                yerr=yerr,
                fmt=line_styl,
                color=line_colo,
                alpha=line_alph,
                linewidth=line_widt,
                markersize=mark_size,
                markerfacecolor=mark_colo,
                markeredgewidth=mark_widt,
                capsize=capp_size,
                ecolor=erro_colo,
                label=line_labl,
                zorder=3,
            )
            #
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
                       
            # Axis, Grid
            # ---- 
            def _annotate_n_delt(ax, x, df, mean_base, y_range, dy=0.06):
                """Annotate sample sizes below the Delta CI bars."""
                offset = dy * y_range
                for xi, (_, row) in zip(x, df.iterrows()):
                    # The bottom of the error bar in delta-space
                    y_bottom = (row["ci_lower"] - mean_base)                   
                    ax.text(
                        x=xi,
                        y=y_bottom - offset,
                        s=f"n = {int(row['n'])}",
                        ha="center",
                        va="top",
                        fontsize=9,
                        color="0.4",  # Subtle grey
                        fontstyle="normal"
                    )

            # ---- 1. Calculate the Delta Bounds for the Y-axis
            # We use the actual lower/upper limits in Delta space
            delt_low_limit = df_fram["ci_lower"] - mean_base
            delt_upp_limit = df_fram["ci_upper"] - mean_base

            ymin_delt = delt_low_limit.min()
            ymax_delt = delt_upp_limit.max()
            yrng_delt = ymax_delt - ymin_delt
            # ---- 2. Set limits with room for the annotations at the bottom
            ax.set_ylim(ymin_delt - 0.25 * yrng_delt, ymax_delt + 0.15 * yrng_delt)
            # ---- 3. Call the annotation
            _annotate_n_delt(ax, x, df_fram, mean_base, yrng_delt)
            #plt.show()
            pass 
                    
        # ====
        case "d":  # Veines QOL Score change from T0 at T1,T2 : Individual and Mean scores [between timepoints]
        # ======
            print (df_orig)
            # Individual patient trajectories
            for pid, g in df_orig.groupby("patient_id", sort=False):
                xi = g["timepoint"].cat.codes
                ax.plot(
                    xi,
                    g["VEINES_QOL_t"],
                    color="blue",
                    alpha=0.15,
                    linewidth=1.0,
                    zorder=1
                )

            # Overlay mean trajectory
            ax.plot(
                x,
                df_fram["mean"],
                color=line_colo,
                linewidth=line_widt,
                marker="o",
                markersize=mark_size,
                markerfacecolor=mark_colo,
                label=line_labl,
                zorder=3,
            )
            #plt.show()
            pass       

        # ====
        case "e":  
        # Observed means ± CI (secondary layer)
        # Mixed model estimates ± CI (primary layer)
        # ====
        
            # Observed values
            # ----
            yerr=[
                df_fram["mean"] - df_fram["ci_lower"],
                df_fram["ci_upper"] - df_fram["mean"]
            ]
            if False:
                ax.errorbar(
                    df_fram["timepoint"], 
                    df_fram["mean"], 
                    yerr=yerr, 
                    fmt=line_styl, 
                    color=line_colo, 
                    alpha=line_alph, 
                    linewidth=line_widt, 
                    markersize=mark_size, 
                    markerfacecolor=mark_colo, 
                    markeredgewidth=mark_widt, 
                    capsize=capp_size, 
                    ecolor=erro_colo, 
                    label=line_labl)
            
            ax.errorbar(
                x,
                df_fram["mean"],
                yerr=[df_fram["mean"] - df_fram["ci_lower"], df_fram["ci_upper"] - df_fram["mean"]],
                fmt='o-',
                color='blue',
                markersize=8,
                markerfacecolor='blue',
                markeredgewidth=1.2,
                capsize=5,
                label="Observed mean",
                zorder=2
            )
            #
            # Mixed model estimates ± CI (primary layer)
            # ----
            yerr_model = [
                df_fram["modl_estimate"] - df_fram["modl_ci_lower"],
                df_fram["modl_ci_upper"] - df_fram["modl_estimate"]
            ]
            if False:
                ax.errorbar(
                    x,
                    df_fram["model_estimate"],
                    yerr=yerr_model,
                    fmt=line_styl,
                    color="red",
                    linewidth=line_widt,
                    markersize=mark_size,
                    markerfacecolor="white",
                    markeredgewidth=1.5,
                    capsize=capp_size,
                    ecolor=erro_colo,
                    label="Mixed-effects model estimate",
                    zorder=3
                )
            ax.errorbar(
                x,
                df_fram["model_estimate"],
                yerr=[df_fram["model_estimate"] - df_fram["modl_ci_lower"],
                    df_fram["modl_ci_upper"] - df_fram["model_estimate"]],
                fmt='s--',
                color='black',
                markersize=8,
                markerfacecolor='white',
                markeredgewidth=1.5,
                capsize=5,
                label="Mixed model estimate",
                zorder=3
            )
            #
            def _annotate_n(ax, x, df, y_range, dy=0.04):
                """Annotate sample sizes below CI."""
                offset = dy * y_range
                for xi, (_, row) in zip(x, df.iterrows()):
                    ax.text(
                        x=xi,
                        y=row["ci_lower"] - offset,
                        s=f"n = {int(row['n'])}",
                        ha="center",
                        va="top",
                        fontsize=9,
                        color="0.35",
                    )   
            ymin = min(df_fram["ci_lower"].min(), df_fram["modl_ci_lower"].min()) - 2
            ymax = max(df_fram["ci_upper"].max(), df_fram["modl_ci_upper"].max()) + 2
            yrng = ymax - ymin
            ax.set_ylim(ymin - 0.15 * yrng, ymax + 0.15 * yrng)
            _annotate_n(ax, x, df_fram, yrng)
            pass

        case _:
            raise Exception()
            
    # Axis, Grid
    # ----
    xlab = plot_tran_mixd.xlab # "Timepoint"
    ax.set_xlabel(xlab)
    ax.margins(x=0.10)  # x-axis padding
    #
    tick_dict = plot_tran_mixd.tick_dict # {"T0": "Baseline", "T1": "3 months", "T2": "12 months"}
    labl_list = [tick_dict.get(cat, cat) for cat in time_list]
    ax.set_xticks(x)               # numeric positions
    ax.set_xticklabels(labl_list)  # readable labels

    ylab = plot_tran_mixd.ylab # "VEINES-QOL score"
    ax.set_ylabel(ylab)
    
    grid_alph = plot_tran_mixd.grid_alph
    if grid_alph is not None:
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=grid_alph) # ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)
        ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.3, alpha=0.5)
    
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