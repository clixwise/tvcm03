    
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_vcss_11_grph_plot_ import PlotTranVCSS_11_mixd_mono
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# https://chatgpt.com/c/69667f25-5780-8330-88c0-fdb7649b4a96
# https://chatgpt.com/c/698849d4-d684-832d-a367-fd26b2f1ac4f # <- best

def exec_plot_mixd_mono(plot_tran_mixd: PlotTranVCSS_11_mixd_mono) -> None:
    # from vcss_10.c02_vcss_11_grph_plot_ import PlotTranVCSS_11_mixd_mono

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
    df_orig = plot_tran_mixd.fram_dict["resu_fram"] # workbook, date1, date2, timepoint, patient_id, name, none_m, none_z, none_t, ...
    df_fram = plot_tran_mixd.fram_dict["resu_plot"] # "timepoint", "n", "mean", "ci_lower", "ci_upper", "modl_mean", "modl_ci_lower", "modl_ci_upper", ...
    df_orig = df_orig.copy()
    df_fram = df_fram.copy()
    if trac:
        print_yes(df_orig, labl="df_orig")
        print_yes(df_fram, labl="df_fram")
    
    # Exec : df_orig
    # ----
    df_orig_R = df_orig[df_orig["Limb"] == "R"]
    df_orig_L = df_orig[df_orig["Limb"] == "L"]

    # Exec : df_fram
    # ----
    cols_requ = {"timepoint", "n", "mean", "ci_lower", "ci_upper"}
    if not cols_requ.issubset(df_fram.columns):
        raise ValueError(f"Missing columns: {cols_requ - set(df_fram.columns)}")
    #
    time_list = df_fram["timepoint"].cat.categories # T0, T1, ... categories https://gemini.google.com/app/f11fdb0ca1fc6a70
    x = np.arange(len(time_list))
    #
    df_R = df_fram[df_fram['Limb'] == 'R'].drop(columns='Limb')
    df_R = df_R.set_index("timepoint", drop=False) # -> column is not dropped
    mean_base_R = df_R.loc[time_list[0], "mean"] # note : requires 'timepoint' to be df_fram 'index'
    df_L = df_fram[df_fram['Limb'] == 'L'].drop(columns='Limb')
    df_L = df_L.set_index("timepoint", drop=False) # -> column is not dropped
    mean_base_L = df_L.loc[time_list[0], "mean"] # note : requires 'timepoint' to be df_fram 'index'
    if trac:
        print_yes(df_R, labl="df_R")
        print_yes(df_L, labl="df_L")
      
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
        case "a": # VCSS Score at T0,T1,T2 : Mean ± 95% CI [at timepoints]
        # ====

            def case_a(df_fram, ax, x_shift=0.05, line_styl='-o', line_colo='blue', line_alph=0.7,
                    line_widt=1.5, mark_size=8, mark_colo='white', mark_widt=1,
                    erro_colo='blue', capp_size=5, line_labl=None):
                # Définir les positions x avec décalage
                x = np.arange(len(df_fram)) + x_shift

                # Calcul des erreurs pour errorbar
                def _comp_yerr(df):
                    return np.vstack([
                        df["mean"] - df["ci_lower"],
                        df["ci_upper"] - df["mean"],
                    ])
                yerr = _comp_yerr(df_fram)

                # Tracer les errorbars
                ax.errorbar(
                    x,
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
                    label=line_labl,
                )

                # Remplir l'intervalle de confiance
                ax.fill_between(
                    x,
                    df_fram["ci_lower"],
                    df_fram["ci_upper"],
                    color=erro_colo,
                    alpha=0.25,
                    linewidth=0,
                    zorder=1,
                )

                # Annoter les tailles d'échantillon
                def _annotate_n(ax, x, df, y_range, dy=0.04):
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

                # Ajustement des limites de l'axe y
                ymin = df_fram["ci_lower"].min() - 2
                ymax = df_fram["ci_upper"].max() + 2
                yrng = ymax - ymin
                ax.set_ylim(ymin - 0.15 * yrng, ymax + 0.15 * yrng)
                _annotate_n(ax, x, df_fram, yrng)
                
            case_a(df_R, ax, x_shift=-0.1, line_colo='blue', line_labl='Right Limb (R)')
            case_a(df_L, ax, x_shift=0.1, line_colo='red', line_labl='Left Limb (L)')
            #plt.show()
            pass
         
        # ====
        case "b":  # VCSS Score at T0,T1,T2 : Mean ± 95% CI [between timepoints]
        # ====
            
            def case_b(df_fram, ax, x_shift=0.05, line_styl='-o', line_colo='blue', line_alph=0.7,
                    line_widt=1.5, mark_size=8, mark_colo='white', mark_widt=1,
                    erro_colo='blue', capp_size=5, line_labl=None):
                # Définir les positions x avec décalage
                x = np.arange(len(df_fram)) + x_shift

                # Calcul des erreurs pour errorbar
                def _comp_yerr(df):
                    return np.vstack([
                        df["mean"] - df["ci_lower"],
                        df["ci_upper"] - df["mean"],
                    ])
                yerr = _comp_yerr(df_fram)

                # Tracer les errorbars
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

                # Remplir l'intervalle de confiance
                ax.fill_between(
                    x,
                    df_fram["ci_lower"],
                    df_fram["ci_upper"],
                    color=erro_colo,
                    alpha=0.25,
                    linewidth=0,
                    zorder=1,
                )

                # Annoter les tailles d'échantillon
                def _annotate_n(ax, x, df, y_range, dy=0.04):
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

                # Ajustement des limites de l'axe y
                ymin = df_fram["ci_lower"].min() - 2
                ymax = df_fram["ci_upper"].max() + 2
                yrng = ymax - ymin
                ax.set_ylim(ymin - 0.15 * yrng, ymax + 0.15 * yrng)
                _annotate_n(ax, x, df_fram, yrng)
                
            case_b(df_R, ax, x_shift=-0.1, line_colo='blue', line_labl='Right Limb (R)')
            case_b(df_L, ax, x_shift=0.1, line_colo='red', line_labl='Left Limb (L)')
            #plt.show()
            pass       

        # ====
        case "c":  # VCSS Score change from T0 to T1,T2
        # ======
 
            def case_c(df_fram, mean_base, ax, x_shift=0.05,
                line_styl='-o', line_colo='blue', line_alph=0.7,
                line_widt=1.5, mark_size=8, mark_colo='white', mark_widt=1,
                erro_colo='blue', capp_size=5, line_labl=None):

                # X positions
                x = np.arange(len(df_fram)) + x_shift

                # ----
                # Convert to DELTA space
                # ----
                mean_d = df_fram["mean"] - mean_base
                low_d  = df_fram["ci_lower"] - mean_base
                upp_d  = df_fram["ci_upper"] - mean_base

                # ----
                # Errors (unchanged mathematically)
                # ----
                yerr = np.vstack([
                    mean_d - low_d,
                    upp_d - mean_d,
                ])

                # ----
                # Plot line
                # ----
                ax.plot(
                    x,
                    mean_d,
                    color=line_colo,
                    linewidth=line_widt,
                    marker="o",
                    markersize=mark_size,
                    markerfacecolor=mark_colo,
                    label=line_labl,
                    zorder=3,
                )

                # ----
                # Fill CI
                # ----
                ax.fill_between(
                    x,
                    low_d,
                    upp_d,
                    color=erro_colo,
                    alpha=0.25,
                    linewidth=0,
                    zorder=1,
                )

                # ----
                # Annotate n
                # ----
                def _annotate_n(ax, x, low_vals, df, y_range, dy=0.04):
                    offset = dy * y_range
                    for xi, lv, (_, row) in zip(x, low_vals, df.iterrows()):
                        ax.text(
                            xi,
                            lv - offset,
                            f"n = {int(row['n'])}",
                            ha="center",
                            va="top",
                            fontsize=9,
                            color="0.35",
                        )

                # ----
                # Y limits
                # ----
                ymin = low_d.min()
                ymax = upp_d.max()
                yrng = ymax - ymin
                ax.set_ylim(ymin - 0.25 * yrng, ymax + 0.15 * yrng)

                _annotate_n(ax, x, low_d, df_fram, yrng)

                # Baseline reference
                ax.axhline(0, color="black", linestyle="--", linewidth=1)

            case_c(df_R, mean_base_R, ax, x_shift=-0.1, line_colo='blue', line_labl='Right Limb (R)')
            case_c(df_L, mean_base_L, ax, x_shift=0.1, line_colo='red', line_labl='Left Limb (L)')
            # plt.show()
            pass
                    
        # ====
        case "d":  # Veines VCSS Score change from T0 at T1,T2 : Individual and Mean scores [between timepoints]
        # ======
            
            def case_d(df_indiv, df_mean, ax, x_shift=0.0,
                            line_colo='blue', line_widt=2,
                            mark_size=8, mark_colo='white',
                            line_labl=None):

                # x positions for mean
                x = np.arange(len(df_mean)) + x_shift

                # ----
                # Individual patient trajectories
                # ----
                for pid, g in df_indiv.groupby("patient_id", sort=False):
                    g = g.sort_values("timepoint")
                    xi = g["timepoint"].cat.codes + x_shift

                    ax.plot(
                        xi,
                        g["VCSS"],
                        color=line_colo,
                        alpha=0.12,      # light -> background
                        linewidth=1.0,
                        zorder=1
                    )

                # ----
                # Overlay mean
                # ----
                ax.plot(
                    x,
                    df_mean["mean"],
                    color=line_colo,
                    linewidth=line_widt,
                    marker="o",
                    markersize=mark_size,
                    markerfacecolor=mark_colo,
                    label=line_labl,
                    zorder=3,
                )
            case_d(df_orig_R, df_R, ax, x_shift=-0.1, line_colo='blue', line_labl='Right Limb (R)')
            case_d(df_orig_L, df_L, ax, x_shift=0.1, line_colo='red', line_labl='Left Limb (L)')
            #plt.show()
            pass       

        # ====
        case "e":  # TODO !!!!
        # Observed means ± CI (secondary layer)
        # Mixed model estimates ± CI (primary layer)
        # ====
        
            # Observed values
            # ----
            yerr=[
                df_fram["mean"] - df_fram["ci_lower"],
                df_fram["ci_upper"] - df_fram["mean"]
            ]
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