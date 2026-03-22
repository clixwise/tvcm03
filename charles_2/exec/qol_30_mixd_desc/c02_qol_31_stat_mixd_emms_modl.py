    
    
#from __future__ import annotations
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_qol_31_stat_ import StatTranQOL_31_mixd
    
import pandas as pd
import numpy as np
from pandas import DataFrame
from qol_30_mixd_desc.c02_qol_31_stat_adat import StatTranQOL_31_mixd
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----
# Timepoint outcomes
# ----
#
# pairwise contrasts (T1–T0, T2–T0, T2–T1) 
# https://copilot.microsoft.com/shares/M71sUTzXVPVFDSEeZkTLq
#
def exec_stat_mixd_emms_modl(stat_tran_adat: StatTranQOL_31_mixd, df_modl, result) -> None:
    # from qol_30_mixd_desc.c02_qol_31_stat_ import StatTranQOL_31_mixd

    trac = True

    # Exec : df_emms : EMMs for timepoint in model: VEINES_QOL_t ~ C(timepoint) ; categories: T0 (ref), T1, T2
    # ----
    # Fixed-effect params only
    fe = result.fe_params
    vcov_fe = result.cov_params().loc[fe.index, fe.index]

    levels = pd.Categorical(df_modl["timepoint"]).categories.tolist()
    ref = levels[0]  # should be 'T0'

    rows = []
    for lvl in levels:
        # Build contrast vector for fixed effects
        c = np.zeros(len(fe))
        # Intercept always 1
        c[fe.index.get_loc("Intercept")] = 1.0
        # Add dummy if not reference
        if lvl != ref:
            # param name is like C(timepoint)[T.T1] for lvl='T1'
            pname = f"C(timepoint)[T.{lvl}]"
            if pname in fe.index:
                c[fe.index.get_loc(pname)] = 1.0
        #
        emm = float(c @ fe)
        se = float(np.sqrt(c @ vcov_fe @ c))
        #
        rows.append({
            "timepoint": lvl,
            "EMM": emm,
            "SE": se,
            "Lower95": emm - 1.96 * se,
            "Upper95": emm + 1.96 * se
        })
    #
    df_emms = pd.DataFrame(rows)
    #
    if trac:
        print_yes(df_emms, "df_emms")
    
    # Exit
    # ----    
    stat_tran_adat.mixd_emms_modl = df_emms
    
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# TODO
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def plot_emm_trajectory(emms):
        """
        Pure matplotlib trajectory plot for EMMs with 95% CI.
        Expects columns: timepoint, EMM, Lower95, Upper95
        """

        # X positions
        x = range(len(emms))

        plt.figure(figsize=(6, 4))

        # Plot the EMM line
        plt.plot(
            x,
            emms["EMM"],
            marker="o",
            color="black",
            linewidth=2
        )

        # Add vertical error bars
        for i, row in emms.iterrows():
            plt.vlines(
                x=i,
                ymin=row["Lower95"],
                ymax=row["Upper95"],
                color="black",
                linewidth=2
            )

        # Formatting
        plt.xticks(x, emms["timepoint"])
        plt.xlabel("Timepoint")
        plt.ylabel("VEINES t-score (EMM ± 95% CI)")
        plt.title("Estimated Marginal Means of VEINES t-score")
        plt.grid(axis="y", linestyle=":", alpha=0.4)
        plt.tight_layout()
        plt.show()
        pass

    plot_emms = False
    if plot_emms:
        plot_emm_trajectory(df_emms)

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