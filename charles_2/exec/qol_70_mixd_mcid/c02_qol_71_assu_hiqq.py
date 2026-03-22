    
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_qol_71_assu_ import AssuTranQOL_71_hiqq
    
import pandas as pd
import sys
import os  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        
# ----
# Assumption : histogram normality thru qq-plot
# https://chatgpt.com/c/693e6518-a904-8525-8023-774a660fcc0d
# A1 — completeness (pati_isok)
# A2 — score distribution (histogram, Q–Q, skew/kurtosis)
# C1 — ceiling/floor effects
# ----

def exec_assu_hiqq(assu_tran_hiqq: AssuTranQOL_71_hiqq) -> None:
    from qol_70_mixd_mcid.c02_qol_71_assu_ import AssuTranQOL_71_hiqq
    
    trac = True

    # Data
    # ---- 
    df_fram = assu_tran_hiqq.assu_tran.fram
    
    # Trac
    # ----
    if trac:
        print_yes(df_fram, labl="df_fram")

    # Exec
    # ----  
    '''
               patient_id        T0    T1         T2 
    0          PT_2024_02_00078  46.9  45.860276  53.349384
    1          PT_2024_02_08277  47.2  51.987495  54.795861
    2          PT_2024_02_10578  50.5  54.055014  58.917011
    '''
    df_wide = (
        df_fram
        .pivot(index="patient_id", columns="timepoint", values="VEINES_QOL_t")
        .reset_index()
    )
    #
    if trac:
        print_yes(df_wide, labl="df_wide")
    
    # Exit
    # ----
    assu_tran_hiqq.resu_plot = df_wide
    
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

'''
    # Panel A : distribution
    axes[0].hist(baseline_scores, bins=8, edgecolor="white", color="#4C72B0")
    axes[0].axvline(baseline_scores.mean(), color="red",    linestyle="--", label=f"Mean {baseline_scores.mean():.1f}")
    axes[0].axvline(baseline_scores.median(), color="orange", linestyle="--", label=f"Median {baseline_scores.median():.1f}")
    axes[0].set_title("Distribution")
    axes[0].set_xlabel("VEINES-QOL score")
    axes[0].legend(fontsize=8)

    # Panel B : boxplot with individual points
    axes[1].boxplot(baseline_scores, vert=True, patch_artist=True, boxprops=dict(facecolor="#4C72B0", alpha=0.4))
    axes[1].scatter([1]*len(baseline_scores), baseline_scores, alpha=0.6, color="#4C72B0", zorder=3)
    axes[1].set_title("Spread & Outliers")
    axes[1].set_ylabel("VEINES-QOL score")
    axes[1].set_xticks([])

    # Panel C : Q-Q plot
    stats.probplot(baseline_scores, dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q Plot (Normality)")

    plt.tight_layout()
    plt.savefig("T0_profile.png", dpi=150)
    plt.show()
'''