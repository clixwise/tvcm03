    
#from __future__ import annotations
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_vcss_11_stat_ import StatTranVCSS_11_mixd
    
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
import patsy
from vcss_10_mixd.c02_vcss_11_stat_mixd_open import exec_stat_mixd_open
from vcss_10_mixd.c02_vcss_11_stat_mixd_clau import exec_stat_mixd_clau
from vcss_10_mixd.c02_vcss_11_stat_adat import StatTranVCSS_11_mixd

# ----
# Timepoint outcomes
# ----

def exec_stat_mixd(stat_tran_adat: StatTranVCSS_11_mixd) -> None:
    df_lon1 = exec_stat_mixd_raw(stat_tran_adat)
    exec_stat_mixd_clau(stat_tran_adat, df_lon1)
    exec_stat_mixd_open(stat_tran_adat, df_lon1)
    
def exec_stat_mixd_raw(stat_tran_adat: StatTranVCSS_11_mixd) -> None:
    # from vcss_10_mixd.c02_vcss_11_stat_ import StatTranVCSS_11_mixd

    trac = True

    # Data
    # ---- 
    df_frax = stat_tran_adat.stat_tran.frax
    
    # Trac
    # ----
    if trac:
        print_yes(df_frax, labl='df_frax')

    # Data : df_fram
    # ====
    df_modl = df_frax.copy()
    # df_modl["timepoint"] = pd.Categorical(df_modl["timepoint"],categories=["T0", "T1", "T2"],ordered=True)
    
    # Raw  : df_lon1
    # ===
    df_lon1 = (df_frax.groupby(["timepoint", "Limb"]).agg(mean=("VCSS", "mean"),sd=("VCSS", "std"),n=("VCSS", "count")).reset_index())
    df_lon1["se"] = df_lon1["sd"] / np.sqrt(df_lon1["n"])
    df_lon1["ci_lower"] = df_lon1["mean"] - 1.96 * df_lon1["se"]
    df_lon1["ci_upper"] = df_lon1["mean"] + 1.96 * df_lon1["se"]
    if trac:
        print_yes(df_lon1, labl="df_lon1")
    
    '''
    ----
    Fram labl : df_lon1
    ----
    df:6 type:<class 'pandas.core.frame.DataFrame'>
    timepoint Limb  mean  sd    n   se    ci_lower  ci_upper
    0  T0        R    6.73  4.98  30  0.91  4.95      8.51
    1  T0        L    6.97  5.24  30  0.96  5.09      8.84
    2  T1        R    4.03  4.64  30  0.85  2.37      5.69
    3  T1        L    4.40  4.76  30  0.87  2.70      6.10
    4  T2        R    3.30  4.42  30  0.81  1.72      4.88
    5  T2        L    3.70  4.49  30  0.82  2.09      5.31
    :RangeIndex(start=0, stop=6, step=1)
    :Index(['timepoint', 'Limb', 'mean', 'sd', 'n', 'se', 'ci_lower', 'ci_upper'], dtype='object')
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6 entries, 0 to 5
    Data columns (total 8 columns):
    #   Column     Non-Null Count  Dtype
    ---  ------     --------------  -----
    0   timepoint  6 non-null      category
    1   Limb       6 non-null      category
    2   mean       6 non-null      float64
    3   sd         6 non-null      float64
    4   n          6 non-null      int64
    5   se         6 non-null      float64
    6   ci_lower   6 non-null      float64
    7   ci_upper   6 non-null      float64
    '''
    
    stat_tran_adat.resu_fram = df_frax
    stat_tran_adat.resu_plot = pd.DataFrame()
    stat_tran_adat.resu_plot_raw = df_lon1
    
    # Exit
    # ----
    return df_lon1
    
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
        #
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        print ("Categorical columns:")
        for column in categorical_columns:
            categories = sorted(df[column].dropna().unique())  # Supprime les NaN et trie
            print(f"Column: {column}: [{', '.join(categories)}]")
    pass