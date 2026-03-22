    
#from __future__ import annotations
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_vcss_01_stat_ import StatTranVCSS_01_mixd
    
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from vcss.c02_vcss_01_stat_adat import StatTranVCSS_01_mixd

# ----
# Timepoint outcomes
# ----

# https://chatgpt.com/c/698849d4-d684-832d-a367-fd26b2f1ac4f

def exec_stat_mixd(stat_tran_adat: StatTranVCSS_01_mixd) -> None:
    # from vcss.c02_vcss_01_stat_ import StatTranVCSS_01_mixd

    trac = True
    
    # Data
    # ---- 
    df_frax = stat_tran_adat.stat_tran.frax
    
    # Trac
    # ----
    '''
    ----
    Fram labl : df_frax
    ----
    df:180 type:<class 'pandas.core.frame.DataFrame'>
        workbook                                                              patient_id        timepoint  Age Sexe Limb  VCSS
    90            2025-11-01 2025-11-01 T0 PT_2024_02_00078 MAKOLA ODETTE V03  PT_2024_02_00078  T0        59   F    L     5
    0             2025-11-01 2025-11-01 T0 PT_2024_02_00078 MAKOLA ODETTE V03  PT_2024_02_00078  T0        59   F    R     5
    ...
    179   2025-11-01 2025-11-01 T2 PT_2025_11_06389 YOKA MBONGO CHRISTIAN V04  PT_2025_11_06389  T2        51   M    L     0
    89    2025-11-01 2025-11-01 T2 PT_2025_11_06389 YOKA MBONGO CHRISTIAN V04  PT_2025_11_06389  T2        51   M    R    11
    [180 rows x 7 columns]
    Data columns (total 7 columns):
    #   Column      Non-Null Count  Dtype
    ---  ------      --------------  -----
    0   workbook    180 non-null    object
    1   patient_id  180 non-null    object
    2   timepoint   180 non-null    category
    3   Age         180 non-null    int64
    4   Sexe        180 non-null    category
    5   Limb        180 non-null    category
    6   VCSS        180 non-null    int64
    '''
    if trac:
        print_yes(df_frax, labl="df_frax")

    # =========================
    # Data
    # =========================
    df_modl = df_frax.copy()

    # =========================
    # Model
    # =========================
    model = smf.mixedlm("VCSS ~ C(timepoint) * C(Limb)", df_modl, groups=df_modl["patient_id"])
    result = model.fit(reml=True, method="powell")
    if trac:
        print(result.summary())

    # =========================
    # Global info
    # =========================
    groups = result.model.groups
    _, group_counts = np.unique(groups, return_counts=True)

    model_info = {
        "model": type(result.model).__name__,
        "dependent_variable": result.model.endog_names,
        "n_observations": int(result.nobs),
        "method": result.method,
        "n_groups": len(group_counts),
        "scale": result.scale,
        "log_likelihood": result.llf,
        "converged": result.converged,
        "min_group_size": int(group_counts.min()),
        "max_group_size": int(group_counts.max()),
        "mean_group_size": float(group_counts.mean()),
    }

    df_glob = pd.DataFrame.from_dict(model_info, orient="index", columns=["Value"])

    # =========================
    # Detailed coefficients
    # =========================
    params = result.params
    bse = result.bse
    zvals = result.tvalues
    pvals = result.pvalues
    conf_int = result.conf_int()

    df_deta = pd.DataFrame({
        "Coef": params,
        "Std.Err": bse,
        "z": zvals,
        "P>|z|": pvals,
        "CI_lower": conf_int[0],
        "CI_upper": conf_int[1]
    })

    # =========================
    # EMM computation
    # =========================
    fe_params = result.fe_params
    cov_all = result.cov_params()
    cov_fe = cov_all.loc[fe_params.index, fe_params.index]

    time_levels = ["T0", "T1", "T2"]
    limb_levels = ["R", "L"]

    index = pd.MultiIndex.from_product(
        [time_levels, limb_levels],
        names=["timepoint", "Limb"]
    )

    emm_design = pd.DataFrame(0.0, index=index, columns=fe_params.index)

    # Intercept
    emm_design.loc[:, "Intercept"] = 1.0

    # Time effects
    for t in ["T1", "T2"]:
        col = f"C(timepoint)[T.{t}]"
        if col in emm_design.columns:
            emm_design.loc[(t, slice(None)), col] = 1.0

    # Limb effect
    if "C(Limb)[T.L]" in emm_design.columns:
        emm_design.loc[(slice(None), "L"), "C(Limb)[T.L]"] = 1.0

    # Interaction
    for t in ["T1", "T2"]:
        col = f"C(timepoint)[T.{t}]:C(Limb)[T.L]"
        if col in emm_design.columns:
            emm_design.loc[(t, "L"), col] = 1.0

    # Predictions
    emm_mean = emm_design @ fe_params
    emm_var = np.diag(emm_design @ cov_fe @ emm_design.T)
    emm_se = np.sqrt(emm_var)

    # =========================
    # Plot dataframe
    # =========================
    df_plot = pd.DataFrame({
        # We do not want to create it as columns, since they already exist as indexes and we want to keep the 'reset_index()' in 'data52_oupu'
        # "timepoint": emm_design.index.get_level_values("timepoint").values,
        # "Limb": emm_design.index.get_level_values("Limb").values,{
        "mean": emm_mean.values,
        "se": emm_se,
        "ci_lower": emm_mean - 1.96 * emm_se,
        "ci_upper": emm_mean + 1.96 * emm_se,
        }, index=emm_design.index)
    df_plot = df_plot.reset_index()
    print_yes(df_plot, labl="df_plot [EMM dataframe]")
    
    # Descriptive n (not model based)
    df_plot["n"] = (
        df_modl
        .groupby(["timepoint", "Limb"], observed=False)["VCSS"]
        .count()
        .values
    )

    cate_list = ["T0", "T1", "T2"]
    df_plot["timepoint"] = pd.Categorical(
        df_plot["timepoint"], categories=cate_list, ordered=True
    )

    df_plot = df_plot.reset_index()
    if trac:
        print_yes(df_plot, labl="df_plot [EMM dataframe]")

    # Exit
    # ----
    stat_tran_adat.resu_glob = df_glob
    stat_tran_adat.resu_deta = df_deta
    stat_tran_adat.resu_plot = df_plot
    
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