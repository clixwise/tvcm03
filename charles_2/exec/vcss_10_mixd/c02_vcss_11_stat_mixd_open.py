    
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
from pandas import DataFrame
from vcss_10_mixd.c02_vcss_11_stat_adat import StatTranVCSS_11_mixd

# ----
# Timepoint outcomes
# ----

# https://chatgpt.com/c/693e6518-a904-8325-8023-774a660fcc0d
#
# https://chatgpt.com/c/698849d4-d684-832d-a367-fd26b2f1ac4f 2026-02-08
#
def exec_stat_mixd_open(stat_tran_adat: StatTranVCSS_11_mixd, df_lon1:DataFrame) -> None:
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
    # Note : we can expermiment with 'sum', 'mean', 'max' !!! [2026-02-08] !!! 
    df_modl = (
    df_frax
        .groupby(["patient_id", "timepoint"], observed=False)["VCSS"]
        .sum()
        .reset_index(name="VCSS_sum")
    )
    # df_modl["timepoint"] = pd.Categorical(df_modl["timepoint"],categories=["T0", "T1", "T2"],ordered=True)
  
    # Modl : df_modl [Fit the linear mixed-effects model]
    # ==== 
    model = smf.mixedlm("VCSS_sum ~ C(timepoint)", df_modl, groups=df_modl["patient_id"])
    result = model.fit(reml=True, method="powell")
    if trac:
        print(result.summary())

    # Glob
    # ====
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
    if trac:
        print_yes(df_glob, labl="df_glob")
        
    # Deta
    # ====
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
    if trac:
        print_yes(df_deta, labl="df_deta")

    # Plot : Estimated marginal means
    # ====
    fe_params = result.fe_params
    cov_all = result.cov_params()

    cov_fe = cov_all.loc[fe_params.index, fe_params.index]

    emm_design = pd.DataFrame(
        0.0,
        index=["T0", "T1", "T2"],
        columns=fe_params.index
    )

    emm_design.loc[:, "Intercept"] = 1.0

    if "C(timepoint)[T.T1]" in emm_design.columns:
        emm_design.loc["T1", "C(timepoint)[T.T1]"] = 1.0

    if "C(timepoint)[T.T2]" in emm_design.columns:
        emm_design.loc["T2", "C(timepoint)[T.T2]"] = 1.0

    emm_mean = emm_design @ fe_params
    emm_var = np.diag(emm_design @ cov_fe @ emm_design.T)
    emm_se = np.sqrt(emm_var)

    df_plot = pd.DataFrame({
        "timepoint": emm_design.index,
        "mean": emm_mean.values,
        "se": emm_se,
        "ci_lower": emm_mean.values - 1.96 * emm_se,
        "ci_upper": emm_mean.values + 1.96 * emm_se
    })

    df_plot["n"] = (
    df_modl
    .groupby("timepoint", observed=False)["VCSS_sum"]
    .count()
    .reindex(["T0", "T1", "T2"])
    .values
    )
    cate_list = ["T0", "T1", "T2"]
    df_plot["timepoint"] = pd.Categorical(
        df_plot["timepoint"],
        categories=cate_list,
        ordered=True
    )
    if trac:
        print_yes(df_plot, labl="df_plot")
        
    # Exit
    # ----
    stat_tran_adat.resu_fram = df_frax # original Tx records / patient
    stat_tran_adat.resu_glob = df_glob
    stat_tran_adat.resu_deta = df_deta
    stat_tran_adat.resu_plot_lme = df_plot # derived  Tx records / timepoint
    
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
        categorical_columns = df.select_dtypes(include=['category']).columns
        print ("Categorical columns:")
        for column in categorical_columns:
            categories = sorted(df[column].dropna().unique())  # Supprime les NaN et trie
            print(f"Column: {column}: [{', '.join(categories)}]")
    pass