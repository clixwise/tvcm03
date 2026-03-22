    
#from __future__ import annotations
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_qol_31_stat_ import StatTranQOL_31_mixd
    
import pandas as pd
import numpy as np
from pandas import DataFrame
from qol_30_mixd_desc.c02_qol_31_stat_adat import StatTranQOL_31_mixd
    
import pandas as pd
import statsmodels.formula.api as smf
# from qol_10_mixd.c02_qol_11_stat_mixd_gemi_to_copi import exec_stat_mixd_gemi_to_copi
from qol_30_mixd_desc.c02_qol_31_stat_mixd_mean_raww import exec_stat_mixd_mean_raww
from qol_30_mixd_desc.c02_qol_31_stat_mixd_mean_modl import exec_stat_mixd_mean_modl
from qol_30_mixd_desc.c02_qol_31_stat_mixd_mean_merg import exec_stat_mixd_mean_merg
from qol_30_mixd_desc.c02_qol_31_stat_mixd_emms_modl import exec_stat_mixd_emms_modl
from qol_30_mixd_desc.c02_qol_31_stat_mixd_pair_modl import exec_stat_mixd_pair_modl
from qol_30_mixd_desc.c02_qol_31_stat_mixd_assu_rand import exec_stat_mixd_assu_rand
from qol_30_mixd_desc.c02_qol_31_stat_mixd_assu_resi import exec_stat_mixd_assu_resi
from qol_30_mixd_desc.c02_qol_31_stat_mixd_assu_merg import exec_stat_mixd_assu_merg

# ----
# Timepoint outcomes
# ----

def exec_stat_mixd(stat_tran_adat: StatTranQOL_31_mixd) -> None:
    
    # from qol_30_mixd_desc.c02_qol_31_stat_ import StatTranQOL_31_mixd

    trac = True

    # Data
    # ---- 
    df_fram = stat_tran_adat.stat_tran.fram
    
    # Trac
    # ----
    if trac:
        print_yes(df_fram, "df_fram")

    # Data : df_fram
    # ----
    df_modl = df_fram.copy()
    # df_modl["timepoint"] = pd.Categorical(df_modl["timepoint"],categories=["T0", "T1", "T2"],ordered=True)
   
    # Modl : df_modl [Fit the linear mixed-effects model]
    # ---- 
    # modl = smf.mixedlm("VEINES_QOL_t ~ C(timepoint, Treatment(reference='T0'))", df_modl, groups=df_modl["patient_id"])
    # change_3m = params["C(timepoint, Treatment(reference='T0'))[T.T1]"]
    # https://chatgpt.com/c/699078a2-17c8-8393-9a47-5b31164971e2
    modl = smf.mixedlm("VEINES_QOL_t ~ C(timepoint)", df_modl, groups=df_modl["patient_id"])
    result = modl.fit(reml=True, method="powell")
    if trac:
        print(result.summary())
        
    # Assu
    # ----
    exec_stat_mixd_assu_rand(stat_tran_adat, df_modl.copy(), result)
    exec_stat_mixd_assu_resi(stat_tran_adat, df_modl.copy(), result)
    exec_stat_mixd_assu_merg(stat_tran_adat)
        
    # Exec 1 : df_mean : modeled means
    # ----
    exec_stat_mixd_mean_raww(stat_tran_adat, df_fram.copy())
    exec_stat_mixd_mean_modl(stat_tran_adat, df_modl.copy(), result)
    exec_stat_mixd_mean_merg(stat_tran_adat)
    
    # Exec 2 : df_emms : estimated marginal means
    # ----
    exec_stat_mixd_emms_modl(stat_tran_adat, df_modl.copy(), result)
   
    # Exec 3 : df_pair : pairwise contrasts (T1–T0, T2–T0, T2–T1) 
    # ----
    exec_stat_mixd_pair_modl(stat_tran_adat, df_modl.copy(), result)
    pass

    #exec_stat_mixd_gemi_to_copi(stat_tran_adat, df_lon1) # TO IMPORT FROM QOL_11
    #exec_stat_mixd_mist(stat_tran_adat)
    # exec_stat_mixd_copi(stat_tran_adat)
    #exec_stat_mixd_open(stat_tran_adat)

    ''''
    `vari` is the **variance of the estimated mean** at each timepoint, extracted from the model's **variance-covariance matrix** (`vcov`).
    ## Breakdown by Timepoint
    **For T0 (reference):**
    vari = vcov.loc["Intercept", "Intercept"]
    - Pure variance of the Intercept estimate
    - `Var(μ_T0) = Var(β_Intercept)`

    **For T1/T2 (vs T0):**
    vari = (vcov.loc["Intercept", "Intercept"] + 
        vcov.loc[coef, coef] + 
        2 * vcov.loc["Intercept", coef])
    - `Var(μ_T1) = Var(β_Intercept + β_T1)`
    - **Variance addition formula**: `Var(a + b) = Var(a) + Var(b) + 2⋅Cov(a,b)`
    - `coef = "C(timepoint)[T.T1]"` or `"C(timepoint)[T.T2]"`

    ## Why This Works
    Your **linear combination** `μ_T1 = Intercept + C(timepoint)[T.T1]` has variance:
    Var(μ_T1) = Var(Intercept) + Var(β_T1) + 2 × Cov(Intercept, β_T1)
    The `vcov` matrix contains **all pairwise variances/covariances** from the mixed model fit.

    ## Then...
    `se = np.sqrt(vari)` → **standard error** of the mean at each timepoint
    `ci_low = est - 1.96 * se` → **95% CI lower bound**

    **Publication note**: "Means and 95% CIs derived from model variance-covariance matrix 
    accounting for random effects and repeated measures." 
    [groups.google](https://groups.google.com/g/pystatsmodels/c/KXF3CxqYZcI)
    '''
    '''
    T0:  est=50.003, se=1.23  → 95% CI: 47.58–52.43  [precise population mean estimate]
    T0: residuals SD=4.54     → individual patients scatter ±4.54 around that mean [data spread]
    '''

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