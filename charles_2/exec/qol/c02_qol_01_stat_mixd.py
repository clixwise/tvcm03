    
#from __future__ import annotations
#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from c02_qol_01_stat_ import StatTranQOL_01_mixd
    
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from patsy import dmatrix
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from qol.c02_qol_01_stat_adat import StatTranQOL_01_mixd

# ----
# Timepoint outcomes
# ----

# https://chatgpt.com/c/693e6518-a904-8325-8023-774a660fcc0d
# https://copilot.microsoft.com/shares/wieaGQFPWWSBPtpm6kP64

'''
Intercept = mean t‑score at T0.
Coefficient for C(timepoint)[T1] = mean change T1–T0.
Coefficient for C(timepoint)[T2] = mean change T2–T0.
Random intercept = each patient’s baseline level.
'''
'''
Fit      	            Estimation method	Optimizer	      Practical effect
1. fit(reml=True)	    REML (True)         default optimizer Standard, safe, may struggle with convergence in some datasets
2. fit(method="lbfgs")	REML (by default)	L‑BFGS	          Same statistical method, but more robust numerical optimization
'''

def exec_stat_mixd(stat_tran_adat: StatTranQOL_01_mixd) -> None:
    #from qol.c02_qol_01_stat_ import StatTranQOL_01_mixd

    trac = True
    
    # Publ
    # ----
    mark_dict = stat_tran_adat.stat_tran.proc_tran.orch_tran.ta04_endp_prim_modl

    # Data
    # ---- 
    df_fram = stat_tran_adat.stat_tran.fram
    
    # Trac
    # ----
    if trac:
        df_trac = df_fram[['workbook','patient_id','timepoint','VEINES_QOL_t']]
        df_trac = df_fram[['patient_id','timepoint','VEINES_QOL_t']]
        print_yes(df_trac, labl='df_fram')
        print_yes(df_trac.iloc[9:12], labl='df_trac.iloc[9:12]')

    # Data
    # ----
    df_modl = df_fram.copy()
    if trac:
        # print_yes(df_modl, labl='df_modl')
        print_yes(df_modl[["patient_id","timepoint","VEINES_QOL_t"]], labl='df_modl')
    
    # Warn : all patients should have at least 2 timepoints
    # ----
    counts = df_modl['patient_id'].value_counts()
    singletons = counts[counts == 1]
    if not singletons.empty:
        print(f"⚠️ Warning: {len(singletons)} patients have only 1 timepoint.")
        print(singletons) 
        raise Exception()
    
    # Warn : Check if any patient has 0 variance across their timepoints
    # ----
    patient_variance = df_modl.groupby('patient_id')['VEINES_QOL_t'].std()
    flatline_patients = patient_variance[patient_variance == 0]
    if not flatline_patients.empty:
        print(f"⚠️ Warning: {len(flatline_patients)} patients have identical scores at every timepoint.")
        print(flatline_patients)
        raise Exception()
    
    # Warn : the score should not be constant for any timepoint : If any SD is 0.0, that's your problem.
    # ----
    variance_check = df_modl.groupby('timepoint')['VEINES_QOL_t'].std()
    if (variance_check == 0).any():
        print("Error: Constant score detected at one or more timepoints.")
        print("Standard Deviation by Timepoint:")
        print(variance_check)
        raise Exception("Execution halted: Zero variance found in timepoint groups.")

    # Warn : Create the matrix the model sees
    # ----
    matrix = dmatrix("C(timepoint)", df_modl)
    column_names = matrix.design_info.column_names
    rank = np.linalg.matrix_rank(matrix)
    print(f"Design Matrix Columns: {column_names}")
    print(f"Matrix Rank: {rank}")
    if rank < matrix.shape[1]:
        print("❌ ERROR: Your timepoints are collinear. Check if a timepoint only exists for one patient.") 
        raise Exception()  
        
    # Prec
    # ----    
    # 1. Calculate the 'Within-Patient' spread
    # This shows how much each patient's score actually moves
    patient_variation = df_modl.groupby('patient_id')['VEINES_QOL_t'].std().mean()
    # 2. Calculate the 'Between-Patient' spread
    # This shows how different the patients are from each other
    between_variation = df_modl.groupby('patient_id')['VEINES_QOL_t'].mean().std()
    # Logic: If between_variation is much smaller than patient_variation, the model will naturally hit 'Group Var = 0'.
    print(f"Average variation within a patient: {patient_variation:.2f}")
    print(f"Variation between different patients: {between_variation:.2f}")
    if between_variation < patient_variation:
        print(f"The model will naturally hit 'Group Var = 0'")

     # Prec Apply Jitter : We use a very small scale so it doesn't impact clinical meaning
     # ----
    jitt = False
    if jitt:
        seed = 2026
        rng = np.random.default_rng(seed=seed)
        df_modl['VEINES_QOL_t'] = df_modl['VEINES_QOL_t'] + rng.normal(0, 0.001, size=len(df_modl))
       
    # Modl : Fit the linear mixed-effects model
    # Debug : https://gemini.google.com/app/bfb2c5b464a0ed2e !!!
    # ---- 
    # Note : VEINES_QOL ~ time + bilateral + (1 | patient_id)
    model = smf.mixedlm("VEINES_QOL_t ~ C(timepoint)", df_modl, groups=df_modl["patient_id"])
    false = False
    if false:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The MLE may be on the boundary")
            result = model.fit(reml=True, method="powell", maxiter=2000)
    # Use Powell for the final run
    # result = model.fit(reml=True, method="lbfgs") # -> Singular Matrix
    # result = model.fit(reml=True) # -OK
    # result = model.fit(reml=True, method="bfgs", maxiter=2000) # OK but marginal
    # result = model.fit(reml=True, method="cg") # OK but marginal
    result = model.fit(reml=True, method="powell", maxiter=2000) # The Winner
    if trac:
        print(result.summary())
        
    # Extracting results into a dictionary for easy reporting
    stats_summary = {
        "Intercept (T0)": f"{result.params['Intercept']:.2f}",
        "T1_Change": f"{result.params['C(timepoint)[T.T1]']:.2f}",
        "T1_p_value": f"{result.pvalues['C(timepoint)[T.T1]']:.3f}",
        "T2_Change": f"{result.params['C(timepoint)[T.T2]']:.2f}",
        "T2_p_value": f"{result.pvalues['C(timepoint)[T.T2]']:.3f}",
        # "Converged": result.mle_retvals['converged']
    }

    print("--- Final Statistics ---")
    for key, val in stats_summary.items():
        print(f"{key}: {val}")
                
    # Glob
    # ----
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
    "wald_test": result.f_test("C(timepoint)[T.T1] = 0, C(timepoint)[T.T2] = 0"), # Wald test to produce overall p
    "converged": result.converged,
    }
    #
    df_glob = pd.DataFrame.from_dict(model_info, orient="index", columns=["Value"])
    if trac:
        print_yes(df_glob, labl="df_glob")
    
    # Mark : Wald test
    # ----
    # 1. Perform the test and store it
    wald_res = result.f_test("C(timepoint)[T.T1] = 0, C(timepoint)[T.T2] = 0")
    # 2. Extract values from the ContrastResults object
    f_val = wald_res.fvalue
    p_val = wald_res.pvalue
    df_n = wald_res.df_num
    df_d = wald_res.df_denom
    # 3. Format the string
    p_str = "p< 0.001" if p_val < 0.001 else f"p= {p_val:.3f}"
    wald_info = f"({int(df_n)},{int(df_d)}) = {f_val:.2f}; {p_str}"
    mark_dict['Wald F'] = (wald_info, Path(__file__).stem)   # F(2,81) = 29.27; p < 0.001 # you are fully justified in stating an overall time effect.
    
    # Deta
    # ----   
    params = result.params            # coefficients
    bse = result.bse                  # standard errors
    zvals = result.tvalues            # z statistics
    pvals = result.pvalues            # p-values
    conf_int = result.conf_int()      # confidence intervals (DataFrame)
    # 
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
        
        
    # Mark : adjusted mean difference 
    # ----
    def mean_diff_util(df, time):
        beta = df.loc[f"C(timepoint)[T.{time}]", "Coef"]
        ci_l = df.loc[f"C(timepoint)[T.{time}]", "CI_lower"]
        ci_u = df.loc[f"C(timepoint)[T.{time}]", "CI_upper"]
        pval = df.loc[f"C(timepoint)[T.{time}]", "P>|z|"]
        p_form = "p < 0.001" if pval < 0.001 else f"p = {pval:.3f}"
        mean_diff = (f"{beta:.2f} " f"(95% CI {ci_l:.2f} to {ci_u:.2f}; " f"{p_form}).")
        return mean_diff
    mean_diff = mean_diff_util(df_deta, 'T1') ; mark_dict['Adjusted mean difference β at T1'] = (mean_diff, Path(__file__).stem)
    mean_diff = mean_diff_util(df_deta, 'T2') ; mark_dict['Adjusted mean difference β at T2'] = (mean_diff, Path(__file__).stem)

    # ----
    # Extract fixed effects and covariance
    # ----
    fe_params = result.fe_params              # pandas Series
    cov_all = result.cov_params()             # full covariance matrix
    cov_fe = cov_all.loc[
        fe_params.index,
        fe_params.index
    ]

    # ----
    # Build aligned EMM design matrix
    # ----
    emm_design = pd.DataFrame(
        0.0,
        index=["T0", "T1", "T2"],
        columns=fe_params.index
    )

    # Intercept
    emm_design.loc[:, "Intercept"] = 1.0

    # Timepoint contrasts
    if "C(timepoint)[T.T1]" in emm_design.columns:
        emm_design.loc["T1", "C(timepoint)[T.T1]"] = 1.0
    if "C(timepoint)[T.T2]" in emm_design.columns:
        emm_design.loc["T2", "C(timepoint)[T.T2]"] = 1.0

    # ----
    # EMMs, SEs, CIs
    # ----
    emm_mean = emm_design @ fe_params
    emm_var = np.diag(
        emm_design @ cov_fe @ emm_design.T
    )
    emm_se = np.sqrt(emm_var)

    # Fini
    # ----
    df_plot = pd.DataFrame({
        "timepoint": emm_design.index,
        "mean": emm_mean.values,
        "se": emm_se,
        "ci_lower": emm_mean.values - 1.96 * emm_se,
        "ci_upper": emm_mean.values + 1.96 * emm_se
    })

    # Exec : Add n per timepoint (descriptive, not model-based)
    # ----
    # df_plot["timepoint"] = pd.Categorical(df_plot["timepoint"], categories=["T0", "T1", "T2"], ordered=True)
    df_plot["n"] = (
        df_modl
        .groupby("timepoint", observed=False)["VEINES_QOL_t"]
        .count()
        .reindex(["T0", "T1", "T2"])
        .values
    )
    #
    cate_list = ['T0','T1','T2'] ; df_plot["timepoint"] = pd.Categorical(df_plot["timepoint"], categories=cate_list, ordered=True)
    #
    if trac:
        print_yes(df_plot, labl="df_plot")

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