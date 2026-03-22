    
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_qol_01_stat_ import StatTranQOL_01_mixd
    
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from patsy import dmatrix
import warnings
import matplotlib.pyplot as plt
import seaborn as sns


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
    from qol.c02_qol_01_stat_ import StatTranQOL_01_mixd

    trac = True
    
    # Publ
    # ----
    mark_dict = stat_tran_adat.stat_tran.proc_tran.orch_tran.ta01_base_char

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
              
     # Count how many rows each patient has
    counts = df_modl['patient_id'].value_counts()
    singletons = counts[counts == 1]
    if not singletons.empty:
        print(f"⚠️ Warning: {len(singletons)} patients have only 1 timepoint.")
        print(singletons) 
    else:  
        print(f"No singletons")
        
    # Check if the score is constant for any timepoint : If any SD is 0.0, that's your problem.
    variance_check = df_modl.groupby('timepoint')['VEINES_QOL_t'].std()
    print("Standard Deviation by Timepoint:")
    print(variance_check)

    # Create the matrix the model sees
    matrix = dmatrix("C(timepoint)", df_modl)
    column_names = matrix.design_info.column_names
    rank = np.linalg.matrix_rank(matrix)
    print(f"Design Matrix Columns: {column_names}")
    print(f"Matrix Rank: {rank}")
    if rank < matrix.shape[1]:
        print("❌ ERROR: Your timepoints are collinear. Check if a timepoint only exists for one patient.")   
        
    # Check if any patient has 0 variance across their timepoints
    patient_variance = df_modl.groupby('patient_id')['VEINES_QOL_t'].std()
    flatline_patients = patient_variance[patient_variance == 0]

    if not flatline_patients.empty:
        print(f"⚠️ Warning: {len(flatline_patients)} patients have identical scores at every timepoint.")
        print(flatline_patients)
    else:
        print(f"No patients have identical scores at every timepoint.")
        
        
        
    # 1. Calculate the 'Within-Patient' spread
    # This shows how much each patient's score actually moves
    patient_variation = df_modl.groupby('patient_id')['VEINES_QOL_t'].std().mean()

    # 2. Calculate the 'Between-Patient' spread
    # This shows how different the patients are from each other
    between_variation = df_modl.groupby('patient_id')['VEINES_QOL_t'].mean().std()

    print(f"Average variation within a patient: {patient_variation:.2f}")
    print(f"Variation between different patients: {between_variation:.2f}")

    # Logic: If between_variation is much smaller than patient_variation, 
# the model will naturally hit 'Group Var = 0'.    
        
        
        
        
        
        
        
        
     # 1. Apply Jitter
    # We use a very small scale so it doesn't impact clinical meaning
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

    # Execute with your dataframe
    # plot_qol_spaghetti(df_modl)    
        
    # Glob
    # ----
    # Global model-level statistics
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

    # Convert to DataFrame if you prefer tabular format
    df_glob = pd.DataFrame.from_dict(model_info, orient="index", columns=["Value"])
    if trac:
        print_yes(df_glob, labl="df_glob")
    
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