    
    
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
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import shapiro, normaltest, levene, skew, kurtosis
import pandas as pd
import numpy as np
from scipy.stats import shapiro, skew, kurtosis, anderson
import matplotlib.pyplot as plt
import scipy.stats as stats

# ----
# https://copilot.microsoft.com/shares/7rZLRYu1wYqmA3sQABBX4
# ----
#
# Random effects : checking for normality
# ============
# Grph
# ====
# see    : c02_qol_11_grph_plot_mixd_resi : residuals
# Desc : check for non-normality LMM approach [https://chat.mistral.ai/chat/8a997b72-eed0-4629-b55e-9d3aa089ae45]
#
def exec_stat_mixd_assu_rand(stat_tran_adat: StatTranQOL_31_mixd, df_modl, result) -> None:
    # from qol_30_mixd_desc.c02_qol_31_stat_ import StatTranQOL_31_mixd

    trac = True
    
    # Exec
    # ----
    # Extract random intercepts
    random_effects = result.random_effects
    re_values = np.array([re[0] for re in random_effects.values()])

    # Basic stats
    re_mean = np.mean(re_values)
    re_std = np.std(re_values)
    re_skew = skew(re_values)
    re_kurt = kurtosis(re_values)

    # Normality tests
    re_shapiro_stat, re_shapiro_p = shapiro(re_values)
    re_anderson = anderson(re_values)

    re_results = [
        {
            'Component': 'Random Intercepts',
            'Assumption': 'Mean ≈ 0',
            'Metric': 'Mean',
            'Value': f"{re_mean:.4f}",
            'Comment': "✓ Close to zero" if abs(re_mean) < 0.1 else "⚠ Deviates from zero"
        },
        {
            'Component': 'Random Intercepts',
            'Assumption': 'Dispersion',
            'Metric': 'Std Dev',
            'Value': f"{re_std:.4f}",
            'Comment': "✓ Reasonable"  # subjective but informative
        },
        {
            'Component': 'Random Intercepts',
            'Assumption': 'Normality',
            'Metric': f"Shapiro-Wilk W={re_shapiro_stat:.4f}",
            'Value': f"p={re_shapiro_p:.4f}",
            'Comment': "✓ Normal" if re_shapiro_p > 0.05 else "⚠ Non-normal"
        },
        {
            'Component': 'Random Intercepts',
            'Assumption': 'Normality (Anderson)',
            'Metric': f"A²={re_anderson.statistic:.4f}",
            'Value': f"Critical={re_anderson.critical_values[2]:.4f}",
            'Comment': "✓ Normal" if re_anderson.statistic < re_anderson.critical_values[2] else "⚠ Non-normal"
        },
        {
            'Component': 'Random Intercepts',
            'Assumption': 'Shape',
            'Metric': 'Skewness',
            'Value': f"{re_skew:.4f}",
            'Comment': "✓ Symmetric" if abs(re_skew) < 0.5 else "⚠ Skewed"
        },
        {
            'Component': 'Random Intercepts',
            'Assumption': 'Shape',
            'Metric': 'Kurtosis',
            'Value': f"{re_kurt:.4f}",
            'Comment': "✓ Mesokurtic" if abs(re_kurt) < 1 else "⚠ Heavy/light tails"
        }
    ]

    df_rand = pd.DataFrame(re_results) 
    if trac:
        print_yes(df_rand, labl="df_rand")  
        
    # Exec
    # ----
    df_rand_plot = pd.DataFrame(random_effects).T
    df_rand_plot.index.name = 'patient_id'
    df_rand_plot = df_rand_plot.reset_index()
    if trac:
        print_yes(df_rand_plot, labl="df_rand_plot")  
    
    # Exit
    # ----    
    stat_tran_adat.mixd_assu_rand = df_rand
    stat_tran_adat.mixd_assu_rand_plot = df_rand_plot
    
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