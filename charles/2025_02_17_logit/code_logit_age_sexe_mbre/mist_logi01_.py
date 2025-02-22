import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.special import expit
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
from scipy.stats import norm

from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from mist_logi01_plot_01_perp import mist_logi01_plot_01_perp_glob, mist_logi01_plot_01_perp_deta
from mist_logi01_plot_01_mist import mist_logi01_plot_01_mist
from util_file_mngr import write
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
import scipy.stats as scs
from statsmodels.miscmodels.ordinal_model import OrderedModel

from scipy.interpolate import make_interp_spline

# -------------------------------
# Logit
# Google : https://www.google.com/search?q=youtube+Logit+Regression+Results&oq=youtube+Logit+Regression+Results
# Datatab : https://www.youtube.com/watch?v=C5268D9t9Ak
# -------------------------------

# ----
# Fonction prototype : age, sexe, ceap
# ----
def mist_logi01_exec11(what, df_line, logi_indx_list):
    
    # Sample
    if False:
        data = {
            'dossier': ['P1', 'P1', 'P1', 'P2'],
            'sexe': ['M', 'M', 'M', 'F'],
            'age': [54, 54, 54, 45],
            'age_bin': ['50_59', '50_59', '50_59', '40_49'],
            'mbre': ['G', 'G', 'D', 'G'],
            'ceap': ['C4', 'C5', 'C3', 'C2']
        }
        df = pd.DataFrame(data)

    # Prec
    # ----
    df = df_line[['age','sexe', 'ceap']]
    
    # Data
    # ----
    df['sexe_nume'] = df['sexe'].map({'M': logi_indx_list[0], 'F': logi_indx_list[1]}) # 'sexe' must be 'numeric'
    df['C3'] = (df['ceap'] == 'C3').astype(int) # Create binary outcome for C3
    df = df[['sexe_nume', 'age','C3', 'ceap']]
    df = df.rename(columns={'sexe_nume': 'sexe'})
    print (df)
    
    # Exec
    # ----
    # Prepare the data for logistic regression
    X = pd.get_dummies(df[['sexe', 'age']], drop_first=True)
    y = df['C3'] 
    log_model = sm.Logit(y, sm.add_constant(X)) # Fit the logistic regression model
    result = log_model.fit()
    print(f"\n---\nData : {what}\nLogistic Regression Perplexity [2025_02_12] : y='ceap'='C3' ; x='sexe','age'\n---")
    write(f"\n---\nData : {what}\nLogistic Regression Perplexity [2025_02_12] : y='ceap'='C3' ; x='sexe','age'\n---")
    print(result.summary())
    write(result.summary().as_text())
    pass

# ----
# Utilitaire pour approche généralisée
# ----
def logi(what, df, colu_cate_list, x_list):
    
    # Data
    # ----
    X = pd.get_dummies(df[x_list], drop_first=True)
    for ceap in colu_cate_list:
        df[ceap] = (df['ceap'] == ceap).astype(int) # Create binary outcome for 'ceap_levl'
    print (df)
    
    # Exec
    # ----
    print(f"\n---\nData : {what}\nLogistic Regression Perplexity [2025_02_12] : y={colu_cate_list} ; x='{x_list}'\n---")
    write(f"\n---\nData : {what}\nLogistic Regression Perplexity [2025_02_12] : y={colu_cate_list} ; x='{x_list}'\n---")
    resu_dict_deta = {}
    resu_dict_glob = {}
    for ceap in colu_cate_list:
        y = df[ceap]
        log_model = sm.Logit(y, sm.add_constant(X))
        result = log_model.fit()
        
        # Fit the OrderedModel
        '''
        ord_model = OrderedModel(y, X, distr='logit')
        result = ord_model.fit(method='bfgs')
        '''
        print(result.summary())
        write(result.summary().as_text())
        resu_dict_glob[ceap] = {
            "llr": result.llr_pvalue, # Likelihood ratio test
            "r-squared": result.prsquared, # Pseudo R-squared
            "aic": result.aic,
            "bic": result.bic
        }
        odds_ratio = np.exp(result.params)
        ci_lower = np.exp(result.conf_int().iloc[:, 0])
        ci_upper = np.exp(result.conf_int().iloc[:, 1])
        odds_ratio_neg = np.exp(-result.params)
        resu_dict_deta[ceap] = {
            "coef": result.params, # odds ratios
            "std err": result.bse,
            "z": result.tvalues,
            "P>|z|": result.pvalues,
            "[0.025": result.conf_int().iloc[:, 0], # CI
            "0.975]": result.conf_int().iloc[:, 1], # CI
            'odds_ratio' : odds_ratio,
            'ci_lower' : ci_lower,
            'ci_upper' : ci_upper,
            'odds_ratio_neg' : odds_ratio_neg
        }
        # break

    # Glob
    # ----
    df_resu_glob = pd.DataFrame(resu_dict_glob).transpose()
    frmt = lambda value: value if not (type(value) in (int, float, complex)) else f"{value:.3f}" # f"{value:.3e}" if value < 0.001 else f"{value:.3f}"
    df_resu_glob['llr'] = df_resu_glob['llr'].apply(frmt)
    df_resu_glob['r-squared'] = df_resu_glob['r-squared'].apply(frmt)
    df_resu_glob['aic'] = df_resu_glob['aic'].apply(frmt)
    df_resu_glob['bic'] = df_resu_glob['bic'].apply(frmt)
    # print (df_resu_glob)
    
    # Deta
    # ----
    resu_list_deta = []
    for ceap, metrics in resu_dict_deta.items():
        df_temp = pd.DataFrame(metrics)
        df_temp['ceap'] = ceap
        resu_list_deta.append(df_temp)
        # Adding a separator row of NaN values
        separator_row = pd.DataFrame('-', index=['-'], columns=df_temp.columns)
        separator_row['ceap'] = ceap
        resu_list_deta.append(separator_row)
    df_resu_deta = pd.concat(resu_list_deta).reset_index()
    #
    frmt = lambda value: value if not (type(value) in (int, float, complex)) else f"{value:.3f}" # f"{value:.3e}" if value < 0.001 else f"{value:.3f}"
    df_resu_deta['coef'] = df_resu_deta['coef'].apply(frmt)
    df_resu_deta['std err'] = df_resu_deta['std err'].apply(frmt)
    df_resu_deta['z'] = df_resu_deta['z'].apply(frmt)
    df_resu_deta['P>|z|'] = df_resu_deta['P>|z|'].apply(frmt)
    df_resu_deta['[0.025'] = df_resu_deta['[0.025'].apply(frmt)
    df_resu_deta['0.975]'] = df_resu_deta['0.975]'].apply(frmt)
    df_resu_deta['odds_ratio'] = df_resu_deta['odds_ratio'].apply(frmt)
    df_resu_deta['ci_lower'] = df_resu_deta['ci_lower'].apply(frmt)
    df_resu_deta['ci_upper'] = df_resu_deta['ci_upper'].apply(frmt)
    df_resu_deta['odds_ratio_neg'] = df_resu_deta['odds_ratio_neg'].apply(frmt)

    #
    cols = list(df_resu_deta.columns)
    df_resu_deta['pval'] = df_resu_deta.apply(lambda row: row['P>|z|'] if row['index'] not in ['-', 'const'] and pd.to_numeric(row['P>|z|'], errors='coerce') <= 0.05 else '', axis=1)
    pval_index = cols.index('P>|z|') + 1
    cols.insert(pval_index, cols.pop())
    #
    df_resu_deta['accs'] = df_resu_deta['ceap'].astype(str) + '_' + df_resu_deta['index'].astype(str)
    df_resu_deta.set_index('accs', drop=False, inplace=True) # drop column ; inplace ie same df
    df_resu_deta = df_resu_deta[['ceap', 'index', 'coef', 'std err', 'z', 'P>|z|', '[0.025', '0.975]','odds_ratio', 'ci_lower', 'ci_upper', 'odds_ratio_neg']]
    df_resu_deta = df_resu_deta.rename(columns={'ceap': 'ceap=y(dependant)', 'index': 'coef=x(independant)'})

    # Exit
    # ----
    return df_resu_glob, df_resu_deta


def mist_logi01_exec12(what, df_line, ind1_cate_list, colu_cate_list, ind1_name, colu_name, logi_indx_list):
   
    # Prec
    # ----
    df = df_line[['age', ind1_name, colu_name]]
    ind1_nam1 = ind1_cate_list[0]
    ind1_nam2 = ind1_cate_list[1]
    
    # Data
    # ----
    df[f'{ind1_name}_nume'] = df[ind1_name].map({ind1_nam1: logi_indx_list[0], ind1_nam2: logi_indx_list[1]}) # 'sexe' must be 'numeric'
    df = df[[f'{ind1_name}_nume', 'age', colu_name]]
    df = df.rename(columns={f'{ind1_name}_nume': ind1_name})
    
    # Exec
    # ----
    x_list = [ind1_name, 'age']
    df_resu_glob, df_resu_deta = logi(what, df, colu_cate_list, x_list)
    df_prnt_glob = df_resu_glob.copy()
    df_prnt_deta = df_resu_deta.copy()
    print(f"\n")
    write(f"\n")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        # print(df_resu.to_string(index=False))
        # write(df_resu.to_string(index=False))
        print(df_prnt_glob.reset_index(drop=True))
        write(df_prnt_glob.reset_index(drop=True).to_string())
        print(df_prnt_deta.reset_index(drop=True))
        write(df_prnt_deta.reset_index(drop=True).to_string())
    pass

    # Exit
    # ----
    return df_resu_glob, df_resu_deta

def mist_logi01_exec22(what, df_line, ind1_cate_list, ind2_cate_list, colu_cate_list, ind1_name, ind2_name, colu_name, logi_indx_list):
   
    # Prec
    # ----
    df = df_line[['age', ind1_name, ind2_name, colu_name]]
    ind1_nam1 = ind1_cate_list[0]
    ind1_nam2 = ind1_cate_list[1]
    ind2_nam1 = ind2_cate_list[0]
    ind2_nam2 = ind2_cate_list[1]
    
    # Data
    # ----
    df[f'{ind1_name}_nume'] = df[ind1_name].map({ind1_nam1: logi_indx_list[0], ind1_nam2: logi_indx_list[1]}) # 'sexe' must be 'numeric'
    df[f'{ind2_name}_nume'] = df[ind2_name].map({ind2_nam1: logi_indx_list[0], ind2_nam2: logi_indx_list[1]}) # 'mbre' must be 'numeric'
    df = df[[f'{ind1_name}_nume', f'{ind2_name}_nume', 'age', colu_name]]
    df = df.rename(columns={f'{ind1_name}_nume': ind1_name, f'{ind2_name}_nume': ind2_name})
    
    # Exec
    # ----
    x_list = [ind1_name, ind2_name,'age']
    df_resu_glob, df_resu_deta = logi(what, df, colu_cate_list, x_list)
    df_prnt_glob = df_resu_glob.copy()
    df_prnt_deta = df_resu_deta.copy()
    print(f"\n")
    write(f"\n")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        print(df_prnt_glob.reset_index(drop=True))
        write(df_prnt_glob.reset_index(drop=True).to_string())
        print(df_prnt_deta.reset_index(drop=True))
        write(df_prnt_deta.reset_index(drop=True).to_string())
    pass

    # Xlsx
    # ----
    xlsx = True
    if xlsx: 
        glob_file_name = 'df_glob_prnt.xlsx'
        deta_file_name = 'df_deta_prnt.xlsx'
        df_prnt_glob.to_excel(glob_file_name, index=True)
        df_prnt_deta.to_excel(deta_file_name, index=True)

    # Exit
    # ----
    return df_resu_glob, df_resu_deta

# ----
# Point 'd'entrée' : orchestrateur
# Note : les plots sous forme fichier sont aussi fournis séparément
# ----
def mist_logi01_exec(what, df_line, ind1_cate_list, ind2_cate_list, colu_cate_list, ind1_name, ind2_name, colu_name):

    wha1 = f"'{ind1_name}' '{colu_name}' ; {ind1_cate_list} {colu_cate_list}"
    wha2 = f"'{ind2_name}' '{colu_name}' ; {ind2_cate_list} {colu_cate_list}"
    logi_indx_list = [0,1] # version M=0,F=1 ; G=0,D=1
    logi_indx_list = [1,0] # version M=1,F=0 ; G=1,D=0 [préférée pour représentations graphiques]
 
    # Version prototype pour un Logit relatif au seul 'C3'
    # ----------------------------------------------------
    if True:
        # C(EAP) f(age, sexe) : Analyse
        mist_logi01_exec11(wha1, df_line, logi_indx_list)
    

    # Version généralisée pour un Logit relatif à tous les C(EAP) avec (age, sexe) OU (age,mbre)
    # ------------------------------------------------------------------------------------------
    if True:        
        # C(EAP) f(age, sexe) : Analyse
        df_resu_glob, df_resu_deta = mist_logi01_exec12(wha1, df_line, ind1_cate_list, colu_cate_list, ind1_name, colu_name, logi_indx_list)
        # C(EAP) f(age, sexe) : Plots
        # mist_logi01_plot_01_perp_glob(wha1, df_resu_glob) ; mist_logi01_plot_01_perp_deta(wha1, df_resu_deta)        
        
        # C(EAP) f(age, mbre) : Analyse
        df_resu_glob, df_resu_deta = mist_logi01_exec12(wha2, df_line, ind2_cate_list, colu_cate_list, ind2_name, colu_name, logi_indx_list)
        # C(EAP) f(age, mbre) : Plots
        # mist_logi01_plot_01_perp_glob(wha1, df_resu_glob) ; mist_logi01_plot_01_perp_deta(wha1, df_resu_deta)
    
    # Version généralisée pour un Logit relatif à tous les C(EAP) avec (age, sexe, mbre)
    # ----------------------------------------------------------------------------------
    if True:
        # C(EAP) f(age, sexe, mbre) : Analyse
        df_resu_glob, df_resu_deta = mist_logi01_exec22(what, df_line, ind1_cate_list, ind2_cate_list, colu_cate_list, ind1_name, ind2_name, colu_name, logi_indx_list)
        # C(EAP) f(age, sexe, mbre) : Plots
        if False:
            mist_logi01_plot_01_perp_glob(wha1, df_resu_glob) # plots 'results globaux' utilisés dan l'étude
            mist_logi01_plot_01_perp_deta(wha1, df_resu_deta) # plots 'results détails' utilisés dan l'étude
        if False:
            mist_logi01_plot_01_mist(wha1, df_line) # plots non utilisés dans l'étude
    pass 
