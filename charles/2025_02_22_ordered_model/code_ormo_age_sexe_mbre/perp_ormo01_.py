from datetime import datetime
import os
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
from util_file_mngr import write
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
import scipy.stats as scs
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt


def perp_ormo01_exec11(what, df_line, logi_indx_list):
    
    # Prec
    # ----
    df_ = df_line[['sexe', 'ceap']]
    print (df_)
    
    # Data
    # ----
    # Exclude 'NA' values and convert CEAP to numeric
    # df = df[df['ceap'] != 'NA'].copy()
    df = df_.copy()
    df['ceap_nume'] = pd.Categorical(df['ceap'], ordered=True, categories=['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']).codes
    df['sexe_nume'] = df['sexe'].map({'M': logi_indx_list[1], 'F': logi_indx_list[0]}) # df['sexe_nume'] = pd.Categorical(df['sexe']).codes
    print (df)
    df = df[['sexe_nume', 'ceap_nume']]
    df = df.rename(columns={'ceap_nume': 'ceap'})
    df = df.rename(columns={'sexe_nume': 'sexe'})

    # Exec
    # ----
    model = OrderedModel(df['ceap'], df[['sexe']], distr='logit') # Fit the ordinal logistic regression model
    result = model.fit()
    print(f"\n---\nData : {what}\nLogistic Regression Perplexity [2025_02_12] : y='ceap'='C3' ; x='sexe','age'\n---")
    write(f"\n---\nData : {what}\nLogistic Regression Perplexity [2025_02_12] : y='ceap'='C3' ; x='sexe','age'\n---")
    print(result.summary())
    write(result.summary().as_text())
    pass

def ormo(what, df, colu_cate_list, x_list):
    
    # Data
    # ----
    # Given list to order the column
    cate_list = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    # Convert the column to a categorical type with the specified order
    df['ceap'] = pd.Categorical(df['ceap'], categories=cate_list, ordered=True)
    # Sort the DataFrame based on the 'ceap' column
    df = df.sort_values('ceap')
    # Map 'ceap' to numerical codes
    df['ceap_nume'] = df['ceap'].cat.codes
    #
    df = df[x_list + ['ceap_nume']]
    df = df.rename(columns={'ceap_nume': 'ceap'})
    # df = df.sort_values(by=['ceap', 'mbre', 'sexe', 'age'], ascending=[True, False, False, True])
    print (df)
    
    # Exec
    # ----
    model = OrderedModel(df['ceap'], df[x_list], distr='logit') # Fit the ordinal logistic regression model
    result = model.fit(maxiter=2000)
    print(f"\n---\nData : {what}\nLogistic Regression Perplexity [2025_02_12] : y={colu_cate_list} ; x='{x_list}'\n---")
    write(f"\n---\nData : {what}\nLogistic Regression Perplexity [2025_02_12] : y={colu_cate_list} ; x='{x_list}'\n---")
    resu_dict_deta = {}
    resu_dict_glob = {}
    ceap = "ceap"
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
    df_resu_deta = df_resu_deta.round({
        'coef': 3,
        'std_err': 3,
        'z': 3,
        'P>|z|': 3,
        '[0.025': 3,
        '0.975]': 3,
        'odds_ratio': 3,
        'ci_lower': 3,
        'ci_upper': 3,
        'odds_ratio_neg': 3
    })
    #
    cols = list(df_resu_deta.columns)
    df_resu_deta['pval'] = df_resu_deta.apply(lambda row: row['P>|z|'] if row['index'] not in ['-', 'const'] and pd.to_numeric(row['P>|z|'], errors='coerce') <= 0.05 else '', axis=1)
    pval_index = cols.index('P>|z|') + 1
    cols.insert(pval_index, cols.pop())
    #
    # df_resu_deta['accs'] = df_resu_deta['ceap'].astype(str) + '_' + df_resu_deta['index'].astype(str)
    # df_resu_deta['accs'] = df_resu_deta.apply( lambda row: row['ceap'] + '_' + str(row['index']) if '/' in row['index'] else row['index'], axis=1 )
    replacements = {
        '0/1': 'NA/C0',
        '1/2': 'C0/C1',
        '2/3': 'C1/C2',
        '3/4': 'C2/C3',
        '4/5': 'C3/C4',
        '5/6': 'C4/C5',
        '6/7': 'C5/C6'
    }

    # Replace values in the 'index' column
    df_resu_deta['accs'] = df_resu_deta['index'].replace(replacements)
    print (df_resu_deta)
    df_resu_deta.set_index('accs', drop=False, inplace=True) # drop column ; inplace ie same df
    df_resu_deta = df_resu_deta[['ceap', 'index', 'coef', 'std err', 'z', 'P>|z|', '[0.025', '0.975]', 'odds_ratio', 'ci_lower', 'ci_upper', 'odds_ratio_neg']]
    df_resu_deta = df_resu_deta.rename(columns={'ceap': 'ceap=y(dependant)', 'index': 'coef=x(independant)'})
    
    # Exit
    # ----
    return df_resu_glob, df_resu_deta

def ormo_plot_01(df, suff, date_time):
    
    '''
    To qualify the odds_ratio values with respect to the independent variables (sexe, mbre, and age), you can analyze how these variables influence the odds ratios. Here's a step-by-step approach to interpret the results:

    Understanding Odds Ratios:

    An odds ratio (OR) greater than 1 indicates an increased likelihood of the event occurring compared to the reference group.
    An OR less than 1 indicates a decreased likelihood.
    An OR of 1 suggests no effect.
    Interpreting Independent Variables:

    Sexe: If sexe = 1 (Male) has an OR > 1, it means males have a higher likelihood of the event compared to females (sexe = 0).
    Mbre: If mbre = 1 (e.g., Right laterality) has an OR > 1, it means the right side has a higher likelihood compared to the left side (mbre = 0).
    Age: As a continuous variable, an OR > 1 for age indicates that as age increases, the likelihood of the event increases.
    Qualifying Odds Ratios:

    For sexe, mbre, and age, you can directly interpret the OR values from the results DataFrame.
    For transitions (e.g., 0/1, 1/2), you can compare the OR to 1 to determine if the transition increases or decreases the likelihood of the event.
    Visualization:

    The plot you've created visualizes the ORs for CEAP transitions with 95% confidence intervals.
    The horizontal line at y=1 represents "No Effect." ORs above this line indicate an increased likelihood, and those below indicate a decreased likelihood.
    '''
    
    df = df[~df.index.str.contains('-')]
    print (df, df.index)
    # Extract only CEAP transitions
    transitions = df.index # ['ceap_sexe', 'ceap_mbre', 'ceap_age', 'ceap_0/1', 'ceap_1/2', 'ceap_2/3', 'ceap_3/4', 'ceap_4/5', 'ceap_5/6', 'ceap_6/7']
    df_transitions = df # df.loc[transitions]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.errorbar(df_transitions.index, df_transitions['odds_ratio'], 
                yerr=[df_transitions['odds_ratio'] - df_transitions['ci_lower'], 
                    df_transitions['ci_upper'] - df_transitions['odds_ratio']], 
                fmt='o', capsize=5, label="Odds Ratio", linewidth=1, color='black')

    # Labels & Styling
    plt.axhline(y=1, color='gray', linestyle='--', label="No Effect (OR=1)")
    plt.ylabel("Odds Ratio (log scale)")
    plt.yscale("log")  # Log scale for better visualization
    plt.xlabel("CEAP Transitions")
    plt.title("Odds Ratios for CEAP Transitions (with 95% CI)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Add spacing on y-axis
    y_min, y_max = plt.ylim()
    plt.ylim(y_min * 0.5, y_max * 1.5)

    # Add spacing on x-axis
    plt.xlim(-0.5, len(transitions) - 0.5)

    # Annotate the plot
    for i, transition in enumerate(transitions):
        or_value = df_transitions.loc[transition, 'odds_ratio']
        ci_upper = df_transitions.loc[transition, 'ci_upper']
        ci_lower = df_transitions.loc[transition, 'ci_lower']
        if or_value > 1:
            plt.text(i, ci_upper + (y_max * 1.5 - y_max) * 0.05, f'Increased\nOR={or_value:.3f}', ha='center', va='bottom', color='green')
        elif or_value < 1:
            plt.text(i, ci_lower - (y_min * 1.5 - y_min) * 0.1, f'Decreased\nOR={or_value:.3f}', ha='center', va='top', color='red')

    # Add gray background for vertical areas
    for i in range(len(transitions)):
        if i % 2 == 0:  # Apply gray background to every other area
            plt.axvspan(i - 0.5, i + 0.5, color='lightgray', alpha=0.5)
            
    # Position the legend below the plot, aligned to the right
    plt.legend(loc="lower right")

    # Output
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    parent_dir = os.path.dirname(script_dir)
    what = f'{suff} [plot 1]'
    plt.figtext(0.98, 0.02, f'{suff}', horizontalalignment='right', fontsize=8)
    plt.tight_layout()  # Adjust layout to prevent overlap
    os.makedirs(f'{parent_dir}\\plot_results {date_time}', exist_ok=True)
    file_path = os.path.join(script_dir, f'{parent_dir}\\plot_results {date_time}\\{what}.pdf')
 
    # plt.show()
    plt.savefig(file_path)
    pass

def ormo_plot_02(df, suff,date_time):
    df = df[~df.index.str.contains('-')]
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    # Plot coefficients and confidence intervals
    df['coef'].plot(kind='bar', yerr=df['std err'], ax=ax1, capsize=5)
    ax1.set_title('Coefficients with 95% Confidence Intervals')
    ax1.set_ylabel('Coefficient Value')
    ax1.axhline(y=0, color='r', linestyle='--')

    # Plot odds ratios and confidence intervals
    df['odds_ratio'].plot(kind='bar', yerr=[df['odds_ratio'] - df['ci_lower'], 
                                                df['ci_upper'] - df['odds_ratio']], 
                            ax=ax2, capsize=5, log=True)
    ax2.set_title('Odds Ratios with 95% Confidence Intervals')
    ax2.set_ylabel('Odds Ratio (log scale)')
    ax2.axhline(y=1, color='r', linestyle='--')

    # Rotate x-axis labels for both plots
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

    # Output
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    parent_dir = os.path.dirname(script_dir)
    what = f'{suff} [plot 2]'
    plt.figtext(0.98, 0.02, f'{suff}', horizontalalignment='right', fontsize=8)
    plt.tight_layout()  # Adjust layout to prevent overlap
    os.makedirs(f'{parent_dir}\\plot_results {date_time}', exist_ok=True)
    file_path = os.path.join(script_dir, f'{parent_dir}\\plot_results {date_time}\\{what}.pdf')
 
    # plt.show()
    plt.savefig(file_path)
    pass

def perp_ormo01_exec12(what, df_line, ind1_cate_list, colu_cate_list, ind1_name, colu_name, logi_indx_list):
   
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
    df_resu_glob, df_resu_deta = ormo(what, df, colu_cate_list, x_list)
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

def perp_ormo01_exec22(what, df_line, ind1_cate_list, ind2_cate_list, colu_cate_list, ind1_name, ind2_name, colu_name, logi_indx_list):
   
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
    df_resu_glob, df_resu_deta = ormo(what, df, colu_cate_list, x_list)
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
# Note : les plots sous forme fichier sont créés
# ----
def perp_ormo01_exec(what, df_line, ind1_cate_list, ind2_cate_list, colu_cate_list, ind1_name, ind2_name, colu_name):
    
    date_curr = datetime.now()
    date_time = date_curr.strftime('%Y_%m_%d %H_%M_%S')
    
    logi_indx_list = [1,0] ; info = "version M=1,F=0 ; G=1,D=0" # version M=1,F=0 ; G=1,D=0 [préférée pour représentations graphiques]
    date_tim1 = f'{date_time} {info}'
    perp_ormo01_perf(what, df_line, ind1_cate_list, ind2_cate_list, colu_cate_list, ind1_name, ind2_name, colu_name, logi_indx_list, info, date_tim1)
    logi_indx_list = [0,1] ; info = "version M=0,F=1 ; G=0,D=1" # version M=0,F=1 ; G=0,D=1
    date_tim2 = f'{date_time} {info}'
    perp_ormo01_perf(what, df_line, ind1_cate_list, ind2_cate_list, colu_cate_list, ind1_name, ind2_name, colu_name, logi_indx_list, info, date_tim2)
    pass

def perp_ormo01_perf(what, df_line, ind1_cate_list, ind2_cate_list, colu_cate_list, ind1_name, ind2_name, colu_name, logi_indx_list, info, date_time):

    print (df_line)

    wha1 = f"'{ind1_name}' '{colu_name}' ; {ind1_cate_list} {colu_cate_list}"
    wha2 = f"'{ind2_name}' '{colu_name}' ; {ind2_cate_list} {colu_cate_list}"
  
    # Version prototype : 'ceap','sexe'
    # ---------------------------------
    if True:
        # C(EAP) f(ceap, sexe) : Analyse
        perp_ormo01_exec11(wha1, df_line, logi_indx_list)
    
    # Version généralisée pour un Logit relatif à tous les C(EAP) avec (age, sexe) OU (age,mbre)
    # ------------------------------------------------------------------------------------------
    if True:        
        # C(EAP) f(age, sexe) : Analyse
        df_resu_glob, df_resu_deta = perp_ormo01_exec12(wha1, df_line, ind1_cate_list, colu_cate_list, ind1_name, colu_name, logi_indx_list)
        # C(EAP) f(age, sexe) : Plots
        suff = f'C(EAP) f(age, sexe) {info}'
        df_plot_deta = df_resu_deta.copy()
        ormo_plot_01(df_plot_deta, suff, date_time)
        ormo_plot_02(df_plot_deta, suff, date_time)       
        
        # C(EAP) f(age, mbre) : Analyse
        df_resu_glob, df_resu_deta = perp_ormo01_exec12(wha2, df_line, ind2_cate_list, colu_cate_list, ind2_name, colu_name, logi_indx_list)
        # C(EAP) f(age, mbre) : Plot
        suff = f'C(EAP) f(age, mbre) {info}'
        df_plot_deta = df_resu_deta.copy()
        ormo_plot_01(df_plot_deta, suff, date_time)
        ormo_plot_02(df_plot_deta, suff, date_time) 
        pass
    
    # Version généralisée pour un Logit relatif à tous les C(EAP) avec (age, sexe, mbre)
    # ----------------------------------------------------------------------------------
    if True:
        # C(EAP) f(age, sexe, mbre) : Analyse
        df_resu_glob, df_resu_deta = perp_ormo01_exec22(what, df_line, ind1_cate_list, ind2_cate_list, colu_cate_list, ind1_name, ind2_name, colu_name, logi_indx_list)
        # C(EAP) f(age, sexe, mbre) : Plots
        if True:
            suff = f'C(EAP) f(age, sexe, mbre) {info}'
            df_plot_deta = df_resu_deta.copy()
            ormo_plot_01(df_plot_deta, suff, date_time)
            ormo_plot_02(df_plot_deta, suff, date_time)
    pass 
