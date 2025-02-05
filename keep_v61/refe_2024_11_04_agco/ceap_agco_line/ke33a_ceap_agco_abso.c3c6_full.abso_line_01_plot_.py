from datetime import datetime
import os
import sys
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

from util_file_inpu_ceap import inp1
from util_file_mngr import set_file_objc, write
        
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import permutation_test
from scipy.stats import spearmanr

def inpu(file_path):
        
    # ----
    # Exec
    # ----
    df1, df2, df3 = inp1(file_path)

    # ----
    # Exit
    # ----
    return df1, df2, df3

def spea_mann_1(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list):
    
    trac = True

    # Prec
    # ----
    sexM = parm_sexe_list[0]
    sexF = parm_sexe_list[1]
        
    # ----
    # Exec
    # ----
    df = df_line  
    # Remove 'NA' if it's the first element
    if False:
        if parm_ceap_list and parm_ceap_list[0] == 'NA':
            parm_ceap_list = parm_ceap_list[1:]   
    # Create numeric equivalent
    parm_ceap_dict = {value: index for index, value in enumerate(parm_ceap_list)}
    df['ceap_nume'] = df['ceap'].map(parm_ceap_dict)
    df['age_rank'] = df['age'].rank() # not used but could be
    print(f"\nStep 0 : df.size:{len(df)} df2.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")

    if False:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='ceap', y='age', data=df)
        plt.title('Box Plot of Age by CEAP Category')
        plt.xlabel('CEAP Category')
        plt.ylabel('Age')
        plt.show()
    if False:
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='ceap', y='age', data=df)
        plt.title('Violin Plot of Age by CEAP Category')
        plt.xlabel('CEAP Category')
        plt.ylabel('Age')
        plt.show()
    if False:
        plt.figure(figsize=(10, 6))
        sns.stripplot(x='ceap', y='age', data=df, jitter=True)
        plt.title('Scatter Plot with Jitter of Age by CEAP Category')
        plt.xlabel('CEAP Category')
        plt.ylabel('Age')
        plt.show()
    if False:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='ceap', y='age', data=df, ci='sd')
        plt.title('Bar Plot of Mean Age by CEAP Category with Error Bars')
        plt.xlabel('CEAP Category')
        plt.ylabel('Mean Age')
        plt.show()

    pass


def spea_mann_2(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list):
    
    trac = True

    # Prec
    # ----
    sexM = parm_sexe_list[0]
    sexF = parm_sexe_list[1]
        
    # ----
    # Exec
    # ----
    df = df_line  
    # Remove 'NA' if it's the first element
    if False:
        if parm_ceap_list and parm_ceap_list[0] == 'NA':
            parm_ceap_list = parm_ceap_list[1:]   
    # Create numeric equivalent
    parm_ceap_dict = {value: index for index, value in enumerate(parm_ceap_list)}
    df['ceap_nume'] = df['ceap'].map(parm_ceap_dict)
    df['age_rank'] = df['age'].rank() # not used but could be
    print(f"\nStep 0 : df.size:{len(df)} df2.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
    
    # Convert 'ceap' to numeric values for plotting
    ceap_mapping = {'C0': 1, 'C1': 2, 'C2': 3, 'C3': 4, 'C4': 5, 'C5': 6, 'C6': 7}
    df['ceap_numeric'] = df['ceap'].map(ceap_mapping)
    # df = df[df[parm_sexe] == 'F']

    # Calculate Spearman's rho
    rho, p_value = spearmanr(df['age'], df['ceap_numeric'])

    plt.figure(figsize=(10, 6))
    sns.regplot(x='ceap_numeric', y='age', data=df, scatter_kws={'s':10}, line_kws={"color":"red"})
    plt.title('Scatter Plot with Regression Line of Age by CEAP Category')
    plt.xlabel('CEAP Category (Numeric)')
    plt.ylabel('Age')
    plt.annotate(f'Spearman\'s rho = {rho:.3f}\np-value = {p_value:.3f}',
                xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                horizontalalignment='left', verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="wheat", alpha=0.5))
    plt.show()
    pass
    
def stat_oper_NA_NOT(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list):
    spea_mann_1(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)
    spea_mann_2(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)

def stat_exec_NA_NOT(df_line):
    parm_ceap = 'ceap'
    parm_sexe = 'sexe'
    parm_agee = 'age'
    parm_ceap_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'] 
    parm_sexe_list = ['M','F']
    what = f"'{parm_ceap}' '{parm_agee}' ; {parm_sexe} {parm_sexe_list}"
    stat_oper_NA_NOT(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)
        
    parm_ceap = 'ceap'
    parm_sexe = 'mbre'
    parm_agee = 'age'
    parm_ceap_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'] 
    parm_sexe_list = ['G','D']
    what = f"'{parm_ceap}' '{parm_agee}' ; {parm_sexe} {parm_sexe_list}"
    stat_oper_NA_NOT(what, df_line, parm_ceap, parm_agee, parm_sexe, parm_ceap_list, parm_sexe_list)
        
if __name__ == "__main__":

    # Step 1
    exit_code = 0           
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    script_name = os.path.basename(__file__)
    print (f"len(sys.argv): {len(sys.argv)}")
    print (f"sys.argv: {sys.argv}")
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
    else:
        file_path = script_dir
    
    # Step 2
    suppress_suffix = ".py"
    script_name = script_name[:-len(suppress_suffix)]
    jrnl_file_path = os.path.join(script_dir, f"{script_name}jrnl.txt")
    with open(jrnl_file_path, 'w') as file:
        
        set_file_objc(file)
        date_curr = datetime.now()
        date_form = date_curr.strftime('%Y-%m-%d %H:%M:%S')
        write (">>> >>> >>>")
        write (date_form)
        write (">>> >>> >>>")
        
        # Inpu
        df1, df2, df3 = inpu(file_path)
        # Exec NA_NOT
        exec_NA_NOT = True
        if exec_NA_NOT:
            write ("")
            write (">>> >>> >>>")
            write (f'{date_form} NA_NOT')
            write (">>> >>> >>>")
            df_line = df2
            trac = True
            if trac:
                print(f"\nInput file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
                write(f"\nInput file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
            stat_exec_NA_NOT(df_line)
        pass 