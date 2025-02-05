import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from scipy.stats import chi2_contingency

# -------------------------------
# Chi2
# -------------------------------
'''
ceap  NA  C0  C1  C2   C3  C4  C5  C6
sexe
M     52  31   5  44   93  38  18  97
F     53  36   6  54  156  59  35  99
- iterate each ceap
- compare its M,F values with the sum of the other M,F values
example :
NA :    This That
sexe
M       52   326
F       53   445 sum:876
'''
def caty_chi2_orig(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    
    # An example
    if False:
        data = {
            'NA': [52, 53],
            'C0': [31, 36],
            'C1': [5, 6],
            'C2': [44, 54],
            'C3': [93, 156],
            'C4': [38, 59],
            'C5': [18, 35],
            'C6': [97, 99]
        }
        df = pd.DataFrame(data, index=['Male', 'Female'])
    
    # ----
    # Prec
    # ----
    indx_cate_nam1 = indx_cate_list[0]
    indx_cate_nam2 = indx_cate_list[1]

    # ----
    # Exec
    # ----
    loop_jrnl = False
    # Perform Z-test for proportions in each age bin
    if loop_jrnl:
        print(f"\nData : {what}\nChi2 2024_12_15 [2025_01_17] : Iterations per {colu_name}")
        write(f"\nData : {what}\nChi2 2024_12_15 [2025_01_17] : Iterations per {colu_name}")
    resu_dict = {}
    alpha = 0.05
    H0 = f"H0 : The proportions in {indx_cate_nam1} and {indx_cate_nam2} are similar for column {colu_name}"
    Ha = f"Ha : The proportions in {indx_cate_nam1} and {indx_cate_nam2} are not similar for column {colu_name}"
    Hx = f"({colu_cate_list}) vs ({indx_cate_list})"
    for ceap_class in df.columns:
        
        # Exec
        # ----
        observed_males = df.loc[indx_cate_nam1, ceap_class]
        observed_females = df.loc[indx_cate_nam2, ceap_class]
        # Create a DataFrame for Current and Other
        this_ = df[ceap_class]
        # Calculate Other counts by summing all classes except the current one
        that_ = df.sum(axis=1) - this_
        # Create a new DataFrame for Current and Other
        df_ceap = pd.DataFrame({
            'This': this_,
            'That': that_
        })   
        if loop_jrnl:
            print(f"{ceap_class} : {df_ceap} sum:{df_ceap.sum().sum()}")
        this_dict = {index: value for index, value in df_ceap['This'].items()}
        that_dict = {index: value for index, value in df_ceap['That'].items()}
        
        # Perform Chi2 Test
        chi2, pval, dof, expected = chi2_contingency(df_ceap)
        stat = chi2   
          
        # Intp
        # ----
        if pval < alpha:
            if loop_jrnl:
                print(f"Chi2 2024_12_15 [2025_01_17] : Reject the null hypothesis:\n{Ha}")
                write(f"Chi2 2024_12_15 [2025_01_17] : Reject the null hypothesis:\n{Ha}")
            HV = "Ha"
            HT = Ha
        else:
            if loop_jrnl:
                print(f"Chi2 2024_12_15 [2025_01_17] : Fail to reject the null hypothesis:\n{H0}")
                write(f"Chi2 2024_12_15 [2025_01_17] : Fail to reject the null hypothesis:\n{H0}")
            HV = "H0"
            HT = H0
        
        # Resu
        # ----
        resu_dict[ceap_class] = {
            colu_name: ceap_class,
            f'{indx_cate_nam1}': observed_males,
            f'{indx_cate_nam2}': observed_females,
            f'{indx_cate_nam1}_thi': this_dict[indx_cate_nam1],
            f'{indx_cate_nam1}_tha': that_dict[indx_cate_nam1],
            f'{indx_cate_nam1}_sum': this_dict[indx_cate_nam1]+that_dict[indx_cate_nam1],
            f'{indx_cate_nam2}_thi': this_dict[indx_cate_nam2],
            f'{indx_cate_nam2}_tha': that_dict[indx_cate_nam2],
            f'{indx_cate_nam2}_sum': this_dict[indx_cate_nam2]+that_dict[indx_cate_nam2],
            f'_sum': this_dict[indx_cate_nam1]+that_dict[indx_cate_nam1]+this_dict[indx_cate_nam2]+that_dict[indx_cate_nam2],
            'stat': stat,
            'pval': pval,
            'H': HV
        }

    df_resu = pd.DataFrame.from_dict(resu_dict, orient='index')
    frmt = lambda value: f"{value:.3e}" if value < 0.001 else f"{value:.3f}"
    df_resu['stat'] = df_resu['stat'].apply(frmt)
    df_resu['pval'] = df_resu['pval'].apply(frmt)
        
    print(f"\n---\nData : {what}\nChi2 2024_12_15 [2025_01_17] :\n{H0}\n{Ha}\n---")
    write(f"\n---\nData : {what}\nChi2 2024_12_15 [2025_01_17] :\n{H0}\n{Ha}\n---")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        print(f"\n{df_resu}")
        write(f"\n{df_resu}")
    print(f"{Hx}")
    write(f"{Hx}")
    pass
def caty_chi2_normOLD(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    
    # Trac
    trac = True
        
    # Prec
    indx_cate_nam1 = indx_cate_list[0]
    indx_cate_nam2 = indx_cate_list[1]
    indx_expe_nam1 = f'{indx_cate_nam1}_expe'
    indx_expe_nam2 = f'{indx_cate_nam2}_expe'
    
    # Observed counts from your table
    observed_data = {
        'CEAP': ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'],
        'Males': [31, 5, 44, 93, 38, 18, 97],
        'Females': [36, 6, 54, 156, 59, 35, 99]
    }
    df1 = pd.DataFrame(observed_data)
    
    print(f"\Input file filtered : df.size:{len(df)} df.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
    write(f"\Input file filtered : df.size:{len(df)} df.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")

    # Exec
    # ----
    if trac:
        dfT = pd.DataFrame({ indx_name: [df.loc[indx_cate_nam1].sum(), df.loc[indx_cate_nam2].sum(), df.loc[indx_cate_nam1].sum()+df.loc[indx_cate_nam2].sum()]}, index=[indx_cate_nam1, indx_cate_nam2, 'T'])
        print(f"\nContingency table  : totals:{dfT.T}")
        write(f"\nContingency table  : totals:{dfT.T}")
    observed_males = df.loc[indx_cate_nam1].sum()
    observed_females = df.loc[indx_cate_nam2].sum()


    #total_males = 378   # Total males in the dataset
    #total_females = 498 # Total females in the dataset
    observed_population = observed_males + observed_females

    # Global sex ratios
    male_ratio = observed_males / observed_population  # ~43%
    female_ratio = observed_females / observed_population  # ~57%

    # Step 1: Compute expected counts based on global proportions
    dfc = pd.DataFrame()
    print(f"\Input file filtered : dfc.size:{len(dfc)} dfc.type:{type(dfc)}\n{dfc}\n:{dfc.index}\n:{dfc.columns}")
    write(f"\Input file filtered : dfc.size:{len(dfc)} dfc.type:{type(dfc)}\n{dfc}\n:{dfc.index}\n:{dfc.columns}")
    dfc[indx_cate_nam1] = df.loc[indx_cate_nam1]
    dfc[indx_cate_nam2] = df.loc[indx_cate_nam2]
    dfc[indx_expe_nam1] = (df.loc[indx_cate_nam1] + df.loc[indx_cate_nam2]) * male_ratio # df['Expected_Males'] = (df['Males'] + df['Females']) * male_ratio
    dfc[indx_expe_nam2] = (df.loc[indx_cate_nam1] + df.loc[indx_cate_nam2]) * female_ratio # df['Expected_Females'] = (df['Males'] + df['Females']) * female_ratio
    print(f"\Input file filtered : dfc.size:{len(dfc)} dfc.type:{type(dfc)}\n{dfc}\n:{dfc.index}\n:{dfc.columns}")
    write(f"\Input file filtered : dfc.size:{len(dfc)} dfc.type:{type(dfc)}\n{dfc}\n:{dfc.index}\n:{dfc.columns}")

    # Step 2: Compare observed vs expected counts
    medi = False
    if medi:
        dfc[f'{indx_cate_nam1}_devi'] = df.loc[indx_cate_nam1] - dfc[indx_expe_nam1] # df['Males'] - df['Expected_males']
        dfc[f'{indx_cate_nam2}_devi'] = df.loc[indx_cate_nam2] - dfc[indx_expe_nam2] # df['Females'] - df['Expected_Females']
        print(f"\Input file filtered : dfc.size:{len(dfc)} dfc.type:{type(dfc)}\n{dfc}\n:{dfc.index}\n:{dfc.columns}")
        write(f"\Input file filtered : dfc.size:{len(dfc)} dfc.type:{type(dfc)}\n{dfc}\n:{dfc.index}\n:{dfc.columns}")
    print(f"\Input file filtered : dfc.size:{len(dfc.T)} dfc.type:{type(dfc.T)}\n{dfc.T}\n:{dfc.T.index}\n:{dfc.T.columns}")
    write(f"\Input file filtered : dfc.size:{len(dfc.T)} dfc.type:{type(dfc.T)}\n{dfc.T}\n:{dfc.T.index}\n:{dfc.T.columns}")
    
    # Step 3: Perform a Chi-square goodness-of-fit test for each CEAP class
    def chi_square_test(row):
        print (row)
        observed = [row.loc[indx_cate_nam1], row.loc[indx_cate_nam2]] # [row['Males'], row['Females']]
        expected = [row.loc[indx_expe_nam1], row.loc[indx_expe_nam2]] # [row['Expected_Males'], row['Expected_Females']]
        print (expected)
        print (observed)
        chi2, pval = chi2_contingency([observed, expected])[:2]
        return pd.Series({'Chi2': chi2, 'P-value': pval})

    tests = dfc.apply(chi_square_test, axis=1)
    dfT = pd.concat([dfc, tests], axis=1)
    print (dfT)
def caty_chi2_norm(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    
    # Trac
    trac = True
        
    # Prec
    indx_cate_nam1 = indx_cate_list[0]
    indx_cate_nam2 = indx_cate_list[1]
    indx_expe_nam1 = f'{indx_cate_nam1}_expe'
    indx_expe_nam2 = f'{indx_cate_nam2}_expe'
    
    # Observed counts from your table
    observed_data = {
        'CEAP': ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'],
        'Males': [31, 5, 44, 93, 38, 18, 97],
        'Females': [36, 6, 54, 156, 59, 35, 99]
    }
    df1 = pd.DataFrame(observed_data)
    
    print(f"\Input file filtered : df.size:{len(df)} df.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
    write(f"\Input file filtered : df.size:{len(df)} df.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")

    # Exec
    # ----
    if trac:
        dfT = pd.DataFrame({ indx_name: [df.loc[indx_cate_nam1].sum(), df.loc[indx_cate_nam2].sum(), df.loc[indx_cate_nam1].sum()+df.loc[indx_cate_nam2].sum()]}, index=[indx_cate_nam1, indx_cate_nam2, 'T'])
        print(f"\nContingency table  : totals:{dfT.T}")
        write(f"\nContingency table  : totals:{dfT.T}")
    observed_males = df.loc[indx_cate_nam1].sum()
    observed_females = df.loc[indx_cate_nam2].sum()


    #total_males = 378   # Total males in the dataset
    #total_females = 498 # Total females in the dataset
    observed_population = observed_males + observed_females

    # Global sex ratios
    male_ratio = observed_males / observed_population  # ~43%
    female_ratio = observed_females / observed_population  # ~57%

    # Step 1: Compute expected counts based on global proportions
    dfc = pd.DataFrame()
    print(f"\Input file filtered : dfc.size:{len(dfc)} dfc.type:{type(dfc)}\n{dfc}\n:{dfc.index}\n:{dfc.columns}")
    write(f"\Input file filtered : dfc.size:{len(dfc)} dfc.type:{type(dfc)}\n{dfc}\n:{dfc.index}\n:{dfc.columns}")
    dfc[indx_cate_nam1] = df.loc[indx_cate_nam1]
    dfc[indx_cate_nam2] = df.loc[indx_cate_nam2]
    dfc[indx_expe_nam1] = (df.loc[indx_cate_nam1] + df.loc[indx_cate_nam2]) * male_ratio # df['Expected_Males'] = (df['Males'] + df['Females']) * male_ratio
    dfc[indx_expe_nam2] = (df.loc[indx_cate_nam1] + df.loc[indx_cate_nam2]) * female_ratio # df['Expected_Females'] = (df['Males'] + df['Females']) * female_ratio
    print(f"\Input file filtered : dfc.size:{len(dfc)} dfc.type:{type(dfc)}\n{dfc}\n:{dfc.index}\n:{dfc.columns}")
    write(f"\Input file filtered : dfc.size:{len(dfc)} dfc.type:{type(dfc)}\n{dfc}\n:{dfc.index}\n:{dfc.columns}")

    # Step 2: Compare observed vs expected counts
    medi = False
    if medi:
        dfc[f'{indx_cate_nam1}_devi'] = df.loc[indx_cate_nam1] - dfc[indx_expe_nam1] # df['Males'] - df['Expected_males']
        dfc[f'{indx_cate_nam2}_devi'] = df.loc[indx_cate_nam2] - dfc[indx_expe_nam2] # df['Females'] - df['Expected_Females']
        print(f"\Input file filtered : dfc.size:{len(dfc)} dfc.type:{type(dfc)}\n{dfc}\n:{dfc.index}\n:{dfc.columns}")
        write(f"\Input file filtered : dfc.size:{len(dfc)} dfc.type:{type(dfc)}\n{dfc}\n:{dfc.index}\n:{dfc.columns}")
    print(f"\Input file filtered : dfc.size:{len(dfc.T)} dfc.type:{type(dfc.T)}\n{dfc.T}\n:{dfc.T.index}\n:{dfc.T.columns}")
    write(f"\Input file filtered : dfc.size:{len(dfc.T)} dfc.type:{type(dfc.T)}\n{dfc.T}\n:{dfc.T.index}\n:{dfc.T.columns}")
    
    # Step 3: Perform a Chi-square goodness-of-fit test for each CEAP class
    def chi_square_test(row):
        print (row)
        observed = [row.loc[indx_cate_nam1], row.loc[indx_cate_nam2]] # [row['Males'], row['Females']]
        expected = [row.loc[indx_expe_nam1], row.loc[indx_expe_nam2]] # [row['Expected_Males'], row['Expected_Females']]
        print (expected)
        print (observed)
        chi2, pval = chi2_contingency([observed, expected])[:2]
        return pd.Series({'Chi2': chi2, 'P-value': pval})

    tests = dfc.apply(chi_square_test, axis=1)
    dfT = pd.concat([dfc, tests], axis=1)
    print (dfT)
    # Display results
    pass
def caty_chi2(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name):
    caty_chi2_orig(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name)
    #caty_chi2_norm(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name)
