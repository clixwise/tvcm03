import os
import sys
import pandas as pd
import ast

from util_file_mngr import set_file_objc, write

def oupu(df, file_path):
   
    # ----
    # 1:Inpu
    # ----
    file_oupu = "./ke45_ceap_sexe_agco_abso.c3c6_full.abso.oupu.csv"
    path_oupu = os.path.join(file_path, file_oupu)
    df.to_csv(path_oupu)
def inpu(file_path):
   
    # ----
    # 1:Inpu
    # ----
    file_inpu = "./ke45_ceap_sexe_agco_abso.c3c6_full.abso.csv"
    path_inpu = os.path.join(file_path, file_inpu)
    
    # 1. Read the CSV and create initial DataFrame
    df = pd.read_csv(path_inpu, na_filter=False)
    df = df.drop('mean', axis=1)
    print (df)
    df['ages'] = df['ages'].apply(ast.literal_eval)
   
    # Display the resulting DataFrame
    print(f"\nStep 1 : df.size:{len(df)} df.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
    write(f"\nStep 1 : df.size:{len(df)} df.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
    
    return df

def grop(df):
       
    # Group by the 'ceap' column and concatenate the 'ages' lists
    grouped = df.groupby('ceap')['ages'].apply(lambda x: [item for sublist in x for item in sublist])
    # Create a new DataFrame with the concatenated lists
    df1 = grouped.reset_index()
    # Display the resulting DataFrame
    print(f"\nStep 1 : df.size:{len(df1)} df.type:{type(df1)}\n{df1}\n:{df1.index}\n:{df1.columns}")
    write(f"\nStep 1 : df.size:{len(df1)} df.type:{type(df1)}\n{df1}\n:{df1.index}\n:{df1.columns}")
    
    return df1

def xfrm_sexe_yes(df):

    def get_stats(ages):
        series = pd.Series(ages)
        stats = series.describe()
        return pd.Series({
            'count': stats['count'],
            'mean': stats['mean'],
            'std': stats['std'],
            'min': stats['min'],
            '25%': stats['25%'],
            '50%': stats['50%'],
            '75%': stats['75%'],
            'max': stats['max']
        })

    # Apply the function to each row and create new columns
    desc_list = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    df[desc_list] = df['ages'].apply(get_stats)
    df[desc_list] = df[desc_list].round().astype(int)

    # Create a multi-index using 'ceap' and 'sexe'
    df = df.drop('ages', axis=1, errors='ignore')
    df.set_index(['ceap', 'sexe'], inplace=True)

    # Display the resulting DataFrame
    print(f"\nStep 1 : df.size:{len(df)} df.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns} Ceap.coun:{df['count'].sum()}")
    write(f"\nStep 1 : df.size:{len(df)} df.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns} Ceap.coun:{df['count'].sum()}")
    
    return df

def xfrm_sexe_not(df):

    def get_stats(ages):
        series = pd.Series(ages)
        stats = series.describe()
        return pd.Series({
            'count': stats['count'],
            'mean': stats['mean'],
            'std': stats['std'],
            'min': stats['min'],
            '25%': stats['25%'],
            '50%': stats['50%'],
            '75%': stats['75%'],
            'max': stats['max']
        })

    # Apply the function to each row and create new columns
    desc_list = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    df[desc_list] = df['ages'].apply(get_stats)
    df[desc_list] = df[desc_list].round().astype(int)

    # Create a multi-index using 'ceap' and 'sexe'
    df = df.drop('ages', axis=1, errors='ignore')
    df.set_index(['ceap'], inplace=True)

    # Display the resulting DataFrame
    print(f"\nStep 1 : df.size:{len(df)} df.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns} Ceap.coun:{df['count'].sum()}")
    write(f"\nStep 1 : df.size:{len(df)} df.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns} Ceap.coun:{df['count'].sum()}")
    
    return df

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
    suppress_suffix = "_main.py"
    script_name = script_name[:-len(suppress_suffix)]
    jrnl_file_path = os.path.join(script_dir, f"{script_name}_jrnl.txt")
    with open(jrnl_file_path, 'w') as file:
        print(file)
        set_file_objc(file)
        
        # Step 21
        df0 = inpu(file_path)
        dfa = xfrm_sexe_yes(df0)
        
        dfb = grop(df0)
        dfc = xfrm_sexe_not(dfb)
        
        oupu(dfa, file_path)
        pass