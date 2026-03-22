import os
import sys
import pandas as pd
from pathlib import Path
import re
from pprint import pprint
from pandas.api.types import is_numeric_dtype

# ****
# Clas
# ****

# ----
# File
# ----
class InpuTran():
    def __init__(self, procTran):
        super().__init__()
        procTran.dict[InpuTran.__name__] = self
        self.filt = None
        self.func = None
        self.fram = None
        self.file = None

    def upda(self):
        _upda_file(self)

def _upda_file(inpuTran:InpuTran): 
     
    # Opti
    # ----
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 2000) 
    
    # Inpu
    # ----
    inpu_file = inpuTran.file
    scri_path = os.path.abspath(__file__)
    scri_dire = os.path.dirname(scri_path)
    path_inpu = os.path.join(scri_dire, inpu_file)
    path_inpu = Path(path_inpu).resolve()
    path_inpu = os.path.normpath(path_inpu)
    
    # Exec
    # ----
    filt_list = inpuTran.filt
    inpu_func = inpuTran.func
    df_inpu = inpu_func(path_inpu, filt_list)
    inpuTran.fram = df_inpu
    pass

def inpu_file_exec_selc_0(path_inpu, filt_list):
    
    trac = True

    # Exec
    # ----
    print (path_inpu)
    df_full = pd.read_csv(path_inpu, delimiter="|", na_values=[], keep_default_na=False)
    df_full.columns = df_full.columns.str.strip()
        
    # Trac
    # ----
    if trac:
        print_yes(df_full, labl="df_full")
           
    # Exit
    # ----
    return df_full

def inpu_file_exec_xlat_0(df_fram, what, cate):
 
    trac = True
        
    # Exec
    # ----
    df_fram[what] = pd.Categorical(df_fram[what], categories=cate, ordered=True)
        
    # Trac
    # ----
    if trac:
        print_yes(df_fram, labl="df_fram")
    
    # Exit
    # ----
    return df_fram

def inpu_file_exec_xlat_1(df_fram, what, text):
   
    trac = True
        
    # Trac
    # ----
    if trac:
        print_yes(df_fram, labl="df_fram")

    # Exec
    # ----
    matches = re.findall(r"([\w\s,-]+)\((\d+)\)", text)
    what_dict = {int(code): value.strip() for value, code in matches}
    what_list = [value.strip() for value, _ in matches]
    df_fram[what] = pd.to_numeric(df_fram[what], errors='coerce')
    df_fram[what] = (
        df_fram[what]
        .map(what_dict)
        .astype(pd.CategoricalDtype(categories=what_list, ordered=True))
    )
    
    # Trac
    # ----
    if trac:
        print_yes(df_fram, labl="df_fram")
        
    # Exit
    # ----
    return df_fram

def inpu_file_exec_xlat_2(df_fram, what, text):
   
    trac = True
    
    # ====
    # Process 'text'
    # ====
        
    # Exec
    # ----
    def preprocess_multi_index(text):
        
        # Step 1 : This regex finds the label and the multi-index brackets, e.g., "Label [1, 2]"
        # ----
        # Group 1: The label text
        # Group 2: The comma-separated numbers
        pattern = r"([^\]]+?)\s*\[([\d\s,]+)\]"
        
        def expand_match(match):
            label = match.group(1).strip()
            indices = match.group(2).split(',')
            # Create a new string: "Label [1] Label [2]"
            return " ".join([f"{label} [{idx.strip()}]" for idx in indices])

        # Apply the expansion to the whole text
        text = re.sub(pattern, expand_match, text)
    
        # Step 2: emove any whitespace immediately before or after the square brackets
        # ----
        text = re.sub(r'\s*(\[\d+\])\s*', r'\1', text)
        # print(text)
        
        # Exit
        # ----
        return text

    # Testing it out:
    # text = "Local: Kinshasa [1] National: Rest of DR Congo [2] International: Africa & World [3, 4]"
    text = preprocess_multi_index(text)
        
    # Prec : all numbers in [] should be sequential
    # ----
    found = {int(num) for num in re.findall(r'\[(\d+)\]', text)}
    if not found:
        raise Exception(f"Series is not incremental or complete")
    expected = set(range(1, len(found) + 1))
    if found != expected:
        missing = expected - found
        raise Exception(f"Series is not incremental or complete. Missing: {missing}")
       
    # ====
    # Process 'fram'
    # ====
    
    # Data
    # ----
    cols_base = ["workbook", "patient_id", "timepoint"]
    cols_vari = [what]
    cols_full = cols_base + cols_vari
    df_work = df_fram[cols_full].copy()
        
    # Trac
    # ----
    if trac:
        print_yes(df_work, labl="df_work")

    # Exec
    # ----
    if not is_numeric_dtype(df_work[what]):
        df_work[what] = df_work[what].str.split(',') # Split the comma-separated string into a list
        df_work = df_work.explode(what).reset_index(drop=True) # Explode into multiple rows
        df_work[what] = pd.to_numeric(df_work[what], errors='coerce')
    
    # Exec
    # ----
    # This regex looks for:
    # (.+?) : Any characters (non-greedy) - this captures the text label
    # \s* : Optional whitespace
    # \[    : A literal opening square bracket
    # (\d+) : One or more digits - this captures the ID
    # \]    : A literal closing square bracket
    matches = re.findall(r"(.+?)\s*\[(\d+)\]", text)
    #matches = re.findall(r"([\w\s,-]+)\((\d+)\)", text)
    what_dict = {int(code): value.strip() for value, code in matches}
    # Extract values and remove duplicates while preserving order
    raw_list = [value.strip() for value, _ in matches]
    what_list = list(dict.fromkeys(raw_list)) 
    if trac:
        pprint(what_dict, width=70)
        pprint(what_list, width=70)
    # Now this will work without the ValueError!
    df_work[what] = (
        df_work[what]
        .map(what_dict)
        .astype(pd.CategoricalDtype(categories=what_list, ordered=True))
    )
    #
    df_work = df_work.rename(columns={what: "Resp"})
        
    # Trac
    # ----
    if trac:
        print_yes(df_work, labl="df_work")
        
    # Exit
    # ----
    return df_work

def inpu_file_exec_xlat_3(df_fram, what):
   
    trac = True
    
    # Data
    # ----
    cols_base = ["workbook", "patient_id", "timepoint"]
    cols_vari = [what]
    cols_full = cols_base + cols_vari
    df_work = df_fram[cols_full].copy()
        
    # Trac
    # ----
    if trac:
        print_yes(df_fram, labl="df_fram")

    # Exec
    # ----
    unique_communes = sorted(df_work[what].unique())
    df_work[what] = pd.Categorical(df_work[what], categories=unique_communes, ordered=True)
    #
    df_work = df_work.rename(columns={what: "Resp"})
    
    # Trac
    # ----
    if trac:
        print_yes(df_work, labl="df_fram")
        
    # Exit
    # ----
    return df_work

def print_yes(df, labl=None):
    print (f"\n----\nFram labl : {labl}\n----")
    with pd.option_context(
            'display.max_columns', None,       # Show all columns
            'display.max_rows', None,          # Show more rows before truncating
            'display.max_colwidth', None,      # Don't cut off long text in 'info'
            'display.width', 1000,             # Prevent the table from wrapping to a new line
            'display.precision', 2,            # Round floats to 2 decimal places
            'display.colheader_justify', 'left' # Align headers for better readability
        ):
        print(f"df:{len(df)} type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
        print(df.info())
    pass