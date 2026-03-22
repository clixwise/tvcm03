import numpy as np
import pandas as pd

# ****
# Help
# ****

def fram_prnt(df, labl=None, trunc= None, head=None):
    
    # Exec
    # ----
    print (f"\n----\nFram labl : {labl}\n----")
    
    # Exec
    # ----
    dp = df.head(head) if head is not None else df
    
    # Exec
    # ----
    # 1. Define your base options
    config = {
        'display.max_columns': None, # Show all columns
        'display.max_colwidth': None, # Don't cut off long text in 'info'
        'display.width': 1000, # Prevent the table from wrapping to a new line
        'display.precision': 2, # Round floats to 2 decimal places
        'display.colheader_justify': 'left' # Align headers for better readability
    }
    # 2. Conditionally add the row setting
    if not trunc:
        config['display.max_rows'] = None# Show more rows before truncating
    # 3. Use the unpacked list in the context manager
    options = [item for pair in config.items() for item in pair]
    with pd.option_context(*options):
        print(f"df:{len(df)} type:{type(df)}\n{dp}\n:{df.index}\n:{df.columns}")
        
    # Exec
    # ----
    print(df.info())
    
    # Exec : Summary of all columns and their specific underlying types
    # ----
    type_summary = {}
    for col in df.columns:
        # Get the Pandas dtype
        p_dtype = df[col].dtype
        # Get the actual Python type of the first non-null element
        first_valid = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
        actual_type = type(first_valid).__name__ if first_valid is not None else "All Null"
        #
        type_summary[col] = {"Pandas_Dtype": p_dtype, "Actual_Type": actual_type}
    # Convert to DataFrame for a clean view
    df_types = pd.DataFrame(type_summary).T
    print(df_types)
    pass