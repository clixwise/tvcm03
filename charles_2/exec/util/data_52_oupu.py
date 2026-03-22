import os
import glob
import sys
import pandas as pd
from pathlib import Path
import datetime
from great_tables import GT
import os
import matplotlib.pyplot as plt
    
from util.data_31_figu import FiguTran

# ****
# Clas
# ****
class OupuTran:
    def __init__(self, proc_tran):
        self.proc_tran = proc_tran
        proc_tran.oupu = self
        #
        self.dire = None
        self.func = None
        
    def upda(self):
        raise NotImplementedError
# ----
# Stat
# ----
class OupuTranFile(OupuTran):
    def __init__(self, proc_tran):
        super().__init__(proc_tran)    
        self.fram_dict = {}

    def upda(self):
        oupu_stat_exec(self)

# ----
# Grph
# ----        
class OupuTranGrph(OupuTran):
    def __init__(self, proc_tran):
        super().__init__(proc_tran) 
        self.figu_dict = {}

    def upda(self):
        oupu_grph_exec(self)

# ****
# Meth
# ****

# ----
# Stat
# ----
def oupu_stat_exec(oupuTran:OupuTranFile):
    
    trac = True
    
    # Exec
    # ----
    # oupuTran.fram_dict[StatTranQOL_01_desc.__name__] =  { "resu_publ" : statTran.stat_tran_desc.resu_publ,  "resu_tech" : statTran.stat_tran_desc.resu_tech }
    for key1, val1 in oupuTran.fram_dict.items():       
        for key2, val2 in val1.items(): 
            df = val2['df']      
            mode = val2['mode']    
            print (f'\n****\nkey1:{key1}\n****\n----\nkey2:{key2}\n----\nval2:{df}')
            pass
            dire = oupuTran.dire
            file = f"{key1} {key2}"
            file_path = os.path.join(dire, file)
            file_path = os.path.normpath(file_path)
            _oupu_stat_exec_remo(file_path)
            _oupu_stat_exec_crea(df, mode, file_path)
            pass   
    pass

def _oupu_stat_exec_remo(file):
    
    # Exec
    # ----
    base = os.path.dirname(file)
    fnam = os.path.basename(file)
    patt_list = [os.path.join(base, glob.escape(fnam) + "*")]
    # print (patt_list)
    for patt in patt_list:

        file_list = glob.glob(patt)
        # print (file_list)
        for file_path in file_list:
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    # print(f"Deleted: {file_path}")
            except Exception as e:
                # print(f"Error deleting {file_path}: {e}")
                pass
                
def _oupu_stat_exec_crea(fram_orig, mode, file): 
 
    # Exec
    # ----
    fram = fram_orig.copy() # precaution since we reset index to be incorporated into column(s)
    fram = fram.reset_index()
    date_time = datetime.datetime.now().strftime("%H-%M-%S_%Y-%m-%d")
    file_date = f'{file} {date_time}'
    
    # Wide
    # ----
    tags = [t.strip() for t in mode.split(',')]
    #
    # ----
    exec_csv = 'csv' in tags
    # ----
    if exec_csv:
        separator = '|'
        fram.to_csv(f'{file_date}.csv', sep=separator, index=False)              
    #
    # ----
    exec_md = 'md' in tags
    # ----
    if exec_md:
        with open(f'{file_date}.md', 'w', encoding='utf-8') as f:
            mark_oupu = fram.to_markdown(index=False)
            f.write(mark_oupu)
    #
    # ----
    exec_xlsx = 'xlsx' in tags
    # ----
    if exec_xlsx:
        file_name = f'{file_date}.xlsx'
        with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
            fram.to_excel(writer, index=False, sheet_name='QOL_Report')
            # Auto-adjust column widths ; note simplistic is : fram.to_excel(f'{file_date}.xlsx', index=False)
            worksheet = writer.sheets['QOL_Report']
            for idx, col in enumerate(fram.columns):
                series = fram[col]
                max_len = max(
                    series.astype(str).map(len).max(),  # Len of longest value
                    len(str(series.name))               # Len of column name
                ) + 2  # Add a little extra padding
                worksheet.set_column(idx, idx, max_len)
    #
    # ----
    exec_frm1 = 'ft01' in tags
    # ----
    if exec_frm1:
        
        def export_to_publication_table(df, file_name="Table_1_Demographics.xlsx"):
            # 1. Formatting logic for the 'Value' column
            def format_medical_value(val):
                if pd.isna(val) or '(' not in str(val):
                    return ""
                # Extract N and Percentage, swap dot for comma
                n_val = val.split('(')[0].strip()
                perc_val = val.split('(')[1].split('%')[0].replace('.', ',').strip()
                
                # Tweak: Present as "N (Percentage%)" or "N (Percentage)" 
                # Here we follow the image style but with parentheses for clarity
                return f"{n_val} ({perc_val}%)"

            # 2. Define Column Parameters (Alignment and Width)
            # 0 = Left, 1 = Center, 2 = Right
            column_settings = {
                'Variable': {'align': 'left', 'width': 40, 'header': 'Variable'},
                'Value':    {'align': 'right', 'width': 20, 'header': 'Patients (N,%)'}
            }

            table_data = []
            for category in df['Category'].unique():
                table_data.append({'Text': category, 'Val': '', 'is_cat': True})
                
                sub_df = df[df['Category'] == category]
                for _, row in sub_df.iterrows():
                    label = str(row['Subcategory'])
                    label = "No data" if label in ["-", ""] else label
                    
                    table_data.append({
                        'Text': label, 
                        'Val': format_medical_value(row['Value']), 
                        'is_cat': False
                    })

            # 3. Create Excel and apply styling
            writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
            workbook = writer.book
            ws = workbook.add_worksheet("Demographics")

            # Define Style Objects
            # Header: Bold, Top/Bottom lines, Specific alignment
            header_left = workbook.add_format({'bold': True, 'top': 2, 'bottom': 2, 'align': 'left'})
            header_right = workbook.add_format({'bold': True, 'top': 2, 'bottom': 2, 'align': 'right'})

            # Body Styles
            cat_style = workbook.add_format({'bold': True, 'font_size': 11, 'align': 'left'})
            indent_style = workbook.add_format({'indent': 2, 'font_size': 11, 'align': 'left'})
            
            # Right-aligned value style for the N (%) column
            val_style = workbook.add_format({'align': 'right', 'font_size': 11})
            
            footer_style = workbook.add_format({'top': 2})

            # Write Headers with requested justification
            ws.write(0, 0, column_settings['Variable']['header'], header_left)
            ws.write(0, 1, column_settings['Value']['header'], header_right)

            # Write Content
            row_idx = 1
            for item in table_data:
                if item['is_cat']:
                    ws.write(row_idx, 0, item['Text'], cat_style)
                    ws.write(row_idx, 1, "", val_style)
                else:
                    ws.write(row_idx, 0, item['Text'], indent_style)
                    ws.write(row_idx, 1, item['Val'], val_style)
                row_idx += 1

            # Final footer line
            # ws.set_row(row_idx, None, footer_style)

            # Apply Column Widths
            ws.set_column(0, 0, column_settings['Variable']['width'])
            ws.set_column(1, 1, column_settings['Value']['width'])
            
            ws.hide_gridlines(2)
            writer.close()
            print(f"Table generated with right-justified values: {file_name}")

        # Usage:
        file_name = f'{file_date}.xlsx'
        export_to_publication_table(fram, file_name)
    #
    # ----
    exec_frm2 = 'ft02' in tags
    # ----
    if exec_frm2:
        
        def export_adherence_table(df, file_name="Adherence_Report.xlsx", category_label="Follow-up Adherence"):
            # 1. Calculate the total for percentage computation
            total_n = df['Count'].sum()

            # 2. Setup Excel engine
            writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
            workbook = writer.book
            ws = workbook.add_worksheet("Statistics")

            # 3. Define Styles (Journal Standard)
            header_l = workbook.add_format({'bold': True, 'top': 2, 'bottom': 2, 'align': 'left'})
            header_r = workbook.add_format({'bold': True, 'top': 2, 'bottom': 2, 'align': 'right'})
            cat_bold = workbook.add_format({'bold': True, 'align': 'left'})
            sub_indt = workbook.add_format({'indent': 2, 'align': 'left'})
            val_rght = workbook.add_format({'align': 'right'})
            line_bot = workbook.add_format({'top': 2})

            # 4. Write Headers
            ws.write(0, 0, "Variable", header_l)
            ws.write(0, 1, "Patients (N,%)", header_r)

            # 5. Write Category Header Row
            ws.write(1, 0, category_label, cat_bold)
            ws.write(1, 1, "", val_rght)

            # 6. Process and Write Data Rows
            current_row = 2
            for _, row in df.iterrows():
                n = row['Count']
                # Calculate percentage: (n / total) * 100
                percentage = (n / total_n * 100) if total_n > 0 else 0
                
                # Format: Comma decimal, 1 decimal place -> "15 (51,7%)"
                val_str = f"{n} ({f'{percentage:.1f}'.replace('.', ',')}%)"
                
                ws.write(current_row, 0, row['Follow-up Adherence'], sub_indt)
                ws.write(current_row, 1, val_str, val_rght)
                current_row += 1

            # 7. Final Touches (Bottom Border, Column Width, Gridlines)
            # ws.set_row(current_row, None, line_bot)
            ws.set_column(0, 0, 40)
            ws.set_column(1, 1, 20)
            ws.hide_gridlines(2)

            writer.close()
            print(f"File saved: {file_name}")

        # Usage:
        file_name = f'{file_date}.xlsx'
        export_adherence_table(fram, file_name)

# ----
# Grph
# ----      
def oupu_grph_exec(oupuTran:OupuTranGrph):
    
    # Exec
    # ----
    for key, value in oupuTran.figu_dict.items():
        figuTran:FiguTran = value
        figu = figuTran.fig
        dire = oupuTran.dire
        file = key
        file_path = os.path.join(dire, file)
        file_path = os.path.normpath(file_path)
        _oupu_grph_exec_remo(file_path)
        _oupu_grph_exec_crea(figu, file_path)    
        pass

def _oupu_grph_exec_remo(file):
    
    # Exec
    # ----
    base = os.path.dirname(file)
    fnam = os.path.basename(file)
    patt_list = [os.path.join(base, glob.escape(fnam) + "*")]
    # print (patt_list)
    for patt in patt_list:

        file_list = glob.glob(patt)
        # print (file_list)
        for file_path in file_list:
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    # print(f"Deleted: {file_path}")
            except Exception as e:
                # print(f"Error deleting {file_path}: {e}")
                pass
            
def _oupu_grph_exec_crea(figu, file): 
 
    # Exec
    # ----
    date_time = datetime.datetime.now().strftime("%H-%M-%S_%Y-%m-%d")
    file_date = f'{file} {date_time}'
    
    # Jpeg
    # ----
    jpeg_save = True
    if jpeg_save:
        file_crea = f'{file_date}.jpg'
        figu.savefig(file_crea, format='jpeg', dpi=600)
    
    # Pdfx
    # ----
    pdfx_save = False
    if pdfx_save:
        file_crea = f'{file_date}.pdf'
        figu.savefig(file_crea, format='pdf')
 
    # Clos
    # ----
    plt.close(figu)
    pass
   
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