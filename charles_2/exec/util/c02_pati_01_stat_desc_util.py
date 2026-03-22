

import pandas as pd

def export_to_publication_table(df, filename="Table_1_Demographics.xlsx"):
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
            label = "Primary education" if label == "Primary" else label
            label = "Higher education" if label == "Higher" else label
            label = "No data" if label in ["-", ""] else label
            
            table_data.append({
                'Text': label, 
                'Val': format_medical_value(row['Value']), 
                'is_cat': False
            })

    # 3. Create Excel and apply styling
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
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
    ws.set_row(row_idx, None, footer_style)

    # Apply Column Widths
    ws.set_column(0, 0, column_settings['Variable']['width'])
    ws.set_column(1, 1, column_settings['Value']['width'])
    
    ws.hide_gridlines(2)
    writer.close()
    print(f"Table generated with right-justified values: {filename}")

# Usage:
# export_to_publication_table(df)

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Sample data matching your input structure
    raw_data = {
        'Category': ['Residence', 'Residence', 'Education level', 'Education level', 'Marital status'],
        'Subcategory': ['Kinshasa', 'DR Congo', 'Primary', 'Higher', 'Married'],
        'Value': ['24 (80.0%)', '3 (10.0%)', '1 (3.3%)', '19 (63.3%)', '15 (50.0%)']
    }

    df_input = pd.DataFrame(raw_data)
    export_to_publication_table(df_input)