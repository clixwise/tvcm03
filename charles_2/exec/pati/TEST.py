
import pandas as pd

def generate_medical_table(df, output_file="Medical_Statistics_Table.xlsx"):
    """
    Transforms a long-format DataFrame into a publication-ready Excel table.
    """
    
    def format_value_str(val_str):
        """Converts '24 (80.0%)' to '24 80,0'"""
        if pd.isna(val_str) or "(" not in str(val_str):
            return ""
        n_part = val_str.split("(")[0].strip()
        perc_part = val_str.split("(")[1].replace(")", "").replace("%", "").strip()
        # Replace dot with comma for European/Medical formatting
        perc_part = perc_part.replace(".", ",")
        return f"{n_part} {perc_part}"

    # Mapping for specific subcategory labels
    label_mapping = {
        "Primary": "Primary education",
        "Secondary": "Secondary education",
        "Higher": "Higher education",
        "-": "No data",
        "": "No data"
    }

    formatted_rows = []
    categories = df['Category'].unique()

    for cat in categories:
        # 1. Add the Category Header Row
        formatted_rows.append({
            "Variable": cat, 
            "Value": "", 
            "is_bold": True, 
            "indent": 0
        })
        
        # 2. Filter and process Subcategories
        sub_df = df[df['Category'] == cat]
        for _, row in sub_df.iterrows():
            sub = str(row['Subcategory']).strip()
            
            # Apply custom labels if they exist in mapping
            display_label = label_mapping.get(sub, sub)
            
            formatted_rows.append({
                "Variable": display_label,
                "Value": format_value_str(row['Value']),
                "is_bold": False,
                "indent": 1 # For visual nesting
            })

    # Create Excel with styling
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    workbook  = writer.book
    worksheet = workbook.add_worksheet("Results")

    # Define Formats
    header_fmt = workbook.add_format({'bold': True, 'bottom': 2, 'top': 2, 'font_name': 'Arial', 'font_size': 10})
    cat_fmt    = workbook.add_format({'bold': True, 'font_name': 'Arial', 'font_size': 10})
    std_fmt    = workbook.add_format({'font_name': 'Arial', 'font_size': 10})
    indent_fmt = workbook.add_format({'font_name': 'Arial', 'font_size': 10, 'indent': 2})
    footer_fmt = workbook.add_format({'bottom': 2})

    # Write Headers
    worksheet.write(0, 0, "Variable", header_fmt)
    worksheet.write(0, 1, "Patients (N,%)", header_fmt)

    # Write Data
    for i, row in enumerate(formatted_rows, start=1):
        if row['is_bold']:
            curr_fmt = cat_fmt
        elif row['indent'] > 0:
            curr_fmt = indent_fmt
        else:
            curr_fmt = std_fmt
            
        worksheet.write(i, 0, row['Variable'], curr_fmt)
        worksheet.write(i, 1, row['Value'], curr_fmt)

    # Add the closing bottom line on the last row
    worksheet.set_row(len(formatted_rows), None, footer_fmt)

    # Adjust column widths
    worksheet.set_column(0, 0, 35)
    worksheet.set_column(1, 1, 20)

    writer.close()
    print(f"File saved as: {output_file}")

    
def export_to_publication_tableOLD(df, filename="Table_1_Demographics.xlsx"):
    # 1. Formatting logic for the 'Value' column
    def format_medical_value(val, use_parentheses=True):
        if pd.isna(val) or '(' not in str(val):
            return ""
        
        # Extract N and the percentage string
        n_val = val.split('(')[0].strip()
        # Get the number inside the parentheses, remove %, and swap dot for comma
        perc_val = val.split('(')[1].split('%')[0].replace('.', ',').strip()
        
        # Tweak: Ensure at least one decimal place (e.g., "80" becomes "80,0")
        if ',' not in perc_val:
            perc_val = f"{perc_val},0"

        if use_parentheses:
            return f"{n_val} ({perc_val})"
        else:
            # If you prefer a non-bracketed look but with a clearer separator (like a semi-colon or tab)
            return f"{n_val}  {perc_val}"

    # 2. Build the structured list for the Excel rows
    table_data = []
    for category in df['Category'].unique():
        # Add the Bold Category Header
        table_data.append({'Text': category, 'Val': '', 'is_cat': True})
        
        # Add Subcategories
        sub_df = df[df['Category'] == category]
        for _, row in sub_df.iterrows():
            label = str(row['Subcategory'])
            # Apply specific naming conventions
            label = "Primary education" if label == "Primary" else label
            label = "Higher education" if label == "Higher" else label
            label = "No data" if label in ["-", ""] else label
            
            table_data.append({
                'Text': label, 
                'Val': format_medical_value(row['Value']), 
                'is_cat': False
            })

    # 3. Create Excel and apply the SAGE/Phlebology styling
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    workbook = writer.book
    ws = workbook.add_worksheet("Demographics")

    # Define exact styles based on your image
    header_style = workbook.add_format({'bold': True, 'top': 2, 'bottom': 2, 'font_size': 11})
    category_style = workbook.add_format({'bold': True, 'font_size': 11})
    indent_style = workbook.add_format({'indent': 2, 'font_size': 11})
    value_style = workbook.add_format({'align': 'center', 'font_size': 11})
    footer_style = workbook.add_format({'top': 2}) # For the very bottom line

    # Write Headers
    ws.write(0, 0, "Variable", header_style)
    ws.write(0, 1, "Patients (N,%)", header_style)

    # Write Content
    row_idx = 1
    for item in table_data:
        if item['is_cat']:
            ws.write(row_idx, 0, item['Text'], category_style)
            ws.write(row_idx, 1, "", category_style)
        else:
            ws.write(row_idx, 0, item['Text'], indent_style)
            ws.write(row_idx, 1, item['Val'], value_style)
        row_idx += 1

    # Apply bottom border to the last row of data to close the table
    ws.set_row(row_idx, None, footer_style)

    # Set column widths to match image proportions
    ws.set_column(0, 0, 40)
    ws.set_column(1, 1, 20)
    
    # Hide gridlines for that "Clean Manuscript" look
    ws.hide_gridlines(2)

    writer.close()
    print(f"Success: '{filename}' matches the reference image.")

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

    

    #generate_medical_table(df_input, "Final_Report.xlsx")
