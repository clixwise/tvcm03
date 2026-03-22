import os
import sys
import pandas as pd
from pathlib import Path

# ****
# Clas
# ****

class OrchTran:
    def __init__(self):
        self.dict = {}
 
        # 6.5.1	Study Population and Follow-Up
        # 6.5.1.1	Screening and Inclusion
        self.df_ta01_scre_incl = None
        self.ta01_scre_incl = {}
        '''
        •	Number screened
        •	Number eligible
        •	Number undergoing RFA
        •	Number with baseline data (T0)
        •	Number with T1 (3 months)
        •	Number with T2 (12 months)
        •	Lost to follow-up (with reasons)
        This section is descriptive only.
        '''
        self.ta01_scre_incl['Screened'] = (None, None)
        self.ta01_scre_incl['Eligible'] = (None, None)
        self.ta01_scre_incl['Underwent RFA'] = (None, None)
        self.ta01_scre_incl['T0 (baseline) assessment'] = (None, None)
        self.ta01_scre_incl['T1 (3 months) completed'] = (None, None)
        self.ta01_scre_incl['T2 (12 months) completed'] = (None, None)
        self.ta01_scre_incl['Lost to follow-up at T1 (with reasons)'] = (None, None)
        self.ta01_scre_incl['Lost to follow-up at T2 (with reasons)'] = (None, None)

        # 6.5.1	Study Population and Follow-Up
        # 6.5.1.1	Demographics
        self.wk00_base_char = {}
        self.wk00_base_char['Residential Origin, n (%)'] = {}
        self.wk00_base_char['Source of Referral, n (%)'] = {}
        self.wk00_base_char['Education level, n (%)'] = {}
        self.wk00_base_char['Marital status, n (%)'] = {}
        self.wk00_base_char['Confession, n (%)'] = {}
        self.wk00_base_char['Occupational Status, n (%)'] = {}
        self.wk00_base_char['Occupational Physical Activity, n (%)'] = {}
        self.wk00_base_char['Prior Healthcare-Seeking Behavior, n (%)'] = {}
        self.wk00_base_char['Primary Payer, n (%)'] = {}
        self.wk00_base_char['Barriers to Treatment Adherence, n (%)'] = {} 
        self.wk00_base_char['Follow-up Adherence Pattern, n (%)'] = {} 
        self.wk00_base_char['Service Evaluation, n (%)'] = {}
        
        self.df_ta00_base_char = None
        self.ta00_base_char = {}
        
        # 6.5.1	Study Population and Follow-Up
        # 6.5.1.2	 Baseline Characteristics
        
        # Table 1. Patient assessment and procedural characteristics
        
        self.df_ta01_base_char = None
        self.ta01_base_char = {}
        self.ta01_base_char['Patient characteristics'] = (None, None)
        self.ta01_base_char['Age, years'] = (None, None)
        self.ta01_base_char['Female, n (%)'] = (None, None)
        self.ta01_base_char['BMI, kg/m²'] = (None, None)
        self.ta01_base_char['-1'] = ("-", "-")
        self.ta01_base_char['CEAP class, n (%)'] = ("", "")
        self.ta01_base_char['.C1'] = (None, None)
        self.ta01_base_char['.C2'] = (None, None)
        self.ta01_base_char['.C3'] = (None, None)
        self.ta01_base_char['.C4'] = (None, None)
        self.ta01_base_char['.C5'] = (None, None)
        self.ta01_base_char['.C6'] = (None, None)
        self.ta01_base_char['-2'] = ("-", "-")
        self.ta01_base_char['Laterality on duplex ultrasound, n (%)'] = ("", "")
        self.ta01_base_char['.Unilateral'] = (None, None)
        self.ta01_base_char['.Bilateral'] = (None, None)
        self.ta01_base_char['-3'] = ("-", "-")
        self.ta01_base_char['Baseline scores'] = ("", "")
        self.ta01_base_char['.VEINES-QOL'] = (None, None)
        self.ta01_base_char['.VEINES-Sym'] = (None, None)
        self.ta01_base_char['.VCSS'] = (None, None)     
        self.ta01_base_char['-4'] = ("-", "-")
        self.ta01_base_char['Intended follow-up type, n (%)'] = ("", None)
        self.ta01_base_char['.Preventive'] = (None, None)
        self.ta01_base_char['.Curative'] = (None, None)
        self.ta01_base_char['.None intended'] = (None, None)
        self.ta01_base_char['.Missing info'] = (None, None) 
        '''
        Note : as much patient-level as possible
        ----
|   index | Variable                               | Value          | Script                |
|--------:|:---------------------------------------|:---------------|:----------------------|
|       0 | Patient characteristics                | (N = 30)       | c02_exam_01_stat_desc |
|       1 | Age, years                             | 57.2 ± 13.9    | c02_exam_01_stat_desc |
|       2 | Female, n (%)                          | 11 (36.7%)     | c02_exam_01_stat_desc |
|       3 | BMI, kg/m²                             | 31.0 ± 8.3     | c02_exam_01_stat_desc |
|       4 | -1                                     | -              | -                     |
|       5 | CEAP class, n (%)                      |                |                       |
|       6 | .C3                                    | 7 (23.3%)      | c02_exam_01_stat_desc |
|       7 | .C4                                    | 4 (13.3%)      | c02_exam_01_stat_desc |
|       8 | .C5                                    | 4 (13.3%)      | c02_exam_01_stat_desc |
|       9 | .C6                                    | 15 (50.0%)     | c02_exam_01_stat_desc |
|      10 | -2                                     | -              | -                     |
|      11 | Laterality on duplex ultrasound, n (%) |                |                       |
|      12 | .Unilateral                            | 9 (30.0%)      | c02_exam_01_stat_desc |
|      13 | .Bilateral                             | 21 (70.0%)     | c02_exam_01_stat_desc |
|      14 | -3                                     | -              | -                     |
|      15 | Baseline scores                        |                |                       |
|      16 | .VEINES-QOL                            | 50.0 ± 3.5     | c02_qol_01_stat_desc  |
|      17 | .VEINES-Sym                            |                |                       |
|      18 | .VCSS                                  | 9.0 (5.0–13.0) | c02_vcss_01_stat_desc |
|      19 | -4                                     | -              | -                     |
|      20 | Intended follow-up type, n (%)         |                |                       |
|      21 | .Preventive                            | 15 (50.00%)    | c02_pati_01_stat_foll |
|      22 | .Curative                              | 15 (50.00%)    | c02_pati_01_stat_foll |
|      23 | .None intended                         | 0 (0.00%)      | c02_pati_01_stat_foll |
|      24 | .Missing info                          |                |                       |
        '''
        
        # Table 2. Limb assessment and procedural characteristics

        self.df_ta02_base_char = None
        self.ta02_base_char = {}
        self.ta02_base_char['Limb characteristics'] = (None, None)
        self.ta02_base_char['-1'] = ("-", "-")
        self.ta02_base_char['Limbs with CVD, n'] = (None, None)
        self.ta02_base_char['CVD-affected limbs per patient'] = (None, None) # More precise
        self.ta02_base_char['-2'] = ("-", "-")
        self.ta02_base_char['Treated limbs, n'] = (None, None) # "Treated limbs" is more concise
        self.ta02_base_char['Treated limbs per patient'] = (None, None)
        self.ta02_base_char['-3'] = ("-", "-")
        self.ta02_base_char['Procedure duration per limb, min'] = (None, None) # "Procedure" is better than "Surgery" for RFA
        self.ta02_base_char['Anesthesia'] = ("", "")
        self.ta02_base_char['.Modality, n (%)'] = (None, None)
        self.ta02_base_char['..General'] = (None, None)
        self.ta02_base_char['..Regional'] = (None, None)
        self.ta02_base_char['..Local'] = (None, None)
        self.ta02_base_char['..-31 Not specified'] = (None, None)
        self.ta02_base_char['.Agent used, n (%)'] = (None, None)
        self.ta02_base_char['..Propofol'] = (None, None)
        self.ta02_base_char['..Prilocaine'] = (None, None)
        self.ta02_base_char['..Bupivacaine'] = (None, None)
        self.ta02_base_char['..-32 Not specified'] = (None, None)
        self.ta02_base_char['.Mean concentration, % (SD)'] = (None, None)
        self.ta02_base_char['..Propofol'] = (None, None)
        self.ta02_base_char['..Prilocaine'] = (None, None)
        self.ta02_base_char['..Bupivacaine'] = (None, None)
        self.ta02_base_char['..-33 Not specified'] = (None, None)
        self.ta02_base_char['-4'] = ("-", "-")
        self.ta02_base_char['Target veins treated*, n (%)'] = ("", "")
        self.ta02_base_char['.Great saphenous vein'] = (None, None) # Full name is better for tables; or use "GSV"
        self.ta02_base_char['.Small saphenous vein'] = (None, None) # Full name "Small" is current standard (vs Short)
        self.ta02_base_char['.Anterior accessory saphenous vein'] = (None, None) # ASV is commonly AASV in journals
        self.ta02_base_char['.-4 Not specified'] = (None, None) # More professional than "Missing info"
        self.ta02_base_char['*Data as n (%) of total treated veins'] = (None, None)
        
        '''
        Note : limbs == treatment level only
        ----
|   index | Variable                                 | Value                  | Script                |
|--------:|:-----------------------------------------|:-----------------------|:----------------------|
|       0 | Limb characteristics                     | (L = 60 limbs)         | c02_exam_01_stat_desc |
|       1 | -1                                       | -                      | -                     |
|       2 | Limbs with CVD, n                        | 51                     | c02_exam_01_stat_desc |
|       3 | Limbs with CVD per patient               | 1.7 ± 0.5              | c02_exam_01_stat_desc |
|       4 | -2                                       | -                      | -                     |
|       5 | Limbs treated, n                         | 43                     | c02_exam_01_stat_desc |
|       6 | Limbs treated per patient                | 1.4 ± 0.7              | c02_exam_01_stat_desc |
|       7 | -3                                       | -                      | -                     |
|       8 | Target veins treated*, n (%)             |                        |                       |
|       9 | .GSV                                     | 39 (58.2%)             | c02_exam_01_stat_desc |
|      10 | .SSV                                     | 20 (29.9%)             | c02_exam_01_stat_desc |
|      11 | .ASV                                     | 4 (6.0%)               | c02_exam_01_stat_desc |
|      12 | .Missing info                            | 4 (6.0%)               | c02_exam_01_stat_desc |
|      13 | *Percentages calculated per treated vein | (63 veins in 43 limbs) | c02_exam_01_stat_desc |

        '''
        
        # Table 1. Adjuvant therapies
        
        self.df_ta04_ther_adju = None
        self.ta04_ther_adju = {}
        self.ta04_ther_adju['Adjuvant therapies'] = (None, None)
        self.ta04_ther_adju['UGFS'] = (None, None)
        self.ta04_ther_adju['PRP'] = (None, None)
        self.ta04_ther_adju['Veinotropes'] = (None, None)
        self.ta04_ther_adju['Hyperbaric oxygen therapy'] = (None, None)
        self.ta04_ther_adju['GLP-1 receptor agonists'] = (None, None)
        self.ta04_ther_adju['Hygiene and nutritional counseling'] = (None, None)
        self.ta04_ther_adju['Pressotherapy, bandaging'] = (None, None)
        self.ta04_ther_adju['Physical activity'] = (None, None)
        self.ta04_ther_adju['Compression stockings'] = (None, None)
        self.ta04_ther_adju['.All of the time'] = (None, None)
        self.ta04_ther_adju['.Most of the time'] = (None, None)
        self.ta04_ther_adju['.A good bit of the time'] = (None, None)
        self.ta04_ther_adju['.Some of the time'] = (None, None)
        self.ta04_ther_adju['.Rarely'] = (None, None)
        self.ta04_ther_adju['.Never'] = (None, None)
        
        self.df_ta03_endp_prim_raww = None
        self.ta03_endp_prim_raww = {}
        self.ta03_endp_prim_raww['Mean ± SD at T0'] = (None, None)
        self.ta03_endp_prim_raww['Mean ± SD at T1'] = (None, None)
        self.ta03_endp_prim_raww['Mean ± SD at T2'] = (None, None)
        self.ta03_endp_prim_raww['T0–T2 : Absolute mean ± SD change'] = (None, None)
        self.ta03_endp_prim_raww['T0–T2 : % change'] = (None, None)
        
        self.df_ta04_endp_prim_modl = None
        self.ta04_endp_prim_modl = {}
        self.ta04_endp_prim_modl['Wald F'] = (None, None) # you are fully justified in stating an overall time effect.
        self.ta04_endp_prim_modl['Mean score at T0'] = (None, None)
        self.ta04_endp_prim_modl['Mean score at T1'] = (None, None)
        self.ta04_endp_prim_modl['Mean score at T2'] = (None, None)
        self.ta04_endp_prim_modl['Adjusted mean difference at T1'] = (None, None)
        self.ta04_endp_prim_modl['Adjusted mean difference at T2'] = (None, None) 
        self.ta04_endp_prim_modl['Adjusted mean difference β at T1'] = (None, None)
        self.ta04_endp_prim_modl['Adjusted mean difference β at T2'] = (None, None)
        
        self.df_ta05_endp_prim_modl = None
        '''
          timepoint  mean   se    ci_lower  ci_upper  n
        0  T0        50.02  0.93  48.20     51.83     30
        1  T1        54.16  1.00  52.20     56.13     24
        2  T2        57.29  0.93  55.47     59.10     30
        '''
        
        self.df_ta06_endp_effe_size = None
        self.ta06_endp_effe_size = {}
        self.ta06_endp_effe_size['Mean change (T0–T2)'] = (None, None)
        self.ta06_endp_effe_size['Cohen’s d'] = (None, None)
        self.ta06_endp_effe_size['Standardized response mean (SRM)'] = (None, None)
        self.ta06_endp_effe_size['Model-based mean change (T0–T2)'] = (None, None)
        self.ta06_endp_effe_size['Model-based Cohen’s d'] = (None, None)
        self.ta06_endp_effe_size['Patients achieving MCID ≥4'] = (None, None)

        self.df_ta07_mcid = None
        self.ta07_mcid = {}
        self.ta07_mcid['Anchor-based (mean change)'] = (None, None)
        self.ta07_mcid["Mean change in 'Somewhat better'"] = (None, None)
        self.ta07_mcid['Best threshold'] = (None, None)
        self.ta07_mcid['Distribution-based (0.5 SD baseline)'] = (None, None)
        self.ta07_mcid['Distribution-based (0.3 SD baseline)'] = (None, None)
        self.ta07_mcid['Distribution-based (SEM)'] = (None, None)
        self.ta07_mcid['Distribution-based (MDC95 : Minimal Detectable Change at 95% confidence)'] = (None, None)
        
        '''
        | Metric                           | Value |
        | -------------------------------- | ----- |
        | Mean change (T0–T2)              | X.X   |
        | Standardized response mean (SRM) | X.XX  |
        | Model-based Cohen’s d            | X.XX  |
        | Patients achieving MCID (Distribution) | XX%   |
        | Patients achieving MCID (Anchor)       | XX%   |
        | Patients achieving MCID (ROC)          | XX%   |
        '''
        
class OrchTranStat(OrchTran):
    def __init__(self):
        super().__init__()
        
    def upda(self):
        upda_exec(self)

def upda_exec(orch_tran: OrchTranStat):
    
    trac = True
    
    # Exec
    # ----
    rows = [ (var, val_source[0], val_source[1]) for var, val_source in orch_tran.ta00_base_char.items() ]
    orch_tran.df_ta00_base_char = pd.DataFrame(rows, columns=['Variable', 'Value', 'Script'])
    
    rows = [ (var, val_source[0], val_source[1]) for var, val_source in orch_tran.ta01_scre_incl.items() ]
    orch_tran.df_ta01_scre_incl = pd.DataFrame(rows, columns=['Variable', 'Value', 'Script'])
    
    rows = [ (var, val_source[0], val_source[1]) for var, val_source in orch_tran.ta01_base_char.items() ]
    orch_tran.df_ta01_base_char = pd.DataFrame(rows, columns=['Variable', 'Value', 'Script'])
    rows = [ (var, val_source[0], val_source[1]) for var, val_source in orch_tran.ta02_base_char.items() ]
    orch_tran.df_ta02_base_char = pd.DataFrame(rows, columns=['Variable', 'Value', 'Script'])
    
    rows = [ (var, val_source[0], val_source[1]) for var, val_source in orch_tran.ta04_ther_adju.items() ]
    orch_tran.df_ta04_ther_adju = pd.DataFrame(rows, columns=['Variable', 'Value', 'Script'])
           
    rows = [ (var, val_source[0], val_source[1]) for var, val_source in orch_tran.ta03_endp_prim_raww.items() ]
    orch_tran.df_ta03_endp_prim_raww = pd.DataFrame(rows, columns=['Variable', 'Value', 'Script'])
    
    rows = [ (var, val_source[0], val_source[1]) for var, val_source in orch_tran.ta04_endp_prim_modl.items() ]
    orch_tran.df_ta04_endp_prim_modl = pd.DataFrame(rows, columns=['Variable', 'Value', 'Script'])
    
    rows = [ (var, val_source[0], val_source[1]) for var, val_source in orch_tran.ta06_endp_effe_size.items() ]
    orch_tran.df_ta06_endp_effe_size = pd.DataFrame(rows, columns=['Variable', 'Value', 'Script'])
    
    rows = [ (var, val_source[0], val_source[1]) for var, val_source in orch_tran.ta07_mcid.items() ]
    orch_tran.df_ta07_mcid = pd.DataFrame(rows, columns=['Variable', 'Value', 'Script'])
    
    # Trac
    # ----
    if trac:
        print_yes(orch_tran.df_ta00_base_char, labl="orch_tran.df_ta00_base_char")
        print_yes(orch_tran.df_ta01_scre_incl, labl="orch_tran.df_ta01_scre_incl")
        print_yes(orch_tran.df_ta01_base_char, labl="orch_tran.df_ta01_base_char")
        print_yes(orch_tran.df_ta02_base_char, labl="orch_tran.df_ta02_base_char")
        print_yes(orch_tran.df_ta04_ther_adju, labl="orch_tran.df_ta04_ther_adju")
        print_yes(orch_tran.df_ta03_endp_prim_raww, labl="orch_tran.df_ta03_endp_prim_raww")
        print_yes(orch_tran.df_ta04_endp_prim_modl, labl="orch_tran.df_ta04_endp_prim_modl")
        print_yes(orch_tran.df_ta05_endp_prim_modl, labl="orch_tran.df_ta05_endp_prim_modl")
        print_yes(orch_tran.df_ta06_endp_effe_size, labl="orch_tran.df_ta06_endp_effe_size")
        print_yes(orch_tran.df_ta07_mcid, labl="orch_tran.df_ta07_mcid")
    pass
    
        
def print_yes(df, labl=None):
    if df is None:
        return
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
        