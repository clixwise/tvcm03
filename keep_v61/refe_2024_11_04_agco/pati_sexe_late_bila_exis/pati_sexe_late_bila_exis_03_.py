
from datetime import datetime
import numpy as np
import os
import sys
import pandas as pd
from util_file_mngr import set_file_objc, write
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
import numpy as np
from scipy.stats import chi2_contingency

def stat_mono(what, pati_male, pati_fema, d_res):  
   
    global d_res_curr
    stat_sepa(d_res)
    
    pati_tota = pati_male + pati_fema
    pati_half = pati_tota / 2

    observed = [pati_male, pati_fema]  # Male, Female
    expected = [pati_half, pati_half]  # Expected counts if 50-50 split

    # ----
    # Chi square
    # ----
    chi2, pval = stats.chisquare(observed, expected)
    stat = chi2
    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\nstats.chisquare : Stat:{stat_form} Pval:{pval_form}")
    write(f"\nData : {what}\nstats.chisquare : Stat:{stat_form} Pval:{pval_form}")
    d_res.loc[f'{d_res_curr}'] = [what, 'chisquare', stat_form, pval_form, '',''] ; d_res_curr += 1

    # ----
    # Binomial
    # ----
    result = stats.binomtest(pati_male, n=pati_tota, p=0.5, alternative='two-sided')
    stat = result.statistic
    pval = result.pvalue
    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\nstats.binomtest : Stat:{stat_form} Pval:{pval_form}")
    write(f"\nData : {what}\nstats.binomtest : Stat:{stat_form} Pval:{pval_form}")
    d_res.loc[f'{d_res_curr}'] = [what, 'binomtest', stat_form, pval_form, '',''] ; d_res_curr += 1

    # ----
    # Fisher
    # ----
    # Create a 2x2 contingency table
    observed = [[pati_male, pati_fema],  # Observed counts: [Male, Female]
                [pati_half, pati_half]]  # Expected counts if 50-50 split
    odds_ratio, pval = stats.fisher_exact(observed)
    stat = odds_ratio
    #
    log_odds_ratio = np.log(odds_ratio)
    np_table = np.array(observed)
    se = np.sqrt(sum(1 / np_table.flatten())) # Calculate standard error
    alpha = 0.05  # or any other significance level you prefer
    ci_lower = np.exp(log_odds_ratio - stats.norm.ppf(1 - alpha / 2) * se)
    ci_upper = np.exp(log_odds_ratio + stats.norm.ppf(1 - alpha / 2) * se)

    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    ci_lower_form = f"{ci_lower:.3e}" if ci_lower < 0.001 else f"{ci_lower:.3f}"
    ci_upper_form = f"{ci_upper:.3e}" if ci_upper < 0.001 else f"{ci_upper:.3f}"
    print(f"\nData : {what}\nstats.fisher_exact : Stat:{stat_form} Pval:{pval_form} 95%CI:{ci_lower_form}-{ci_upper_form}")
    write(f"\nData : {what}\nstats.fisher_exact : Stat:{stat_form} Pval:{pval_form} 95%CI:{ci_lower_form}-{ci_upper_form}")
    d_res.loc[f'{d_res_curr}'] = [what, 'fisher_exact', stat_form, pval_form, '',''] ; d_res_curr += 1

    # ----
    # Wilson
    # ----
    def wilson_score_interval(count, nobs, alpha=0.05):
        """
        Calculate Wilson score interval for a proportion.
        
        Parameters:
        count (int): Number of successes
        nobs (int): Total number of observations
        alpha (float): Significance level (default 0.05 for 95% CI)
        
        Returns:
        tuple: (lower bound, upper bound) of the confidence interval
        """
        n = nobs
        p = count / n
        z = stats.norm.ppf(1 - alpha / 2)
        
        denominator = 1 + z**2 / n
        centre_adjusted_probability = p + z**2 / (2 * n)
        adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
        
        lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
        upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
        
        return (lower_bound, upper_bound)

    # Calculate proportion and confidence interval
    proportion = pati_male / pati_tota
    ci_lower, ci_upper = wilson_score_interval(pati_male, pati_tota)

    print(f"\nData : {what}\nWilson score interval : Proportion of unilateral CVI: {proportion:.3f} 95% Confidence Interval: ({ci_lower:.3f}, {ci_upper:.3f})")
    write(f"\nData : {what}\nWilson score interval : Proportion of unilateral CVI: {proportion:.3f} 95% Confidence Interval: ({ci_lower:.3f}, {ci_upper:.3f})")
    d_res.loc[f'{d_res_curr}'] = [what, 'wilson unilat', '', '', ci_lower, ci_upper] ; d_res_curr += 1
    
    # You can also calculate for bilateral cases
    bilateral_cases = pati_fema
    bi_proportion = bilateral_cases / pati_tota
    bi_ci_lower, bi_ci_upper = wilson_score_interval(bilateral_cases, pati_tota)

    print(f"\nData : {what}\nWilson score interval : Proportion of bilateral CVI: {bi_proportion:.3f} 95% Confidence Interval: ({bi_ci_lower:.3f}, {bi_ci_upper:.3f})")
    write(f"\nData : {what}\nWilson score interval : Proportion of bilateral CVI: {bi_proportion:.3f} 95% Confidence Interval: ({bi_ci_lower:.3f}, {bi_ci_upper:.3f})")
    d_res.loc[f'{d_res_curr}'] = [what, 'wilson bilat', '', '', ci_lower, ci_upper] ; d_res_curr += 1

def stat_dual_exam():

    # Chi2 example 1
    # --------------
    exam = True
    if True:
        data = {
            'Sex': ['Male', 'Male', 'Female', 'Female'],
            'Leg Disease': ['Left', 'Right', 'Left', 'Right'],
            'Count': [18, 34, 21, 32]
        }
        df = pd.DataFrame(data)
        print (df)
        contingency_table = df.pivot_table(values='Count', index='Sex', columns='Leg Disease', aggfunc='sum')
        print (contingency_table)
        chi2, pval, dof, expected = chi2_contingency(contingency_table)
        if np.isnan(chi2) or np.isnan(pval):
            raise Exception("Stat or Pval are NaN")
        stat_form = f"{chi2:.3e}" if chi2 < 0.001 else f"{chi2:.3f}"
        pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
        print(f"\nData : Leg Disease by Sex\nchi2_contingency : Stat:{stat_form} Pval:{pval_form}")
        print(f"Degrees of Freedom: {dof}")
        print(f"Expected Frequencies:\n{expected}")
        
    # Chi2 example 2
    # --------------
    exam = True
    if True:
        # Contingency table
        observed = np.array([[18, 34], [21, 32]])
        chi2, pval, dof, expected = chi2_contingency(observed)
        if np.isnan(chi2) or np.isnan(pval):
            raise Exception("Stat or Pval are NaN")
        stat_form = f"{chi2:.3e}" if chi2 < 0.001 else f"{chi2:.3f}"
        pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
        print(f"\nData : Leg Disease by Sex\nchi2_contingency : Stat:{stat_form} Pval:{pval_form}")
        print(f"Degrees of Freedom: {dof}")
        print(f"Expected Frequencies:\n{expected}")

def stat_dual(what, df, indx, colu, d_res):
    
    global d_res_curr
    stat_sepa(d_res)
    
    # Chi2
    # ----
    cont_tabl = df
    print(f"\ncont_tabl.size:{len(cont_tabl)} type:{type(cont_tabl)}\n{cont_tabl}\n:{cont_tabl}\n:{cont_tabl}")
    chi2, pval, dof, expected = chi2_contingency(cont_tabl)
    if np.isnan(chi2) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{chi2:.3e}" if chi2 < 0.001 else f"{chi2:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\nchi2_contingency : Stat:{stat_form} Pval:{pval_form} Dof: {dof} Exp:\n{expected}")
    write(f"\nData : {what}\nchi2_contingency : Stat:{stat_form} Pval:{pval_form} Dof: {dof} Exp:\n{expected}")
    d_res.loc[f'{d_res_curr}'] = [what, 'chi2_contingency', stat_form, pval_form, '',''] ; d_res_curr += 1

def stat_sepa(d_res):
    global d_res_curr
    d_res.loc[f'{d_res_curr}'] = ['', '', '', '', '', ''] ; d_res_curr += 1
def stat_head(d_res):
    global d_res_curr
    stat_sepa(d_res)
    d_res.loc[f'{d_res_curr}'] = ['---', '---', '---', '---', '---', '---'] ; d_res_curr += 1
def oper_sexe(date_form, d_res):
        # OK 2025_01_08
        pati_male = 156
        pati_fema = 206
        write ("\n")
        write (">>> >>> >>>")
        write (f'{date_form} PATI_SEXE : Sexe: M:{pati_male} F:{pati_fema} Ratio:{pati_fema/pati_male}')
        write (">>> >>> >>>")
        # Inpu
        indx_name = 'sexe' ; colu_name = 'freq' ; indx_cate_list = ['M','F'] ; colu_cate_list = ['coun']
        data = { 'freq': [pati_male, pati_fema] } ; df = pd.DataFrame(data, index=indx_cate_list)
        df.index.name = indx_name ; df.columns = [colu_name] ; print(df)
        # Stat
        what = f"'{indx_name}' '{colu_name}' ; {indx_cate_list} {colu_cate_list}"
        pati_male = df.loc['M', 'freq'] # 156
        pati_fema = df.loc['F', 'freq'] # 206  
        stat_mono(what, pati_male, pati_fema, d_res)
'''
      coun_A  coun_M  coun_F
mbre
G        323     138     185
D        296     122     174
'''
def oper_late(date_form, d_res):
    
    stat_head(d_res)
    
    # Stat Mono
    # ---------
    def aide(prof, pati_left, pati_righ, d_res):
        write ("\n")
        write (">>> >>> >>>")
        write (f'{date_form} PATI_LATE : Mbre[Sexe:{prof}]: G:{pati_left} D:{pati_righ} Ratio:{pati_righ/pati_left}')
        write (">>> >>> >>>")
        # Inpu
        indx_name = 'late' ; colu_name = 'freq' ; indx_cate_list = ['G','D'] ; colu_cate_list = ['coun']
        data = { 'freq': [pati_left, pati_righ] } ; df = pd.DataFrame(data, index=indx_cate_list)
        df.index.name = indx_name ; df.columns = [colu_name] ; print(df)
        # Stat
        what = f"Sexe:{prof} '{indx_name}' '{colu_name}' ; {indx_cate_list} {colu_cate_list}"
        pati_left = df.loc['G', 'freq'] # 323
        pati_righ = df.loc['D', 'freq'] # 296   
        stat_mono(what, pati_left, pati_righ, d_res)
        
    # OK 2025_01_08
    pati_left = 323 ; pati_righ = 296
    aide("M&F", pati_left, pati_righ, d_res)
    pati_left = 138 ; pati_righ = 122
    aide("M", pati_left, pati_righ, d_res)
    pati_left = 185 ; pati_righ = 174
    aide("F", pati_left, pati_righ, d_res)
   
    # Stat Dual
    # ---------
    prof = "M<>F"
    write ("\n")
    write (">>> >>> >>>")
    write (f'{date_form} PATI_LATE : Mbre[Sexe:{prof}]')
    write (">>> >>> >>>")
    indx_name = 'sexe' ; indx_cate_list = ['M','F']
    colu_name = 'mbre' ; colu_cate_list = ['G','D']
    df = pd.DataFrame({
        'G': [138, 185],
        'D': [122, 174]
    }, index=indx_cate_list)
    df.index.name = indx_name
    df.columns.name = colu_name
    what = f"Sexe:M<>F '{indx_name}' {indx_cate_list} ; '{colu_name}' {colu_cate_list}"
    stat_dual(what, df, indx_name, colu_name, d_res)

'''
      coun_A  coun_M  coun_F
unbi
U        105      52      53
B        257     104     153
'''              
def oper_bila(date_form, d_res):
    
    stat_head(d_res)
        
    # Stat Mono
    # ---------
    def aide(prof, pati_unil, pati_bila, d_res):
        write ("\n")
        write (">>> >>> >>>")
        write (f'{date_form} PATI_BILA : Unbi[Sexe:{prof}]: U:{pati_unil} B:{pati_bila} Ratio:{pati_bila/pati_unil}')
        write (">>> >>> >>>")
        # Inpu
        indx_name = 'bila' ; colu_name = 'freq' ; indx_cate_list = ['U','B'] ; colu_cate_list = ['coun']
        data = { 'freq': [pati_unil, pati_bila] } ; df = pd.DataFrame(data, index=indx_cate_list)
        df.index.name = indx_name ; df.columns = [colu_name] ; print(df)
        # Stat
        what = f"Sexe:{prof} '{indx_name}' '{colu_name}' ; {indx_cate_list} {colu_cate_list}"
        pati_unil = df.loc['U', 'freq'] # 105
        pati_bila = df.loc['B', 'freq'] # 257  
        stat_mono(what, pati_unil, pati_bila, d_res)
    # OK 2025_01_08
    pati_unil = 105 ; pati_bila = 257
    aide("M&F", pati_unil, pati_bila, d_res)
    pati_unil = 52 ; pati_bila = 104
    aide("M", pati_unil, pati_bila, d_res)
    pati_unil = 53 ; pati_bila = 153
    aide("F", pati_unil, pati_bila, d_res)
   
    # Stat Dual
    # ---------
    prof = "M<>F"
    write ("\n")
    write (">>> >>> >>>")
    write (f'{date_form} PATI_BILA : Unbi[Sexe:{prof}]')
    write (">>> >>> >>>")
    indx_name = 'sexe' ; indx_cate_list = ['M','F']
    colu_name = 'bila' ; colu_cate_list = ['U','B']
    df = pd.DataFrame({
        'U': [52, 53],
        'B': [104, 153]
    }, index=indx_cate_list)
    df.index.name = indx_name
    df.columns.name = colu_name
    what = f"Sexe:M<>F '{indx_name}' {indx_cate_list}; '{colu_name}' {colu_cate_list}"
    stat_dual(what, df, indx_name, colu_name, d_res)
'''
      coun_A  coun_M  coun_F
mbre
G         39      18      21
D         66      34      32
'''  
def oper_exis(date_form, d_res):
    
    stat_head(d_res)
        
    # Stat Mono
    # ---------
    def aide(prof, pati_left, pati_righ, d_res):
        write ("\n")
        write (">>> >>> >>>")
        write (f'{date_form} PATI_EXIS : Exis[Sexe:{prof}]: G:{pati_left} D:{pati_righ} Ratio:{pati_righ/pati_left}')
        write (">>> >>> >>>")
        # Inpu
        indx_name = 'exis' ; colu_name = 'freq' ; indx_cate_list = ['G','D'] ; colu_cate_list = ['coun']
        data = { 'freq': [pati_left, pati_righ] } ; df = pd.DataFrame(data, index=indx_cate_list)
        df.index.name = indx_name ; df.columns = [colu_name] ; print(df)
        # Stat
        what = f"Sexe:{prof} '{indx_name}' '{colu_name}' ; {indx_cate_list} {colu_cate_list}"
        pati_left = df.loc['G', 'freq'] # 39
        pati_righ = df.loc['D', 'freq'] # 66  
        stat_mono(what, pati_left, pati_righ, d_res)
        
    # OK 2025_01_08
    pati_left = 39 ; pati_righ = 66
    aide("M&F", pati_left, pati_righ, d_res)
    pati_left = 18 ; pati_righ = 34
    aide("M", pati_left, pati_righ, d_res)
    pati_left = 21 ; pati_righ = 32
    aide("F", pati_left, pati_righ, d_res)
   
    # Stat Dual
    # ---------
    prof = "M<>F"
    write ("\n")
    write (">>> >>> >>>")
    write (f'{date_form} PATI_EXIS : Exis[Sexe:{prof}]')
    write (">>> >>> >>>")
    indx_name = 'sexe' ; indx_cate_list = ['M','F']
    colu_name = 'mbre' ; colu_cate_list = ['G','D']
    df = pd.DataFrame({
        'G': [18, 21],
        'D': [34, 32]
    }, index=indx_cate_list)
    df.index.name = indx_name
    df.columns.name = colu_name
    what = f"Sexe:M<>F '{indx_name}' {indx_cate_list}; '{colu_name}' {colu_cate_list}"
    stat_dual(what, df, indx_name, colu_name, d_res)
        
def exec(filt_valu, file_path, jrnl_file_path):
    
     with open(jrnl_file_path, 'w') as file:
        
        set_file_objc(file) 
        
        # d_res
        d_res = pd.DataFrame(columns=['what', 'test', 'stat', 'pval', 'ci_lower', 'ci_upper'])
        d_res.index.name = 'indx'
        d_res.columns.name = 'stat'
        global d_res_curr ; d_res_curr = 1
        
        # Exec
        date_curr = datetime.now()
        date_form = date_curr.strftime('%Y-%m-%d %H:%M:%S')
        write ("\n")
        write (">>> >>> >>>")
        write (f'pati_sexe_late_bila_exis_03_ = SEXE (sexe) : {date_form}')
        write (">>> >>> >>>")
        oper_sexe(date_form, d_res) 
        write ("\n")
        write (">>> >>> >>>")
        write (f'pati_sexe_late_bila_exis_03_ = LATE (mbre) : {date_form}')
        write (">>> >>> >>>")
        oper_late(date_form, d_res)
        write ("\n")
        write (">>> >>> >>>")
        write (f'pati_sexe_late_bila_exis_03_ = BILA (unbi) : {date_form}')
        write (">>> >>> >>>")
        oper_bila(date_form, d_res)
        write ("\n")
        write (">>> >>> >>>")
        write (f'pati_sexe_late_bila_exis_03_ = EXIS (exis) : {date_form}')
        write (">>> >>> >>>")
        oper_exis(date_form, d_res)
        
        # d_res
        d_res = d_res.reset_index()
        with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
                print(f"\nData :\n{d_res}")
                write(f"\nData :\n{d_res}")
        pass
        xlsx = True
        if xlsx: 
            file_name = 'pati_sexe_late_bila_exis_03_.xlsx'
            d_res.to_excel(file_name, index=False)
def main():
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
    #
    filt_valu = None
    #
    suppress_suffix = ".py"
    script_name = script_name[:-len(suppress_suffix)]
    jrnl_file_path = os.path.join(script_dir, f'{script_name}_{filt_valu}_trac.txt' if filt_valu is not None else f'{script_name}jrnl.txt')
    exec(filt_valu, file_path, jrnl_file_path) 
    
if __name__ == "__main__":
    main()