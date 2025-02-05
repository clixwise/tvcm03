import numpy as np
import pandas as pd
from util_file_mngr import write
from scipy import stats
from statsmodels.stats.contingency_tables import SquareTable

# -------------------------------
# Kruskal-Wallis H
# -------------------------------
def kruswall_clau(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, colu_name_ordi, df1):
    
    # Trac
    trac = True

    # 'df_tabl' only 
    if df1 is None:
        print(f"\nData : {what}\n(df_table) : Kruskal-Wallis H : requires 'df_line' wich is None : {indx_name}:{indx_cate_list} : {colu_name}:{colu_cate_list}")
        write(f"\nData : {what}\n(df_table) : Kruskal-Wallis H : requires 'df_line' wich is None : {indx_name}:{indx_cate_list} : {colu_name}:{colu_cate_list}")
        return
    
    # Prec
    indx_cate_nam1 = indx_cate_list[0]
    indx_cate_nam2 = indx_cate_list[1]
    colu_list_ord1 = df1[df1[indx_name] == indx_cate_nam1][colu_name_ordi] # colu_list_ord1 ordinal [0...n] : ceap series for M ages
    colu_list_ord2 = df1[df1[indx_name] == indx_cate_nam2][colu_name_ordi] # colu_list_ord2 ordinal [0...n] : ceap series for F ages
    if trac:
        print(f"\nStep 0 : colu_list_ord1.size:{len(colu_list_ord1)} df2.type:{type(colu_list_ord1)}\n{colu_list_ord1}\n:{colu_list_ord1.index}")
        print(f"\nStep 0 : colu_list_ord2.size:{len(colu_list_ord2)} df2.type:{type(colu_list_ord2)}\n{colu_list_ord2}\n:{colu_list_ord2.index}")
        write(f"\nStep 0 : colu_list_ord1.size:{len(colu_list_ord1)} df2.type:{type(colu_list_ord1)}\n{colu_list_ord1}\n:{colu_list_ord1.index}")
        write(f"\nStep 0 : colu_list_ord2.size:{len(colu_list_ord2)} df2.type:{type(colu_list_ord2)}\n{colu_list_ord2}\n:{colu_list_ord2.index}")

    # Exec
    stat, pval = stats.kruskal(colu_list_ord1, colu_list_ord2)

    # Resu
    if np.isnan(stat) or np.isnan(pval):
        raise Exception("Stat or Pval are NaN")
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    print(f"\nData : {what}\n(clau) Kruskal-Wallis H : Stat:{stat_form} Pval:{pval_form}")
    write(f"\nData : {what}\n(clau) Kruskal-Wallis H : Stat:{stat_form} Pval:{pval_form}")  

    # Intp
    H0 = f"(clau) H0 : The '{colu_name}' distributions are the same across '{indx_name}'\n({colu_cate_list}) vs ({indx_cate_list})"
    Ha = f"(clau) Ha : The '{colu_name}' distributions are different across '{indx_name}'\n({colu_cate_list}) vs ({indx_cate_list})"
    alpha = 0.05
    if pval < alpha:
        print(f"(clau) Kruskal-Wallis H : Reject the null hypothesis:\n{Ha}")
        write(f"(clau) Kruskal-Wallis H : Reject the null hypothesis:\n{Ha}")
    else:
        print(f"(clau) Kruskal-Wallis H : Fail to reject the null hypothesis:\n{H0}")
        write(f"(clau) Kruskal-Wallis H : Fail to reject the null hypothesis:\n{H0}")
    pass