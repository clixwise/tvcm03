import os
import sys
import pandas as pd
from ke30_ceap_xxxx import ke30_main

'''
pati : 362
mbre : 724 dont C0...C6:619 ; NA:105
ceap : 876 dont C0...C6:771 ; NA:105
ceap_pair : 524
'''

def ke39_ceap_mbre_c3c6_full_abso():

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
    # Sexe = None
    # -----------
    ceap_mono = False
    indx_name = 'mbre'  
    indx_cate_list = ['G', 'D']
    filt_name = 'sexe'
    filt_valu = None # 'M' 'F'
    #    
    suppress_suffix = ".py"
    script_name_A = script_name[:-len(suppress_suffix)]
    jrnl_file_path = os.path.join(script_dir, f'{script_name_A}jrnl.txt')
    jrnl_file_path = os.path.join(script_dir, f'{script_name_A}jrnl_{filt_valu}.txt' if filt_valu is not None else f'{script_name_A}jrnl.txt')
    ke30_main(indx_name, indx_cate_list, ceap_mono, filt_name, filt_valu, file_path, jrnl_file_path) 
    #
    # Sexe = 'M'
    # ----------
    ceap_mono = False
    indx_name = 'mbre'  
    indx_cate_list = ['G', 'D']
    filt_name = 'sexe'
    filt_valu = 'M' # 'M' 'F'
    #    
    suppress_suffix = ".py"
    script_name_M = script_name[:-len(suppress_suffix)]
    jrnl_file_path = os.path.join(script_dir, f'{script_name_M}jrnl.txt')
    jrnl_file_path = os.path.join(script_dir, f'{script_name_M}jrnl_{filt_valu}.txt' if filt_valu is not None else f'{script_name_M}jrnl.txt')
    ke30_main(indx_name, indx_cate_list, ceap_mono, filt_name, filt_valu, file_path, jrnl_file_path) 
    #
    # Sexe = 'F'
    # ----------
    ceap_mono = False
    indx_name = 'mbre'  
    indx_cate_list = ['G', 'D']
    filt_name = 'sexe'
    filt_valu = 'F' # 'M' 'F'
    #    
    suppress_suffix = ".py"
    script_name_F = script_name[:-len(suppress_suffix)]
    jrnl_file_path = os.path.join(script_dir, f'{script_name_F}jrnl.txt')
    jrnl_file_path = os.path.join(script_dir, f'{script_name_F}jrnl_{filt_valu}.txt' if filt_valu is not None else f'{script_name_F}jrnl.txt')
    ke30_main(indx_name, indx_cate_list, ceap_mono, filt_name, filt_valu, file_path, jrnl_file_path) 
    pass

if __name__ == "__main__":
    ke39_ceap_mbre_c3c6_full_abso()