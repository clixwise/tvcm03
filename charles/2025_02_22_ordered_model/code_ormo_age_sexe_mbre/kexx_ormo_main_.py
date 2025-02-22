import os
import sys
import pandas as pd
from ke30_ormo_ceap_xxxx import ke30_main

def ke37_ceap_sexe_c3c6_full_abso():

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
    # Sexe
    # ---
    ceap_mono = False
    ind1_name = 'sexe'  
    ind1_cate_list = ['M', 'F']
    ind2_name = 'mbre'
    ind2_cate_list = ['G', 'D']
    filt_name = 'sexe'
    filt_valu = None
    #    
    suppress_suffix = ".py"
    script_name = script_name[:-len(suppress_suffix)]
    jrnl_file_path = os.path.join(script_dir, f'{script_name}jrnl.txt')
    ke30_main(ind1_name, ind1_cate_list, ind2_name, ind2_cate_list, ceap_mono, filt_name, filt_valu, file_path, jrnl_file_path) 
    pass

if __name__ == "__main__":
    ke37_ceap_sexe_c3c6_full_abso()