import os
import sys
import pandas as pd
from ke10_ceap_xxxx import ke10_main

'''
pati : 362
mbre : 724 dont C0...C6:619 ; NA:105
ceap : 876 dont C0...C6:771 ; NA:105
ceap_pair : 524
'''

def ke19_c3c6_agbi_mbre_c3c6_full_abso():

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
    ceap_mono = False
    indx_name = 'mbre'  
    indx_cate_list = ['G', 'D']
    filt_name = 'sexe'
    filt_valu = None # 'M' 'F'
    #
    suppress_suffix = ".py"
    script_name = script_name[:-len(suppress_suffix)]
    jrnl_file_path = os.path.join(script_dir, f'{script_name}_{filt_valu}_trac.txt' if filt_valu is not None else f'{script_name}_trac.txt')
    ke10_main(indx_name, indx_cate_list, ceap_mono, filt_name, filt_valu, file_path, jrnl_file_path) 
    pass

#
# inspired from 'C:\tate01\grph01\gr05\keep_v61\pro1\agbi_c3c6_sexe_mbre JUNK\keyy_mbre_c3c6_full_abso_main.py'
#
if __name__ == "__main__":
    ke19_c3c6_agbi_mbre_c3c6_full_abso()