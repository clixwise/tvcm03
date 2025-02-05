import os
import sys
import pandas as pd

from ke82_ceap_comp_c3c6_full_abso import ke82_ceap_comp_c3c6_full_abso
from grph01.gr05.keep_v61.refe_2024_11_06a_pati_xxxx_ceap_xxxx.ke19_pati_agbi_mbre_c3c6_full_abso_ import ke19_pati_agbi_mbre_c3c6_full_abso
from grph01.gr05.keep_v61.refe_2024_11_06a_pati_xxxx_ceap_xxxx.ke46_pati_agbi_unbi_c3c6_PART_abso_ import ke46_pati_agbi_unbi_c3c6_PART_abso
from grph01.gr05.keep_v61.refe_2024_11_06a_pati_xxxx_ceap_xxxx.ke48_pati_agbi_mbre_c3c6_PART_abso_idem_ke19_ import ke48_pati_agbi_mbre_c3c6_PART_abso
from grph01.gr05.keep_v61.refe_2024_11_06a_pati_xxxx_ceap_xxxx.ke47_pati_agbi_sexe_c3c6_PART_abso_idem_ke15_ import ke47_pati_agbi_sexe_c3c6_PART_abso
from grph01.gr05.keep_v61.refe_2024_11_06a_pati_xxxx_ceap_xxxx.ke15_pati_agbi_sexe_c3c6_full_abso_ import ke15_pati_agbi_sexe_c3c6_full_abso
from grph01.gr05.keep_v61.refe_2024_11_06a_pati_xxxx_ceap_xxxx.ke16_pati_agbi_unbi_c3c6_full_abso_ import ke16_pati_agbi_unbi_c3c6_full_abso
from ke15_c3c6_agbi_sexe_c3c6_full_abso_idem_ke47 import ke15_c3c6_agbi_sexe_c3c6_full_abso
from ke19_c3c6_agbi_mbre_c3c6_full_abso_idem_ke48 import ke19_c3c6_agbi_mbre_c3c6_full_abso
from grph01.gr05.keep_v61.refe_2024_11_06a_pati_xxxx_ceap_xxxx.ke35_ceap_agbi_c3c6_full_abso_ import ke35_ceap_agbi_c3c6_full_abso
from grph01.gr05.keep_v61.refe_2024_11_06a_pati_xxxx_ceap_xxxx.ke37_ceap_sexe_c3c6_full_abso_ import ke37_ceap_sexe_c3c6_full_abso
from grph01.gr05.keep_v61.refe_2024_11_06a_pati_xxxx_ceap_xxxx.ke39_ceap_mbre_c3c6_full_abso_ import ke39_ceap_mbre_c3c6_full_abso

if __name__ == "__main__":
    
    # PATI non poindéré
    ke15_pati_agbi_sexe_c3c6_full_abso()
    ke16_pati_agbi_unbi_c3c6_full_abso()
    ke19_pati_agbi_mbre_c3c6_full_abso() # NOOP
    ke15_c3c6_agbi_sexe_c3c6_full_abso() # ke15 idem ke47
    ke19_c3c6_agbi_mbre_c3c6_full_abso() # ke19 idem ke48
    
    # PATI pondéré par C3...C6
    ke46_pati_agbi_unbi_c3c6_PART_abso() # ke19 (NOOP) idem ke46
    ke47_pati_agbi_sexe_c3c6_PART_abso() # ke15 idem ke47
    ke48_pati_agbi_mbre_c3c6_PART_abso() # ke19 idem ke48
    
    # CEAP
    ke35_ceap_agbi_c3c6_full_abso()
    ke37_ceap_sexe_c3c6_full_abso()
    ke39_ceap_mbre_c3c6_full_abso()
    # CEAP Kinshasa ; Bonn
    ke82_ceap_comp_c3c6_full_abso()
