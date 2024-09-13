This repo calculates the electron cosmic ray in general IGM (related paper: https://arxiv.org/abs/2311.18721), and includes following files:

LCSMSphere.py: Density evolution

Energy_loss_class.py: Energy loss

get_source.py: Generate source file; We generated "source_term_Khaire.txt" using Khaire's EBL model (KS_2018_EBL) and "source_term_Haardt.txt" using Haardt's EBL model (UBV.out.txt)

IGM.py will generator 5 npy file: energy density utot, energy per baryon in eV E_eV, the number of electron per volume per erg N, pressure in erg/cm^3 P_e, and pressure in eV/cm^3 P_eV, with tail _{deltalin0}_{source_model}

(wokring on callable)beam.py: using Mrk421

(working on callable): generate F_NP

distribution_diff.py will generate electron distribution counting angular broadening by Compton scattering and Coulomb scattering

(working on): damp rate by beam itself and by CRe
