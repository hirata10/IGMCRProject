This dictionary calculates the electron cosmic ray in general IGM, and includes following files:

LCSMSphere.py: Density evolution

Energy_loss_class.py: Energy loss

get_source.ipynb: Source term; We generate "source_term_Khaire.txt" using Khaire's EBL model (KS_2018_EBL) and "source_term_Haardt.txt" using Haardt's EBL model (UBV.out.txt)

IGM.py: The main class

To use this class, call IGM_N(deltalin0, model)
where deltalin0 is overdensity factor (<0 means underdensity, 0 means mean density, >0 means overdensity),
and model is either 0 or 1 (0 using Khaire's EBL model and 1 using Haardt's EBL model)

all variables should be individual values rather than array or list

.get_utot() will return to the energy density utot and energy per baryon in eV E_eV
.get_P() will return to the number of electron per volume per erg N, pressure in erg cm^-3 P_e and pressure in eV cm^-3 P_eV

temp_history.py: IGM temperature profile

result.ipynb: The default calculating using different over- (or under-) density at different redshift using different EBL model and computing the contribution of electron cosmic ray to general IGM using Khaire's EBL model
