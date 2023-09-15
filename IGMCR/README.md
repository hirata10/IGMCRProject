Electron cosmic ray in general IGM

To use this IGM.py, call IGM_N(deltalin0, model)

where deltalin0 is overdensity factor (<0 means underdensity, 0 means mean density, >0 means overdensity),
and model is either 0 or 1 (0 using Khaire's source model and 1 using Haardt's source model)

all variables should be individual values rather than array or list

.get_utot() will return to the energy density utot and energy per baryon in eV E_eV

.get_P() will return to the number of electron per volume per erg N, pressure in erg cm^-3 P_e and pressure in eV cm^-3 P_eV

result.ipynb is an example that using IGM.py to generate needed valeus
