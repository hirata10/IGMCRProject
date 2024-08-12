Eq3 calculates the number of e+e- pairs produced per photon of lab-frame energy Eγ, per unit lab-frame time, per Lorentz factor of the electron γe, per unit solid angle. Corresponding code files: beam.py, rate_draft.ipynb, rate.pkl, rate-cube.py

Eq5 calculates the number of electrons after time t of cooling; nij in Eq6 represent the number of electron, where i is the angular bin and j is the energy (or Lorentz factor) bin; while nij(t) in Eq7 counts the cooling process. Corresponding code: the def nij part in distribution file
distribution_withdiff_flatfile.py counts the distribution with inverse Compton scattering (Eq24)
distribution_diff.ipynb added the multiple Coulomb scattering (Eq39)

Eq43 calculates the growth/damp rate caused by the beam itself. Corresponding code file: IGMCR/growth_rate_beam_m.ipynb, it also compares the damp rate caused by beam itself and the damp rate caused by MeV cosmic ray

Eq56_draft.ipynb calculates the role of refraction <- currently where problems occur, that sin-1(x) with x >> 1
