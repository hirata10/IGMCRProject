import numpy as np
import pandas as pd
import h5py
import argparse
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from astropy.io import fits
import IGM

r0 = 2.818e-13 # classical electron radius in cm
m_e = 9.10938e-28 # electron mass in gram
c = 2.998e10 # speed of light in cm/s
h = 6.6261e-27 # planck constant in cm^2 g s-1
eV_to_erg = 1.6022e-12
parsec_to_cm = 3.085677581e18 # cm per parsec

kB = 1.3808e-16 # boltzmann constant in cm^2 g s^-2 K^-1
T_CMB = 2.725 # CMB temperature today in K

H_0 = 2.184e-18 # current hubble constant in s^-1
H_r = 3.24076e-18 # 100 km/s/Mpc to 1/s
h_0 = H_0 / H_r
omega_b_0 = 0.0224 / h_0 / h_0  # current baryon abundance
m_p = 1.67262193269e-24 # proton mass in g
G = 6.6743e-8 # gravitational constant in cm^3 g^-1 s^-2
f_He = 0.079 # helium mass fraction
f_H = 0.76 # hydrogen fraction
Y_He = 0.24

E_e = m_e * c * c
rho_crit = 3 * H_0 * H_0 / 8 / np.pi / G
e = 4.80326e-10 # electron charge in esu

z_reion_He = 3 # redshift that He is fully ionized
z_reion_H = 8 # redshift that H is fully ionized

"""
This script will generate a .h5 file storing 't': incident time in second, 'k': wavenumber, 'k_norm': normalized wavenumber k/(omega_p/c), 'omega_x': propagation direction omega_x, and 'damp_rate_beam': damping rate by beam itself
If the --k_file is not None, it will use the desired k, otherwise it will use default k = 1+1e-8 to 2 times omega_p/c, and save as damp_rate_beam_{z}_{deltalin0}.h5

This script will also compare the result between the damp rate by beam and the damp rate by cosmic ray if --CR_file is not None, and will generate a .h5 file compare_{z}_{deltalin0}.h5 with 't', 'omega_x', and 'compare', where 'compare' is the normalized k that damp rate caused by CR starting dominant

run python damp_rate.py --nstep --mstep --E_min --E_max --source_model --deltalin0 --z --k_file --CR_file --readPATH --savePATH

the default setting is reshift bins nstep = 399, energy bins mstep = 399, E_min = 1e-11 erg, E_max = 1e-3 erg,

source_model = '0' for Khaire's model ('1' for Haardt's mode),

overdensity evolution deltalin0 = 0 for mean density (> 0 for overdensity, and < 0 for underdensity), redshfit z = 2,

k_file = None, CR_file = None,

saving path = '', and reading path = '' (should be the same as the saving path of IGM.py)

Please make sure the parameters are consistent with other files
"""

def find_idx(z, z_list):
    if z <= z_list[-1]:
        return -1
    idx = np.where(z_list<=z)[0][0]
    if np.abs(z_list[idx] - z) <= np.abs(z_list[idx-1] - z):
        return idx
    else:
        return idx-1

def main(args):
    global T_CMB
    z = args.z
    deltalin0 = args.deltalin0
    source_model = args.source_model

    z2 = (1 + z) / 3
    T_CMB *= 1 + z

    df = pd.read_csv(f'{args.readPATH}temp_history_{deltalin0}.csv')
    length = len(df[df['z']>=z])
    T_IGM = df[df['z']>=z]['T'][length-1]

    gamma_e_arr = np.logspace(8, 14, 400) * eV_to_erg / m_e / c**2 # lab-frame Lorentz factor of the electron produced
    theta_e_arr = np.logspace(-8,0, 400)

    t_arr = args.T_arr

    with fits.open(f'{args.readPATH}nij.fits') as hdul:
        nij = hdul[0].data # shape (t_num, theta, gamma)

    d_gamma_e = np.array([gamma_e_arr[i] - gamma_e_arr[i-1] for i in range(1, 400)])
    d_gamma_e = np.insert(d_gamma_e, 0, 0)

    n_wt = np.zeros((nij.shape[0], nij.shape[1]))

    for i in range(nij.shape[0]):
        for j in range(nij.shape[1]):
            n_wt[i][j] = np.sum(d_gamma_e / gamma_e_arr * nij[i][j])


    def get_weight(t_num, omega_x):
        j_p = np.where(theta_e_arr>abs(omega_x))[0][0]
    
        n_wt_jp1 = n_wt[t_num][j_p+1:]
        n_wt_jp = n_wt[t_num][j_p:-1]
    
        theta_jp1 = theta_e_arr[j_p+1:]
        theta_jp = theta_e_arr[j_p:-1]
    
        sum_ = np.sum((n_wt_jp1 - n_wt_jp) / (theta_jp1 - theta_jp) * (np.arccosh(theta_jp1 / abs(omega_x)) - np.arccosh(theta_jp / abs(omega_x))))
        return 2 * omega_x * sum_


    IGM_00 = IGM.IGM_N(deltalin0, source_model, args.nstep, args.mstep, args.E_min, args.E_max)

    index_z = find_idx(z, IGM_00.z)
    omega_b = omega_b_0 * (1 + z)**3
    mean_n_b = 3 * omega_b * H_0 * H_0 / (8 * np.pi * G * m_p) # mean number density of baryons at the redshift
    n_b = mean_n_b * IGM_00.Delta[index_z] # local number density of baryons at the redshitf
    
    n_H = n_b * (1 - Y_He) # nuclei in hydrogen
    n_He = n_H * f_He # nuclei in helium, assume He is all 4He

    if z <= z_reion_He: # fully ionized
        n_e = n_H + 2 * n_He
    if z_reion_He < z <= z_reion_H: # He is singly ionized
        n_e = n_H + n_He
    if z > z_reion_H: # neutral
        n_e = 0
    
    omega_p = (4 * np.pi * r0 * n_e)**0.5 * c
    print(f'omega_p = {omega_p}')


    def get_growth_rate(t_num, theta_c, omega_x):
        co1 = get_weight(t_num, omega_x)
        theta = theta_c + omega_x
        co2 = np.pi * omega_p / 2 / n_e * np.cos(theta)**2
        return co1 * co2

    def get_growth_rate_beam(k, omega_x):
        v_ph = omega_p / k
        theta_c = np.arccos(v_ph / c)
        growth_rate = np.zeros((nij.shape[0], k.shape[0], omega_x.shape[0]))
        for i in range(nij.shape[0]):
            for j in range(k.shape[0]):
                for h in range(omega_x.shape[0]):
                    growth_rate[i][j][h] = get_growth_rate(i, theta_c[j], omega_x[h])
        return growth_rate

    if args.k_file is None:
        k = omega_p / c * (np.geomspace(1e-8, 1, 1000) + 1)
    else:
        k_norm = np.load(args.k_file)
        k = k_norm * omega_p / c
        if k.ndim != 1:
            k = k.flatten()
            
    omega_x = np.logspace(-8,-1, 400)
    growth_rate_beam = get_growth_rate_beam(k, omega_x)

    with h5py.File(f'{args.savePATH}damp_rate_beam_{z:.1f}_{deltalin0:.1f}.h5', 'w') as h5f:
        h5f.create_dataset('t', data=t_arr, compression="gzip")
        h5f.create_dataset('k', data=k, compression="gzip")
        h5f.create_dataset('k_norm', data=k/(omega_p/c), compression="gzip")
        h5f.create_dataset('omega_x', data=omega_x, compression="gzip")
        h5f.create_dataset('damp_rate_beam', data=-growth_rate_beam, compression="gzip")
        
    if args.CR_file is not None:
        ## Compare damp rate by beam and damp rate by electron cosmic ray
        with h5py.File(args.CR_file, 'r') as h5f:
            k_CR_norm = h5f['k_norm'][:]
            damp_rate_CR = h5f['rate'][:]

        compare = np.zeros((len(t_arr), len(omega_x)))

        for i in range(len(t_arr)):
            for j in range(len(omega_x)):
                for h in range(len(k_CR_norm)):
                    if damp_rate_CR[h] > -growth_rate_beam[i][h][j]:
                        compare[i][j] = k_CR_norm[h]
                        break

        with h5py.File(f'{args.savePATH}compare_{z:.1f}_{deltalin0:.1f}.h5', 'w') as h5f:
            h5f.create_dataset('t', data=t_arr, compression="gzip")
            h5f.create_dataset('omega_x', data=omega_x, compression="gzip")
            h5f.create_dataset('compare', data=compare, compression="gzip")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nstep', type=int, help='number of redshift bins', default=399)
    parser.add_argument('--mstep', type=int, help='number of Energy bins', default=399)

    parser.add_argument('--E_min', type=float, help='minimum energy in erg', default=1e-11)
    parser.add_argument('--E_max', type=float, help='maximum energy in erg', default=1e-3)

    parser.add_argument('--source_model', type=str, help="0 for Khaire's model, 1 for Haardt's model", default='0')

    parser.add_argument('--deltalin0', type=float, help='overdensity evolution, 0 for mean density, > 0 for overdensity, and < 0 for underdensity', default=0)

    parser.add_argument('--z', type=float, help='desired redshift', default=2)
    parser.add_argument('--source_file', type=str, help='source file corresponding to the redshift', default='KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 2.0.txt')
    
    parser.add_argument('--T_arr', nargs='+', help='List of time values after incident in second', default=[1e8, 1e9, 1e10, 1e11, 1e12, 1e13])
    parser.add_argument('--k_file', type=str, help='path to normalized k file, if none, using default k', default=None)
    parser.add_argument('--CR_file', type=str, help='path to damp rate by cosmic ray, if None, then no comparison', default=None)

    parser.add_argument('--readPATH', type=str, help='reading path for temperature evolution', default='')
    parser.add_argument('--savePATH', type=str, help='saving path', default='')

    args = parser.parse_args()
    
    main(args)