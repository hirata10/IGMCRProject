import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root_scalar
import scipy.special as sp
import pandas as pd
import argparse
import time
import IGM

"""
This script will generator 2 npy files: damp rate under a pre-exist intergalatic magnetic field 'damp_rate_IGMF.npy' (row index for wavenumber k, and column index for angle), and the corresponding 'k_IGMF.npy'

run python IGM.py --nstep --mstep --E_min --E_max --deltalin0 --z --source_file --angle_arr --readPATH --savePATH

the default setting is reshift bins nstep = 399, energy bins mstep = 399, E_min = 1e-11 erg, E_max = 1e-3 erg,

overdensity evolution deltalin0 = 0 for mean density (> 0 for overdensity, and < 0 for underdensity), redshift z = 2, corresponding source_file = KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 2.0.txt,

reading path = '', and saving path = ''

Please make sure the parameters are consistent with other files
"""

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
    z2 = (1 + z) / 3
    T_CMB *= 1 + z 
    deltalin0 = args.deltalin0

    df = pd.read_csv(f'{args.readPATH}temp_history_{deltalin0}.csv')
    length = len(df[df['z']>=z])
    T_IGM = df[df['z']>=z]['T'][length-1]

    omega_b = omega_b_0 * (1 + z)**3
    mean_n_b = 3 * omega_b * H_0 * H_0 / (8 * np.pi * G * m_p) # mean number density of baryons at the redshift
    n_b = mean_n_b # for mean density
    
    n_H = n_b * (1 - Y_He) # nuclei in hydrogen
    n_He = n_H * f_He # nuclei in helium, assume He is all 4He

    n_e = n_H + 2 * n_He # for fully ionized

    omega_p = c * np.sqrt(4 * np.pi * r0 * n_e)
    Npoints = 100
    k = omega_p/c*np.exp(.01*np.sinh(7.5*(np.linspace(0,1,Npoints)-.3)))
    theta_deg = np.array(args.angle_arr)
    theta_rad = np.radians(theta_deg)

    n = n_e + n_H + n_He

    B = np.sqrt(8 * np.pi * kB * T_IGM * n) # equipartion field

    omega_c = e * B / (m_e * c)

    def dispersion(omega, k, theta_rad):
        res = omega**2 - omega_p**2 / (1 - omega_c**2 / omega**2) - c**2 * k**2 * np.cos(theta_rad)**2 - 1 / (1 - omega_c**2 / omega**2) * omega_c**2 * omega_p**4 / ((omega**2 - c**2 * k**2) * (omega**2 - omega_c**2) - omega_p**2 * omega**2) - c**4 * k**4 * np.sin(theta_rad)**2 * np.cos(theta_rad)**2 / (omega**2 - omega_p**2 - c**2 * k**2 * np.sin(theta_rad)**2)
        return res

    omega = np.zeros((len(k), len(theta_rad)))

    for i in range(len(k)):
        for j in range(len(theta_rad)):
            dispersion_fixed = lambda omega: dispersion(omega, k[i], theta_rad[j])
            omega[i][j] = root_scalar(dispersion_fixed, method='bisect', bracket=[.998*omega_p, 1.002*omega_p]).root

    def get_eigen(omega, k, theta_rad):
        zeta_x = np.sin(theta_rad) # for now, as zeta will be canceled out later
        zeta_y = 1j * omega_c * omega_p**2 * omega / ((omega**2 - c**2 * k**2) * (omega**2 - omega_c**2) - omega_p**2 * omega**2) * zeta_x
        zeta_z = -c**2 * k**2 * np.sin(theta_rad) * np.cos(theta_rad) / (omega**2 - omega_p**2 - c**2 * k**2 * np.sin(theta_rad)**2) * zeta_x
        return zeta_x, zeta_y, zeta_z

    def get_D(i, j):
        omega_ij = omega[i][j]
        zeta_x, zeta_y, zeta_z = get_eigen(omega_ij, k[i], theta_rad[j])
        mag2_zeta = zeta_x**2 + np.abs(zeta_y)**2 + zeta_z**2
    
        add1 = 2 * omega_ij * mag2_zeta
        co1 = omega_p**2 * omega_c**2 / (omega_ij**3 * (1 - omega_c**2 / omega_ij**2)**2)
        co2 = 2 * (zeta_x**2 + np.abs(zeta_y)**2) + 2*zeta_x*zeta_y.imag / omega_c * omega_ij * (1 + omega_c**2 / omega_ij**2)
        return add1 + co1 * co2 # co2 is always xxx + 0j, only real part make sense

    D = np.zeros((len(k), len(theta_rad)))
    for i in range(len(k)):
        for j in range(len(theta_rad)):
            D[i][j] = get_D(i, j)

    IGM_00 = IGM.IGM_N(deltalin0, '0', args.nstep, args.mstep, args.E_min, args.E_max) # mean density
    print('getting number density of cosmic ray electron')
    n_CRe = IGM_00.back_Euler()
    print('finish')
    E = IGM_00.E
    gamma = E / E_e + 1

    v = c * np.sqrt(E * (E + 2 * E_e)) / (E + E_e) # shape (mstep+1, )
    p = gamma * m_e * v # shape (mstep+1, )
    omega_s = omega_c / gamma # shape (mstep+1, )

    z_idx = find_idx(z, IGM_00.z)

    n_CRe_z2 = n_CRe[:, z_idx] # number density of CRe at z, shape (mstep,)
    fCR = n_CRe_z2/(IGM_00.E_plus-IGM_00.E_minus)[1:,0]/4./np.pi/p[1:]/(m_e+E[1:]/c**2)

    n_CRe_z2 = np.insert(n_CRe_z2, 0, 0) # insert 0 at beginning for calculation, shape (msteo+1, )
    fCR = np.insert(fCR, 0, 0) # insert 0 at beginning for calculation, shape (mstep+1, )

    def get_n_Bessel(omega, k_para, omega_s, v_):
        # cos(alpha) = (omega - n * omega_s) / k_parallel / v
        n_b1 = (omega - k_para * v_) / omega_s
        n_b2 = (omega + k_para * v_) / omega_s
    
        lower_bound = int(min(n_b1, n_b2)) + 1
        upper_bound = int(max(n_b1, n_b2))
    
        return np.array(range(lower_bound, upper_bound + 1))

    def get_Bessel(omega, n_Bessel, omega_s, k_para, k_orth, k, theta_rad, v_):
        cosa = (omega - n_Bessel * omega_s) / k_para / v_
        sina = np.sqrt(1 - cosa**2)
    
        z_Bessel = k_orth * v_ * sina / omega_s
        Jnz = sp.jv(n_Bessel, z_Bessel)
        Jnz_p = sp.jvp(n_Bessel, z_Bessel, n=1)
    
        zeta_x, zeta_y, zeta_z = get_eigen(omega, k, theta_rad)
    
        res_v = n_Bessel / z_Bessel * Jnz * zeta_x * sina + Jnz * zeta_z * cosa + Jnz_p * zeta_y / 1j * sina
        res = res_v.real**2
    
        return res

    def get_int(i, j):
        k_para = k[i] * np.cos(theta_rad[j])
        k_orth = k[i] * np.sin(theta_rad[j])
    
        int_sum = 0
        for pi in range(len(p)-1):
            Delta = omega_s[pi] / (np.abs(k_para) * c)
            co1 = p[pi]**2 * (fCR[pi+1] - fCR[pi]) * m_e * c * Delta / omega_s[pi]
        
            n_Bessel = get_n_Bessel(omega[i][j], k_para, omega_s[pi], v[pi])
            if np.size(n_Bessel)>0:
                sum_Bessel = sum(get_Bessel(omega[i][j], ni_Bessel, omega_s[pi], k_para, k_orth, k[i], theta_rad[j], v[pi]) for ni_Bessel in n_Bessel)
                int_sum = int_sum + co1 * sum_Bessel
                print(pi, co1 * sum_Bessel, k[i]*v[pi]/omega[i,j], k_orth*v[pi]/omega_s[pi], n_Bessel.shape, n_Bessel[0], n_Bessel[-1])
        
        return int_sum

    def get_damp_rate(i, j):
        res = 2 * np.pi**2 * omega_p**2 * omega[i][j] / (D[i][j] * n_e) * get_int(i,j)
        return res

    damp_rate = np.zeros((len(k), len(theta_rad))) # row index for k, column index for angle
    np.save(f'{args.savePATH}k_IGMF.npy', k)
    time_start = time.time()
    print('start generating damp rate')
    for j in range(len(theta_rad)):
        for i in range(len(k)):
            print(f'i = {i}, j = {j}, {(time.time() - time_start)/60}')
            damp_rate[i][j] = get_damp_rate(i, j)
            np.save(f'{args.savePATH}damp_rate_IGMF.npy', damp_rate)
    print('finsh')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nstep', type=int, help='number of redshift bins', default=399)
    parser.add_argument('--mstep', type=int, help='number of Energy bins', default=399)

    parser.add_argument('--E_min', type=float, help='minimum energy in erg', default=1e-11)
    parser.add_argument('--E_max', type=float, help='maximum energy in erg', default=1e-3)

    parser.add_argument('--deltalin0', type=float, help='overdensity evolution, 0 for mean density, > 0 for overdensity, and < 0 for underdensity', default=0)

    parser.add_argument('--z', type=float, help='desired redshift', default=2)
    parser.add_argument('--source_file', type=str, help='source file corresponding to the redshift', default='KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 2.0.txt')
    
    parser.add_argument('--angle_arr', nargs='+', help='List of angle in degree', default=[5, 45, 80])

    parser.add_argument('--readPATH', type=str, help='reading path for temperature evolution', default='')
    parser.add_argument('--savePATH', type=str, help='saving path for damp rate', default='')

    args = parser.parse_args()
    
    main(args)