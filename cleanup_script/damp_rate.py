import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import bisect
import seaborn as sns
import LCDMSphere
import Energy_loss_class
import IGM

"""
This script will generator 2 cvs files and 2 figures: damp_rate_largek_{z}_{deltalin0}(csv and pdf) for k = 0.8 to 5 times omega_p/c, and damp_rate_zoomk_{z}_{deltalin0}(csv and pdf) for k = 0.995 to 1.1 times omega_p/c

run python damp_rate.py --nstep --mstep --E_min --E_max --source_model --deltalin0 --z

the default setting is reshift bins nstep = 399, energy bins mstep = 399, E_min = 1e-11 erg, E_max = 1e-3 erg,

source_model = '0' for Khaire's model ('1' for Haardt's mode),

overdensity evolution deltalin0 = 0 for mean density (> 0 for overdensity, and < 0 for underdensity), redshfit z = 2,

saving path = '', and reading path = '' (should be the same as the saving path of IGM.py)

Please make sure the parameters are consistent with other files
"""

r0 = 2.818e-13 # classical electron radius in cm
m_e = 9.10938e-28 # electron mass in gram
c = 2.998e10 # speed of light in cm/s
h = 6.6261e-27 # planck constant in cm^2 g s-1
k = 1.3808e-16 # boltzmann constant in cm^2 g s^-2 K^-1
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

z_reion_He = 3 # redshift that He is fully ionized
z_reion_H = 8 # redshift that H is fully ionized


def find_idx(z, z_list):
    if z <= z_list[-1]:
        return -1
    idx = np.where(z_list<=z)[0][0]
    if np.abs(z_list[idx] - z) <= np.abs(z_list[idx-1] - z):
        return idx
    else:
        return idx-1


def main(args):
    z = args.z
    deltalin0 = args.deltalin0
    source_model = args.source_model

    IGM_00 = IGM.IGM_N(deltalin0, source_model, args.nstep, args.mstep, args.E_min, args.E_max)

    N = np.load(f'{args.readPATH}N_{deltalin0}_{source_model}.npy')
    
    index_z = find_idx(z, IGM_00.z)

    omega_b = omega_b_0 * (1 + z)**3
    mean_n_b = 3 * omega_b * H_0 * H_0 / (8 * np.pi * G * m_p) # mean number density of baryons at the redshift
    n_b = mean_n_b * IGM_00.Delta[index_z]

    n_H = n_b * (1 - Y_He) # nuclei in hydrogen
    n_He = n_H * f_He # nuclei in helium, assume He is all 4He

    if z <= z_reion_He: # fully ionized
        n_e = n_H + 2 * n_He
    if z_reion_He < z <= z_reion_H: # He is singly ionized
        n_e = n_H + n_He
    if z > z_reion_H: # neutral
        n_e = 0
    
    omega_p = (4 * np.pi * r0 * n_e)**0.5 * c

    def get_rate(k):
        if k<=omega_p/c: return 0
        E_min = E_e * ((1 - omega_p**2 / c**2 / k**2)**(-0.5) - 1)
    
        n_e_th = n_e
        co1 = -np.pi / 4 * omega_p * (omega_p / c / k)**3 * E_e / n_e_th
    
        if E_min < min(IGM_00.E):
            index_E1 = 0
        else:
            index_E1 = np.where(IGM_00.E < E_min)[-1][-1]
        index_E2 = np.where(IGM_00.E > E_min)[0][0]
        N_Emin = (N[index_E2][index_z] - N[index_E1][index_z])/(IGM_00.E[index_E2]-IGM_00.E[index_E1]
                                                     ) * (E_min-IGM_00.E[index_E1]) + N[index_E1][index_z]
    
        int_NEdE = 0.
        for i in range(index_E2, args.mstep):
            E = IGM_00.E[i]
            if i == index_E2:
                NEdE = N[i][index_z] * (E - E_min) / np.sqrt(E * (E + 2 * E_e))
            else:
                NEdE = N[i][index_z] * (E - IGM_00.E[i - 1]) / np.sqrt(E * (E + 2 * E_e))
            int_NEdE += NEdE
        
        co2 = (E_min + E_e) * N_Emin / np.sqrt(E_min * (E_min + 2 * E_e)) + 2 * int_NEdE
    
        if c * k / omega_p - 1 <= 0:
            co3 = 0
        else:
            co3 = 1
        
        return co1 * co2 * co3

    k = omega_p/c * np.linspace(.8,5,1000)
    rate = np.zeros((len(k), ))
    for i in range(len(k)):
        rate[i] = get_rate(k[i])

    plt.plot(k/(omega_p/c), -rate*1e6, color = 'black')
    plt.title('Landau damping by cosmic rays at $z=2$, $\Delta=1$', fontsize = 14)
    plt.xlabel('scaled wave number $ck/\omega_{\mathrm{p}}$', fontsize = 14)
    plt.ylabel('damping rate $-\Im \omega \ [10^{-6}\,\mathrm{s^{-1}}]$', fontsize = 14)
    plt.grid(color = 'green', linestyle = '-.', linewidth = 0.5)
    plt.xlim([.8,5])
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.savefig(f'{args.savePATH}damp_rate_largek_{z}_{deltalin0}.pdf', bbox_inches='tight', pad_inches=0.15)
    plt.clf()

    df = pd.DataFrame({
        'k': k,
        'rate': -rate
        })
    df.to_csv(f'{args.savePATH}damp_rate_largek_{z}_{deltalin0}.csv')

    k = omega_p/c * np.linspace(.995,1.1,1000)
    rate = np.zeros((len(k), ))
    for i in range(len(k)):
        rate[i] = get_rate(k[i])

    plt.plot(k/(omega_p/c), -rate*1e7, color = 'black')
    plt.title('Landau damping by cosmic rays at $z=2$, $\Delta=1$ [zoom]', fontsize = 14)
    plt.xlabel('scaled wave number $ck/\omega_{\mathrm{p}}$', fontsize = 14)
    plt.ylabel('damping rate $-\Im \omega \ [10^{-7}\,\mathrm{s^{-1}}]$', fontsize = 14)
    plt.grid(color = 'green', linestyle = '-.', linewidth = 0.5)
    plt.xlim([.995,1.1])
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.savefig(f'{args.savePATH}damp_rate_zoomk_{z}_{deltalin0}.pdf', bbox_inches='tight', pad_inches=0.15)
    plt.clf()

    df = pd.DataFrame({
        'k': k,
        'rate': -rate
        })
    df.to_csv(f'{args.savePATH}damp_rate_zoomk_{z}_{deltalin0}.csv')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nstep', type=int, help='number of redshift bins', default=399)
    parser.add_argument('--mstep', type=int, help='number of Energy bins', default=399)

    parser.add_argument('--E_min', type=float, help='minimum energy in erg', default=1e-11)
    parser.add_argument('--E_max', type=float, help='maximum energy in erg', default=1e-3)

    parser.add_argument('--source_model', type=str, help="0 for Khaire's model, 1 for Haardt's model", default='0')

    parser.add_argument('--deltalin0', type=float, help='overdensity evolution, 0 for mean density, > 0 for overdensity, and < 0 for underdensity', default=0)

    parser.add_argument('--z', type=float, help='desired redshift', default=2)
    
    parser.add_argument('--savePATH', type=str, help='saving path for the result', default='')
    parser.add_argument('--readPATH', type=str, help='reading path for the result', default='')

    args = parser.parse_args()
    
    main(args)
