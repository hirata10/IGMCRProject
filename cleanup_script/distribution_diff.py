import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from astropy.io import fits
import scipy.sparse
import scipy.linalg
import time
import argparse
import os
from astropy.io import fits

r0 = 2.818e-13 # classical electron radius in cm
m_e = 9.10938e-28 # electron mass in gram
c = 2.998e10 # speed of light in cm/s
h = 6.6261e-27 # planck constant in cm^2 g s-1
eV_to_erg = 1.6022e-12
parsec_to_cm = 3.085677581e18 # cm per parsec

kB = 1.3808e-16 # boltzmann constant in cm^2 g s^-2 K^-1
T_CMB = 2.725 # CMB temperature today in K

H_0 = 2.184e-18 # current hubble constant (67.4 km/s/Mpc) in s^-1
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
e = 2.718281828459 # base of natural log

"""
This script will generate the beam distribution distribution_with_diffusion (dat and pdf), for interplation, please check our paper 

run python damp_rate.py --nstep --mstep --E_min --E_max --source_model --deltalin0 --z --readPATH --savePATH

the default setting is reshift bins nstep = 399, energy bins mstep = 399, E_min = 1e-11 erg, E_max = 1e-3 erg,

source_model = '0' for Khaire's model ('1' for Haardt's mode),

overdensity evolution deltalin0 = 0 for mean density (> 0 for overdensity, and < 0 for underdensity), redshfit z = 2,

saving path = '', and reading path = '' (should be the same as the saving path of IGM.py)

Please make sure the parameters are consistent with other files
"""

# ### generate theta_rms

def compton_f(x):
    if x > 1e-2:
        return ((-24 - 144*x - 290*x**2 - 194*x**3 + 20*x**4 + 28*x**5) / 3 / x**2 / (1+2*x)**3 + (4+4*x-x**2) / x**3 * np.log(1+2*x))
    return 8/5*x**2 - 32/5*x**3 + 2248/105*x**4 - 1376/21*x**5 + 11840/63*x**6


def compton_FF(x):
    n = 1024
    y = np.linspace(x / (2 * n), x - x / (2 * n), n)
    result = 0.
    for i in range(n): result += compton_f(y[i]) * y[i]
    result *= x / n
    return result


def calculate_solid_angles(theta):
    theta_half = (theta[1:] + theta[:-1]) / 2
    theta_half = np.concatenate([[theta[0]/2], theta_half, [3*theta[-1]/2 - theta[-2]/2]])

    Omega = 2 * np.pi * (theta_half[2:]**2 - theta_half[:-2]**2) / 2  # solid angles of annulus
    Omega = np.concatenate([Omega, [2 * np.pi * (theta[-1]**2 - theta_half[-2]**2)]])  # add the last circle

    return Omega, theta_half


def main(args):
    global T_CMB
    z = args.z
    r = args.r
    source_file = args.source_file
    z2 = (1 + z) / 3
    T_CMB *= 1+z # scale to CMB temperature at that redshift

    with open(f'{args.readPATH}rate_z{z:.1f}_{r:.0f}Mpc.pkl', 'rb') as f:
        rata_arr = pkl.load(f)
    
    rate_arr = np.array(rata_arr)
    rate_arr = rate_arr.reshape(400,400) # [i][j]: gamma[i] theta[j], rate_arr[-1] = nan
    rate_trans = np.transpose(rate_arr) # [i][j]: theta[i] gamma[j], rate_trans[i][-1] = nan
    rate_trans[:,-1]=0. # remove nan
    fits.PrimaryHDU(rate_trans).writeto(f'{args.savePATH}rate_trans.fits', overwrite=True)

    gamma_e_arr = np.logspace(8, 14, 400) * eV_to_erg / m_e / c**2 # lab-frame Lorentz factor of the electron produced
    theta_e_arr = np.logspace(-8, 0, 400)

    jtable = np.loadtxt(source_file)
    nstep = np.shape(jtable)[0] - 1
    jtable[:,0] = 2.99792458e18 / jtable[:,0] # convert to Hz
    nu = (jtable[:-1,0] + jtable[1:,0]) / 2
    dnu = (jtable[:-1,0] - jtable[1:,0])
    JnuEBL = (jtable[:-1,1] + jtable[1:,1]) / 2

    xx = h*nu/kB/T_CMB
    Bnu = 2 * h * nu**3 / c**2 / np.expm1(xx)
    Jnu = JnuEBL + Bnu

    n = 400

    prefix_sums = [0 for _ in range(n)] # theta_rms from gamma_jp to gamma_j = np.sqrt(prefix[j] - prefix[jp+1])

    for j in range(n-1, -1, -1):
        all_int = 0
        factors = Jnu / nu**3 * dnu
        for k in range(nstep):
            all_int += compton_FF(2 * gamma_e_arr[j] * h * nu[k] / m_e / c**2) * factors[k]

        tp2 = 135 * m_e * c**2 / 128 / np.pi**4 / gamma_e_arr[j]**3 * (m_e * c**2 / kB / T_CMB)**4 * all_int
        sumi = tp2 / (m_e * c * gamma_e_arr[j])**2 * np.log(gamma_e_arr[1] / gamma_e_arr[0])

        prefix_sums[j] = sumi

        if j + 1 < n:
            prefix_sums[j] += prefix_sums[j + 1]
        print('prefix_sums[{:3d}]={:12.5E} thetarms={:12.5E} at E={:12.5E} eV'.format(j,prefix_sums[j],prefix_sums[j]**.5,gamma_e_arr[j]*5.11e5))
        
    np.save(f"{args.savePATH}prefix_sums.npy", prefix_sums)

    Omega, theta_half = calculate_solid_angles(theta_e_arr)

    M = np.zeros((400, 400))

    for i in range(400):
        for j in range(400):
            if j == i + 1:
                M[i, j] = -2 * np.pi / Omega[i] * theta_half[i+1] / (theta_e_arr[i+1] - theta_e_arr[i])
            if j == i - 1:
                M[i, j] = -2 * np.pi / Omega[i] * theta_half[i] / (theta_e_arr[i] - theta_e_arr[i-1])
    for i in range(1,399):
        M[i, i] = 2 * np.pi / Omega[i] * (theta_half[i+1] / (theta_e_arr[i+1] - theta_e_arr[i]) + theta_half[i] / (theta_e_arr[i] - theta_e_arr[i-1]))

    M[0,0] = -M[1,0]*Omega[1]/Omega[0]
    M[-1,-1] = -M[-2,-1]*Omega[-2]/Omega[-1]
    fits.PrimaryHDU(M).writeto(f'{args.savePATH}M-matrix.fits', overwrite=True)

    # ### new element of columb
    F_NP = np.loadtxt(f'{args.readPATH}fnpk_z{z}.dat')

    dk = np.array([F_NP[:, 0][i] - F_NP[:, 0][i-1] for i in range(1, 301)])
    dk = np.insert(dk, 0, 0)

    k_cut = F_NP[:, 0][-1]

    omega_b = omega_b_0 * (1 + z)**3
    mean_n_b = 3 * omega_b * H_0 * H_0 / (8 * np.pi * G * m_p) # mean number density of baryons at the redshift
    n_b = mean_n_b # for mean density
    
    n_H = n_b * (1 - Y_He) # nuclei in hydrogen
    n_He = n_H * f_He # nuclei in helium, assume He is all 4He

    n_e = n_H + 2 * n_He # for fully ionized

    nsp = 3 # number of species
    mass_sp = np.array([m_e, m_p, 4*m_p]) # masses
    nth_sp = np.array([n_e,n_H,n_He]) # number densities
    Z_sp = np.array([-1.,1.,2.]) # charges

    Z_proj = 1

    # engineering of M-matrix to get stable exp calculation
    P = np.diag(np.sqrt(1/Omega))
    L = np.linalg.inv(P)@M@P
    eigenvalues_L, eigenvectors_L = np.linalg.eigh(L)
    R_L = eigenvectors_L.copy()
    lambda_l = eigenvalues_L.copy()

    eigenvalues, eigenvectors = np.linalg.eig(M)
    l = np.sqrt(eigenvalues)
    V = eigenvectors.copy()
    V_inv = np.linalg.inv(V)

    t_arr = args.T_arr
    t_num = len(t_arr)

    Omega_x = np.logspace(-8,0, 400).reshape(400, 1)
    
    def T_vectorized(gamma, l): # eq 37
        p_proj = m_e * c * gamma
        v_proj = c / gamma
        co = 2 * np.pi * l**2 * Z_proj**2 * e**4 / (p_proj**2 * v_proj)
    
        ## vectorize
        gamma_E = np.euler_gamma # Euler's constant
        sum_nsp = np.zeros(gamma.shape) # Initialize sum_nsp as an array of zeros with the same shape as gamma
    
        for a in range(nsp):
            int_FNP = np.sum(dk * F_NP[:, 1] / F_NP[:, 0] + 1 - gamma_E - np.log(h/2/np.pi * k_cut * l / p_proj[:, np.newaxis]), axis=1)
            sum_isp = Z_sp[a]**2 * nth_sp[a] * int_FNP
            sum_nsp += sum_isp
    
        return co * sum_nsp


    def T_ij(l, i ,j): # eq 39 bracket part
        Gamma_IC0 = 1.1e-18 * z2**4
        gamma_ij = gamma_e_arr[i:j+1]
        d_gamma_e = np.insert(np.diff(gamma_ij), 0, 0)
        Tl = T_vectorized(gamma_ij, l)
        #print(Tl.shape, d_gamma_e.shape, gamma_ij.shape)
        return -np.sum(Tl * d_gamma_e / Gamma_IC0 / gamma_ij**2)


    def get_nij(i, j, t, M_expm): # nij = dN / (dV dgammae_j dOmegae_i)
        Gamma_IC0 = 1.1e-18 * z2**4
        gamma_max_inv = 1. / gamma_e_arr[j] - Gamma_IC0 * t
        if gamma_max_inv>1e-99:
            gamma_max = 1./gamma_max_inv
        else:
            gamma_max = 1e99
        
        nij = 0
    
        for jp in range(j, 399):
            if gamma_e_arr[jp] <= gamma_max:
                # THIS IS A COMMENT: rate_trans_smoothed = M_expm @ rate_trans
                # Then, rate_trans_smoothed[i][jp] is the dot product of M_expm[jp][i] and rate_trans[:, jp]
                rate_trans_smoothed_ijp = np.dot(M_expm[jp, i, :], rate_trans[:, jp])
                
                sum_ij = (gamma_e_arr[jp] - gamma_e_arr[jp-1]) * rate_trans_smoothed_ijp
            
                if gamma_e_arr[jp+1] > gamma_max:
                    fraction = (gamma_max - gamma_e_arr[jp]) / (gamma_e_arr[jp+1] - gamma_e_arr[jp])
                    sum_ij = sum_ij * fraction
                nij += sum_ij
        nij = nij / (Gamma_IC0 * gamma_e_arr[j]**2)
        return nij


    def get_ni(i, nij): # integrate gamma_e (1/gamma_e) nij
        ni = 0
        for j in range(1, 400):
            sum_i = (gamma_e_arr[j] - gamma_e_arr[j-1]) / gamma_e_arr[j] * nij[i][j]
            ni += sum_i
        return ni

    nij = np.zeros((t_num, 400, 400))
    time_start = time.time()
    # for k in range(t_num):
    #     print(f'time {k} {t_arr[k]:.1e}', end = ' ')
    
    for j in range(1, 400):
        print(f'j = {j}, time = {(time.time()-time_start)/60}')
        M_new = np.zeros((400, 400, 400))
        for jp in range(j, 399):
            theta_rms = np.sqrt(prefix_sums[j] - prefix_sums[jp+1]) # from jp to j
            #M_expm = scipy.linalg.expm(-theta_rms**2/4 * M)
            M_expm = P@R_L@np.diag(np.exp(-theta_rms**2/4*lambda_l))@R_L.T@np.linalg.inv(P)
            
            ### new element for columb scattering
            new_EV = np.zeros((len(l)))
            for x in range(len(l)):
                new_EV[x] = np.exp(T_ij(l[x], j, jp)) # from jp to j, eq 39 exp part
            new_D = np.diag(new_EV)
            columb = np.dot(np.dot(V, new_D), V_inv)
            
            M_new[jp] = M_expm @ columb
            
            if np.isnan(M_new[jp,:,:].any()): print('test', j, jp, np.isnan(M_new[jp,:,:].any()))

        for k in range(t_num):
            for i in range(400):
                nij[k][i][j-1] = get_nij(i, j, t_arr[k], M_new)
    print('t = ', (time.time() - time_start) / 60)

    ## save data
    fits.PrimaryHDU(nij).writeto(f'{args.savePATH}nij.fits', overwrite=True)

    ni = np.zeros((t_num, 400)) # about theta
    for k in range(t_num):
        for i in range(400):
            ni[k][i] = get_ni(i, nij[k])

    mc = np.zeros((400, t_num+1))
    mc[:,0] = theta_e_arr
    mc[:,1:] = ni.T

    deriv_array = np.zeros((400,t_num))
    for j in range(399):
        for jp in range(j,399):
            if j==jp:
                tr = 0.
            else:
                tr = np.arccosh(theta_e_arr[jp]/theta_e_arr[j])
            deriv_array[j,:] += 2 * theta_e_arr[j] * (ni[:,jp+1]-ni[:,jp])/(theta_e_arr[jp+1]-theta_e_arr[jp])          * (np.arccosh(theta_e_arr[jp+1]/theta_e_arr[j]) - tr)
    mc2 = np.zeros((400, t_num+1))
    mc2[:,0] = theta_e_arr
    mc2[:,1:] = deriv_array
    np.savetxt(f'{args.savePATH}distribution_with_diffusion.dat', mc2)

    for i in range(t_num):
        plt.plot(Omega_x, -mc2[:, 1+i], label=f'{t_arr[i]:.1e}')
    
    plt.legend(loc = 'upper right')
    plt.xlim([1e-8, 1e-3])
    plt.ylim([1e-24, 1e-16])
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(f'{args.savePATH}distribution_with_diffusion.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nstep', type=int, help='number of redshift bins', default=399)
    parser.add_argument('--mstep', type=int, help='number of Energy bins', default=399)

    parser.add_argument('--E_min', type=float, help='minimum energy in erg', default=1e-11)
    parser.add_argument('--E_max', type=float, help='maximum energy in erg', default=1e-3)

    #parser.add_argument('--source_model', type=str, help="0 for Khaire's model, 1 for Haardt's model", default='0')

    parser.add_argument('--deltalin0', type=float, help='overdensity evolution, 0 for mean density, > 0 for overdensity, and < 0 for underdensity', default=0)

    parser.add_argument('--z', type=float, help='desired redshift', default=2)
    parser.add_argument('--r', type=float, help='distance to the blazar in Mpc', default=10)
    parser.add_argument('--source_file', type=str, help='source file corresponding to the redshift', default='KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 2.0.txt')
    
    parser.add_argument('--T_arr', nargs='+', help='List of time values after incident in second', default=[1e8, 1e9, 1e10, 1e11, 1e12, 1e13])
    
    parser.add_argument('--readPATH', type=str, help='reading path of attenuation coefficient', default='')
    parser.add_argument('--savePATH', type=str, help='saving path for the result', default='')

    args = parser.parse_args()
    
    main(args)
