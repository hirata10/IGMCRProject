import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time
import IGM
import temp_history

"""
This script will generator files: 'phasespace_e_z{z}.dat', 'allgfunc_{deltalin0}_{z}.dat', and 'fnpk_z{z}.dat'

run python F_NP.py --nstep --mstep --E_min --E_max --source_model --deltalin0 --z --source_file

the default setting is reshift bins nstep = 399, energy bins mstep = 399, E_min = 1e-11 erg, E_max = 1e-3 erg,

source_model = '0' for Khaire's model ('1' for Haardt's mode),

overdensity evolution deltalin0 = 0 for mean density (> 0 for overdensity, and < 0 for underdensity),

redshift z = 2, corresponding source_file = 'KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 2.0.txt',

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
e = 4.80326e-10 # electron charge in esu
Ry = m_e*c**2/137.06**2/2. # Rydberg energy

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


def D_R(y, upsilon = .01):
    upsilon_scale = upsilon * (2*y/(1+y**2))**2
    return 1 + .5 / y * np.log(np.sqrt((1 - y)**2 + upsilon_scale**2) / (1 + y))


def D_I(y):
    return -np.pi/2/y*np.heaviside(y-1,.5)


def main(args):
    global T_CMB
    z = args.z
    deltalin0 = args.deltalin0
    source_model = args.source_model

    IGM_ = IGM.IGM_N(deltalin0, source_model, args.nstep, args.mstep, args.E_min, args.E_max)
    z_idx = find_idx(z, IGM_.z)

    z2 = (1 + z) / 3
    T_CMB *= 1 + z

    df = pd.read_csv(f'{args.readPATH}temp_history_{deltalin0}.csv')
    length = len(df[df['z']>=z])
    T_IGM = df[df['z']>=z]['T'][length-1]
    # rec coefficients 1991A&A...251..680P
    alpha_H = 5.596e-13*(T_IGM/1e4)**(-.6038)/(1. + 0.3436*(T_IGM/1e4)**0.4479)
    alpha_He = 5.596e-13*2*(T_IGM/4e4)**(-.6038)/(1. + 0.3436*(T_IGM/4e4)**0.4479)

    N = np.load(f'{args.readPATH}N_{deltalin0}_{source_model}.npy')
    dNdE_z2 = N[:,z_idx]
    E_z2 = IGM_.E[:np.size(dNdE_z2)]
    p_grid = np.sqrt(E_z2*(2*m_e+E_z2/c**2))
    fp_grid = dNdE_z2/(4.*np.pi*p_grid*(m_e+E_z2/c**2))
    slope_fp_grid = np.log(fp_grid[1:]/fp_grid[:-1])/np.log(p_grid[1:]/p_grid[:-1])

    omega_b = omega_b_0 * (1 + z)**3
    mean_n_b = 3 * omega_b * H_0 * H_0 / (8 * np.pi * G * m_p) # mean number density of baryons at the redshift
    n_b = mean_n_b # for mean density
    
    n_H = n_b * (1 - Y_He) # nuclei in hydrogen
    n_He = n_H * f_He # nuclei in helium, assume He is all 4He

    if z <= z_reion_He: # fully ionized
        n_e = n_H + 2 * n_He
    if z_reion_He < z <= z_reion_H: # He is singly ionized
        n_e = n_H + n_He
    if z > z_reion_H: # neutral
        n_e = 0

    rate_ion_H = alpha_H * n_e * n_H
    rate_ion_He = alpha_He * n_e * n_He

    # get spectra from file
    bgspec = np.loadtxt(args.source_file)[::-1,:]
    bgspec_E = h*c/(bgspec[:,0]*1e-8)
    bgspec_J = bgspec[:,1]

    def sigpi(Z):
        npr2inv = bgspec_E/(Z**2*Ry)-1.
        npr = npr2inv**(-.5)
        X2 = np.where(npr2inv>0, np.exp(4.-4*npr*np.arctan(1/npr))/(1.-np.exp(-2*np.pi*npr))/(1.+npr2inv)**5,1.)
        return X2

    sigma_pi = [0, sigpi(1), sigpi(2)]

    # rate of photoionization producing electrons at momentum > p_e
    # (in cm^-3 s^-1)
    def rate_gt_E(p_e):
        # loop over hydrogen and helium
        tot_rate = np.zeros(np.size(p_e)+1)
        p = np.zeros(np.size(p_e)+1)
        p[:-1] = p_e # append 0 so we have the total rate
        Np = 100
        factor = 1.e2 # Emax/Emin
        e__emin = factor**np.linspace(.5/Np,1-.5/Np,Np)
        for Z in [1,2]:
            eph0 = p**2/2/m_e + Z**2*Ry
            for ie in range(Np):
                eph = eph0*e__emin[ie]
                deph = eph*np.log(factor)/Np
                this_sigma = np.interp(eph,bgspec_E,sigma_pi[Z],right=0.)
                this_rate = this_sigma*deph/eph # contribution to the rate between eph[i] and eph[i+1]
            this_rate /= this_rate[-1]
            if Z==1: tot_rate = tot_rate + rate_ion_H*this_rate
            if Z==2: tot_rate = tot_rate + rate_ion_He*this_rate
        return tot_rate[:-1]

    # dynamical friction
    def loss_dpdt(p_e):
        # formula: -dp/dt = 4 pi n_e e^4 m_e / p_e^2 * ln Lambda
        # Lambda = rmax/rmin = 1/(kD*rmin)
        kD = np.sqrt(4.*np.pi*e**2/kB/T_IGM*(n_e+n_H+4*n_He))
        rmin = e**2/(p_e**2/(2*m_e))
        return 4*np.pi*n_e*e**4*m_e/p_e**2*np.log(1/kD/rmin)

    # phase space density
    def fp_photoionization(p_e):
        return rate_gt_E(p_e)/(4.*np.pi*p_e**2*loss_dpdt(p_e))

    p_e__ = m_e*c*np.logspace(-4,4,81)
    rateg = rate_gt_E(p_e__)
    pdot = loss_dpdt(p_e__)
    fpi = fp_photoionization(p_e__)
    print(f'# rate_ion_H = {rate_ion_H}, rate_ion_He = {rate_ion_He}')
    for i in range(81):
        print('{:12.5E} {:12.5E} {:12.5E} {:12.5E}'.format(p_e__[i],p_e__[i]**2/(2*m_e)/eV_to_erg,rateg[i],fpi[i]))

    # species index i = 0 (e), 1 (p), 2 (alpha)
    nsp = 3 # number of species
    mass_sp = np.array([m_e, m_p, 4*m_p]) # masses
    nth_sp = np.array([n_e,n_H,n_He]) # number densities
    Z_sp = np.array([-1.,1.,2.]) # charges

    # gamma determined from momentum p (g cm/s) and index i
    def gamma_from_p(p,i):
    	return(np.sqrt(1. + (p/(c*mass_sp[i]))**2))

    def v_from_p(p,i):
    	return(p/mass_sp[i]/np.sqrt(1. + (p/(c*mass_sp[i]))**2))

    # phase space density
    def f0_from_p(p,i,return_all=False):
        x = p/np.sqrt(2*kB*T_IGM*mass_sp[i])
        fth = nth_sp[i]*(2*np.pi*kB*T_IGM*mass_sp[i])**(-1.5)*np.exp(-x*x)
        ftot = np.copy(fth)
        if i==0:
            nExt = 60
            p_grid2 = np.zeros((np.size(p_grid)+nExt))
            fp_grid2 = np.zeros((np.size(p_grid)+nExt))
            p_grid2[nExt:] = p_grid
            fp_grid2[nExt:] = fp_grid
            p_th = np.sqrt(2*kB*T_IGM*m_e)
            if p_th>=.5*p_grid[0]: p_th=.5*p_grid[0]
            p_grid2[:nExt] = p_th * (p_grid[0]/p_th)**np.linspace(0,1,nExt+1)[:-1]
            fp_grid2[:nExt] = fp_grid[0]*(p_grid2[:nExt]/p_grid[0])**(-2.)
            fcr = np.interp(p,p_grid2,fp_grid2,right=0.)
            fpi = fp_photoionization(p)
            ftot = fth + fcr + fpi
        if return_all and i==0: return ftot,fth,fcr,fpi
        return(ftot)

    def df0dp_from_p(p,i,dpp=.01):
        return((f0_from_p(p*(1+dpp),i)-f0_from_p(p*(1-dpp),i))/(2*p*dpp))

    p = np.logspace(-22,-12,1001) # this was for test only
    f1tot, f1th, f1cr, f1pi = f0_from_p(p,0,return_all=True)
    np.savetxt(f'{args.savePATH}phasespace_e_z{z}.dat', np.stack((p, (gamma_from_p(p,0)-1)*m_e*c**2/eV_to_erg,
        f0_from_p(p,0), f1th, f1cr, f1pi, df0dp_from_p(p,0))).T)

    def gamma(v_p):
        if abs(v_p**2 / c**2 - 1) <= 1e-28:
            return 1e14
        if v_p > c:
            #print(v_p)
            return 1e14
        return 1 / np.sqrt(1 - v_p**2 / c**2)

    def f0(v_p):
        return np.sqrt(m_e / (2 * np.pi * kB * T_CMB)) * np.exp(-m_e * v_p**2 / (2 * kB * T_CMB))

    def df0(v_p):
        return -2 * m_e * v_p / (2 * kB * T_CMB) * f0(v_p)

    v_p_max = c

    def G_a_cons(v_ph, isp=0): # conservative response
        p = c * mass_sp[isp] * np.logspace(-11,7,18001)
        dlnp = np.log(p[1]/p[0])
        v_p = v_from_p(p,isp)
        int_p = dlnp*np.sum( D_R(v_p/v_ph) * p * gamma_from_p(p,isp) * df0dp_from_p(p,isp) * p )
        # *p at the end is for integration measure
        return 4 * np.pi * mass_sp[isp]**2 * v_ph**2 / nth_sp[isp] * int_p

    def g_a_dist(v_ph, isp=0): # projected distribution
        # exit if phase velocity is too large, nothing resonant
        if 1.-v_ph**2 / c**2 <= 1e-14:
            return 0.
        gamma_ph = 1 / np.sqrt(1 - v_ph**2 / c**2)
        p_min = mass_sp[isp] * v_ph * gamma_ph
        v_p_min = p_min / m_e
        p = p_min * np.logspace(0.0005,9.9995,10000)
        dlnp = np.log(p[1]/p[0])
        int_p = dlnp*np.sum( f0_from_p(p,isp) * p * gamma_from_p(p,isp) * p )
        # *p at the end is for integration measure
        return 4 * np.pi * mass_sp[isp] / nth_sp[isp] * int_p

    def g_a_damp(v_ph, isp=0): # damping response
        # exit if phase velocity is too large, nothing resonant
        if 1.-v_ph**2 / c**2 <= 1e-14:
            return 0.
        gamma_ph = 1 / np.sqrt(1 - v_ph**2 / c**2)
        p_min = mass_sp[isp] * v_ph * gamma_ph
        v_p_min = p_min / m_e
        p = p_min * np.logspace(0.0005,9.9995,10000)
        dlnp = np.log(p[1]/p[0])
        int_p = dlnp*np.sum( df0dp_from_p(p,isp)/v_from_p(p,isp) * p * gamma_from_p(p,isp) * p )
        # *p at the end is for integration measure
        return -4 * np.pi * mass_sp[isp] / nth_sp[isp] * int_p

    Z_e = 1
    def G_cons(v_ph):
        tot = 0.
        for isp in range(nsp):
            tot += 4 * np.pi * e**2 * Z_sp[isp]**2 * nth_sp[isp] / mass_sp[isp] * G_a_cons(v_ph,isp)
        return tot

    def g_dist(v_ph):
        tot = 0.
        den = 0.
        for isp in range(nsp):
            tot += Z_sp[isp]**2 * nth_sp[isp] * g_a_dist(v_ph,isp)
            den += Z_sp[isp]**2 * nth_sp[isp]
        return(tot/den)

    def g_damp(v_ph):
        tot = 0.
        for isp in range(nsp):
            tot += 2 * np.pi**2 * e**2 * v_ph * Z_sp[isp]**2 * nth_sp[isp] * g_a_damp(v_ph,isp)
        return(tot)

    print('all g functions')
    for j in range(81):
        v_ph = 10.**(3+.1*j)
        print('{:12.5E} {:12.5E} {:12.5E} {:12.5E}'.format(v_ph, G_cons(v_ph), g_dist(v_ph), g_damp(v_ph)))

    # output file
    # columns: v_ph; Gcons e,p,alpha; Gcons; gdist e,p,alpha; gdist; gdamp e,p,alpha; gdamp
    allgfunc = np.zeros((801,13))
    for j in range(801):
        v_ph = 10.**(3+.01*j)
        allgfunc[j,0] = v_ph
        for isp in range(3):
            allgfunc[j,1+isp] = 4 * np.pi * e**2 * Z_sp[isp]**2 * nth_sp[isp] / mass_sp[isp] * G_a_cons(v_ph,isp)
        allgfunc[j,4] = G_cons(v_ph)
        for isp in range(3):
            allgfunc[j,5+isp] = Z_sp[isp]**2 * nth_sp[isp] * g_a_dist(v_ph,isp) / np.sum(Z_sp**2*nth_sp)
        allgfunc[j,8] = g_dist(v_ph)
        for isp in range(3):
            allgfunc[j,9+isp] = 2 * np.pi**2 * e**2 * v_ph * Z_sp[isp]**2 * nth_sp[isp] * g_a_damp(v_ph,isp)
        allgfunc[j,12] = g_damp(v_ph)
    np.savetxt(f'{args.savePATH}allgfunc_{deltalin0}_{z}.dat', allgfunc)

    def F_NP(k_perp):
        v_proj = c
        Nv = 4001
        v_ph = np.logspace(0, np.log10(v_proj), Nv)
    
        # we want to do the integral in a way that properly samples the resonance at low k
        # if we integrate a function of the form int_{v_i}^{v_{i+1}} C1/(C2^2+C3^2) dv
        # where the C's vary slowly but C2 may pass through zero, we write C1,C3 = const, C2 = linear fcn of v
        # then this leads to (for C3>0)
        # C1 / C3 * ( tan^-1(C2/C3)[i+1] - tan^-1(C2/C3)[i] ) / (dC2/dv)
        # but this is numerically unstable if C3 is very small
        int_v_ph = 0
        C1 = np.zeros((Nv,))
        C2 = np.zeros((Nv,))
        C3 = np.zeros((Nv,))
        for i in range(Nv):
            co_ph = 1 - v_ph[i]**2 / v_proj**2
            C1[i] = co_ph * g_dist(v_ph[i])
            C2[i] = 1 - k_perp**(-2) * v_ph[i]**(-2) * co_ph * G_cons(v_ph[i])
            C3[i] = k_perp**(-2) * co_ph * g_damp(v_ph[i])
        C3 = C3 + 1e-200 # <-- forces a floor to the damping to get rid of division by zero
            # (shouldn't be necessary when we include CRs)
        int_v_ph = 0.
        for i in range(Nv-1):
            is_nonres = True
            if C2[i]*C2[i+1]<0: is_nonres=False
            if i<Nv-2:
                if C2[i+1]*C2[i+2]<0: is_nonres=False
            if i>0:
                if C2[i-1]*C2[i]<0: is_nonres=False
            if is_nonres:
                int_v_ph += (C1[i]+C1[i+1])/2/( ((C2[i]+C2[i+1])/2)**2+((C3[i]+C3[i+1])/2)**2 )*(v_ph[i+1]-v_ph[i])
            else:
                #print('res', i, v_ph[i:i+2], C1[i:i+2], C2[i:i+2], C3[i:i+2])
                int_v_ph += (C1[i]+C1[i+1])/(C3[i]+C3[i+1]) * (np.arctan2(C2[i+1],C3[i+1])-np.arctan2(C2[i],C3[i]))/\
                    (C2[i+1]-C2[i])*(v_ph[i+1]-v_ph[i])
            # co1 / (co2 + co3) * (v_ph[i+1] - v_ph[i-1])/2. # latter is effective dv_ph
        return int_v_ph

    k_perp = np.logspace(-9, -3, 301)
    k_D = np.sqrt(4.*np.pi*e**2/kB/T_IGM*np.sum(nth_sp*Z_sp**2))
    print('K_D = ', k_D)

    F_NP_K = np.zeros((np.size(k_perp), ))
    time_start = time.time()
    print('Start generating F_NP')
    for i in range(np.size(k_perp)):
        F_NP_K[i] = F_NP(k_perp[i])
        #print(f'i = {i}, F_NP_Ki = {F_NP_K[i]}')
        print('{:12.5E} {:12.5E}'.format(k_perp[i],F_NP_K[i]))
    np.savetxt(f'{args.savePATH}fnpk_z{z}.dat', np.stack((k_perp,F_NP_K)).T)
    print(f'Finish, time={(time.time() - time_start)/60} mins')


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
    
    parser.add_argument('--readPATH', type=str, help='reading path for temperature evolution', default='')
    parser.add_argument('--savePATH', type=str, help='saving path', default='')

    args = parser.parse_args()
    
    main(args)