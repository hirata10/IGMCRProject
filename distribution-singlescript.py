import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from astropy.io import fits

r0 = 2.818e-13 # classical electron radius in cm
m_e = 9.10938e-28 # electron mass in gram
c = 2.998e10 # speed of light in cm/s
h = 6.6261e-27 # planck constant in cm^2 g s-1
eV_to_erg = 1.6022e-12
parsec_to_cm = 3.085677581e18 # cm per parsec

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
e = 2.718281828459 # base of natural log

z = 2
z2 = (1 + z) / 3

## Cell 2: load the arrays

with open('rate_10Mpc.pkl', 'rb') as f:
    rata_arr = pkl.load(f)
    
rate_arr = np.array(rata_arr)
rate_arr = rate_arr.reshape(400,400) # [i][j]: gamma[i] theta[j], rate_arr[-1] = nan
rate_trans = np.transpose(rate_arr) # [i][j]: theta[i] gamma[j], rate_trans[i][-1] = nan

gamma_e_arr = np.logspace(8, 14, 400) * eV_to_erg / m_e / c**2 # lab-frame Lorentz factor of the electron produced
theta_e_arr = np.logspace(-8,0, 400)

## Cell 3: times

t_arr = [1e8, 5e8, 1e9, 5e9, 1e10, 5e10, 1e11, 5e11, 1e12, 5e12, 1e13]
t_num = len(t_arr)

## Cell 4: obtaining particle distribution

def get_nij_old(i, j, t): # nij = dN / (dV dgammae_j dOmegae_i)
    gamma6 = gamma_e_arr[j] / 1e6
    Gamma_IC = 1.1e-12 * z2**4 * gamma6
    
    gamma_max = 1 / (Gamma_IC * (1 / Gamma_IC / gamma_e_arr[j] - t))
    if gamma_max < 0:
        gamma_max = float('inf')
    
    
    nij = 0
    for jp in range(j, 399):
        if gamma_e_arr[jp] <= gamma_max:
            sum_ij = (gamma_e_arr[jp] - gamma_e_arr[jp-1]) * rate_trans[i][jp]
            if gamma_e_arr[jp+1] > gamma_max:
                fraction = (gamma_max - gamma_e_arr[jp]) / (gamma_e_arr[jp+1] - gamma_e_arr[jp])
                sum_ij = sum_ij * fraction
                #print(f't={t:.1e}, fraction={fraction}')
            nij += sum_ij
    nij = nij / (Gamma_IC * gamma_e_arr[j]**2)
    return nij

# new version

def get_nij(i, j, t): # nij = dN / (dV dgammae_j dOmegae_i)
    Gamma_IC0 = 1.1e-18 * z2**4
    gamma_max_inv = 1./gamma_e_arr[j] - Gamma_IC0*t
    if gamma_max_inv>1e-99:
      gamma_max = 1./gamma_max_inv
    else:
      gamma_max = 1e99
    nij = 0
    for jp in range(j, 399):
        if gamma_e_arr[jp] <= gamma_max:
            sum_ij = (gamma_e_arr[jp] - gamma_e_arr[jp-1]) * rate_trans[i][jp]
            if gamma_e_arr[jp+1] > gamma_max:
                fraction = (gamma_max - gamma_e_arr[jp]) / (gamma_e_arr[jp+1] - gamma_e_arr[jp])
                sum_ij = sum_ij * fraction
                if i==0 and j%50==25: print(t, j, jp, fraction)
            nij += sum_ij
    nij = nij / (Gamma_IC0 * gamma_e_arr[j]**2)
    return nij

## Cell 5: number distribution <1/gamma> dN / [dV dOmega]

def get_ni(i, nij): # integrate gamma_e (1/gamma_e) nij
    ni = 0
    for j in range(1, 400):
        sum_i = (gamma_e_arr[j] - gamma_e_arr[j-1]) / gamma_e_arr[j] * nij[i][j]
        ni += sum_i
    return ni

## Cell 6: probability distribution in 1D angle

def get_Px(ni, Omega_x):
    Px = np.zeros((400, 1))
    for i in range(400): # Omega_x
        for j in range(1, 400): # theta
            if theta_e_arr[j-1] > Omega_x[i]:
                sum_ij = ni[j-1] * (np.sqrt(theta_e_arr[j]**2 - Omega_x[i]**2) - np.sqrt(theta_e_arr[j-1]**2 - Omega_x[i]**2))
                Px[i] += sum_ij
    return Px

## Cell 7: 1D angle grid

Omega_x = np.logspace(-8,0, 400).reshape(400, 1)

## Cells 8--10: build the arrays nij, ni, Px

nij = np.zeros((t_num, 400, 400)) # t theta[i] gamma[j]
for k in range(t_num):
    print('time', k, t_arr[k])
    for i in range(400):
        for j in range(1, 400):
            nij[k][i][j-1] = get_nij(i, j, t_arr[k])

ni = np.zeros((t_num, 400)) # about theta
for k in range(t_num):
    for i in range(400):
        ni[k][i] = get_ni(i, nij[k])

Px = np.zeros((t_num, 400)) # # cm^-3 rad^-1
for k in range(t_num):
    Px[k] = get_Px(ni[k], Omega_x).reshape(1, 400)

# Cell 11: ratios

print(Px[2][0]/Px[0][0]) # t = 1e9 compare to t = 1e8
print(Px[4][0]/Px[2][0]) # t = 1e10 compare to t = 1e9
print(Px[6][0]/Px[4][0]) # t = 1e11 compare to t = 1e10
print(Px[8][0]/Px[6][0]) # t = 1e12 compare to t = 1e11

# print the image

fits.PrimaryHDU(np.transpose(nij,axes=(0,2,1))).writeto('nij.fits', overwrite=True)
nij_scale = np.copy(nij)
for i in range(400): nij_scale[:,i,:] *= theta_e_arr[i]**2 * 2 * np.pi
for j in range(400): nij_scale[:,:,j] *= gamma_e_arr[j]
fits.PrimaryHDU(np.transpose(np.log10(np.clip(nij_scale,1e-100,1e100)),axes=(0,2,1))).writeto('nij_scale.fits', overwrite=True)

mc = np.zeros((400, t_num+1))
mc[:,0] = theta_e_arr
mc[:,1:] = ni.T
np.savetxt('ni.dat', mc)

# now get the d[<1/gamma> n_b P]/d Omega_x
#
# use the formula: if f(theta) is given at points theta_j
# then
# d [int f dOmega_y]/dOmega_x
# = int (df/dOmega_x) dOmega_y
# = 2 int_{0}^{infty} (df/dOmega_x) dOmega_y
# = 2 sum_{j'>=j} int_{A_j'}^{A_{j'+1}}  (df/dOmega_x) dOmega_y   # where A_j' = sqrt{theta_j'^2 - theta_j^2}
# = 2 sum_{j'>=j} int_{A_j'}^{A_{j'+1}} df/dtheta dtheta/dOmega_x dOmega_y
# = 2 sum_{j'>=j} [ df/dtheta {between j' and j'+1} ] int_{A_j'}^{A_{j'+1}} dtheta/dOmega_x dOmega_y
# = 2 sum_{j'>=j} [ df/dtheta {between j' and j'+1} ] int_{A_j'}^{A_{j'+1}} Omega_x/sqrt{Omega_x^2 + Omega_y^2} dOmega_y
#
# <-- now set Omega_y = Omega_x sinh q, cosh q = theta_j'/theta_j
#
# int_{A_j'}^{A_{j'+1}} Omega_x/sqrt{Omega_x^2 + Omega_y^2} dOmega_y
# = Omega_x ∆q
# = Omega_x [ cosh^-1 (theta_{j'+1}/theta_j) - cosh^-1 (theta_{j'}/theta_j) ]
#
# so d [int f dOmega_y]/dOmega_x = 2 Omega_x sum_{j'>=j} [ df/dtheta {between j' and j'+1} ] [ cosh^-1 (theta_{j'+1}/theta_j) - cosh^-1 (theta_{j'}/theta_j) ]

deriv_array = np.zeros((400,t_num))
for j in range(399):
  for jp in range(j,399):
    if j==jp:
      tr = 0.
    else:
      tr = np.arccosh(theta_e_arr[jp]/theta_e_arr[j])
    deriv_array[j,:] += 2 * theta_e_arr[j] * (ni[:,jp+1]-ni[:,jp])/(theta_e_arr[jp+1]-theta_e_arr[jp])\
      * (np.arccosh(theta_e_arr[jp+1]/theta_e_arr[j]) - tr)
mc2 = np.zeros((400, t_num+1))
mc2[:,0] = theta_e_arr
mc2[:,1:] = deriv_array
np.savetxt('deriv_array.dat', mc2)

