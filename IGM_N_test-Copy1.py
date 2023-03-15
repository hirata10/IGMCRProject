#!/usr/bin/env python
# coding: utf-8

# In[2]:

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bisect
import seaborn as sns
import LCDMSphere
import Energy_loss_class
import Source_class

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

nstep = 199 # redshift grid
mstep = 199 # energy grid

# command line arguments:
# linear overdensity at z=0
deltalin0 = float(sys.argv[1])

# In[3]:


# calculate overdensity evolution
C = LCDMSphere.XCosmo(0.6774, 0.319)
C.overdensity_evol(15., deltalin0, nstep+1)
z_list = C.z_grid # redshift grid
delta_b = C.Delta_grid # overdensity of baryon
Delta_list = np.array([x for x in delta_b]) # relative overdensity of baryon - grid
ln_Delta = [np.log(x) for x in Delta_list]

dDdz = np.zeros((nstep, )) # d ln(Delta)/dz
for i in range(nstep):
    dDdz[i] = (ln_Delta[i+1] - ln_Delta[i]) / (z_list[i+1] - z_list[i])

z = (z_list[:-1] + z_list[1:]) / 2 # midpoint of z grid, correspond to d ln(Delta)/dz
    
Delta = (Delta_list[:-1] + Delta_list[1:]) / 2 # midpoint of overdensity grid, correspond to d ln(Delta)/dz
    
H = np.zeros((nstep, ))
for i in range(nstep):
    H[i] = C.H(z[i])
    
theta = np.zeros((nstep, ))
for i in range(nstep):
    theta[i] = -3 / (1 + z[i]) + dDdz[i]


# In[4]:


# A1 = -theta, coefficient of N(E, z)


# In[5]:


# coefficient of dN(E, z)/dE
def A2(E, z, Delta_i, theta_i, H):
    loss = Energy_loss_class.Loss(z, E, Delta_i)
    co1 = theta_i / 3 * E * (E + 2 * E_e) / (E + E_e)
    co2 = loss.E_loss() / (-(1 + z) * H)
    co = co1 - co2
    return co


# In[6]:


# coefficient due to secondary electron
def A3(E, E_p, z, H): # E_p means E_prime
    omega_b = omega_b_0 * (1 + z)**3
    n_H = 0.76 * omega_b * rho_crit / m_p
    n_etot = n_H * (1 + 2 * f_He)
    beta_p = np.sqrt(E_p * (E_p + 2 * E_e)) / (E_p + E_e)
    co1 = 2 * np.pi * r0 * r0 * E_e * (E_e + E_p)**2 / E_p / (2 * E_e + E_p) / E / E
    co2 = 1 + E * E / (E_p - E)**2 + E * E / (E_e + E_p)**2 - E_e * (E_e + 2 * E_p) * E/ (E_e + E_p)**2 / (E_p - E)
    return n_etot * c * beta_p * co1 * co2 / (-(1 + z) * H)


# In[7]:


E_min = 1e-8 # temp, in erg
E_max = 1e-3 # temp, in erg

E = np.logspace(np.log10(E_min), np.log10(E_max), mstep + 1)

E_mid = (E[:-1] + E[1:]) / 2 # midpoint for reference
E_plus = np.zeros((mstep+1, 1)) # E_{i+1/2} = sqrt(E[i] * E[i+1])
E_minus = np.zeros((mstep+1, 1)) # E_{i-1/2} = sqrt(E[i-1] * E[i])

for i in range(mstep):
    E_plus[i] = np.sqrt(E[i] * E[i+1]) # length len(E) -1

E_plus[-1] = E_plus[-2] * E[-1] / E[-2] # E[i+1]/E[i] is constant, so in E_plus and E_minus

for i in range(1, mstep+1):
    E_minus[i] = np.sqrt(E[i] * E[i-1])
    
E_minus[0] = E_minus[1] / (E[1]/E[0])


# In[8]:


S = np.loadtxt("source_term.txt") # in 'same columns, same redshift; same row, same energy' format 
S = np.delete(S, 0, axis=1) 
Sz = [[S[i][j] / (-(1 + z[j]) * H[j]) * Delta[j] for j in range(nstep)] for i in range(mstep)]

#print('z', z_list, len(z_list))
#print('E', E, np.size(E))
#print('E^2 S[iE,100] (eV/cm^3/dz) at', z_list[100], ':', -np.array(Sz)[:,100]*E_mid**2/1.602e-12)
#print('theta', theta[100], '∆', Delta[100], 'H', H[100])

thisA3 = np.zeros((mstep,mstep))
for j in range(mstep):
  for i in range(mstep):
    if E[j] >= 2 * E[i]:
      thisA3[i,j] = A3(E[i], E[j], z_list[100], H[100]) * E[i]/E[j] * E[i]
  thisA2 = A2(E[j], z_list[100], Delta[100], theta[100], H[100])
  #print('{:12.5E} {:12.5E} {:12.5E}'.format(E[j], thisA2, np.sum(thisA3[:,j])*E[j]*np.log(1e5)/199/thisA2))
from astropy.io import fits
fits.PrimaryHDU(thisA3).writeto('a3.fits', overwrite=True)

# In[9]:


def get_M(z, Delta, theta, H): # parameters that only depend on redshift
    M = np.zeros((mstep, mstep))

    for i in range(mstep):
        for j in range(mstep):
            if j == i or j == i+1:
                A1_ij = -theta
        
                A2_ij = A2(E[j], z, Delta, theta, H)
                A2_ij_p = A2(E_plus[j], z, Delta, theta, H)
                A2_ij_m = A2(E_minus[j], z, Delta, theta, H)
        
                dE = E[j+1] - E[j]
                dE_p = E_plus[j+1] - E_plus[j]
                dE_m = E_minus[j+1] - E_minus[j]
        
                if j == i: # subcribe of n is equal to j
                    M[i][j] += A1_ij - A2_ij_m/dE_m
                if j == i+1:
                    M[i][j] += A2_ij_m/dE_m
                
            if E[j] >= 2 * E[i]:
                M[i][j] += A3(E[i], E[j], z, H) * (E_plus[i]-E_minus[i]) # integrate E_p from 2E
                
    return M

fits.PrimaryHDU(get_M(z_list[100],Delta[100],theta[100],H[100])).writeto('M.fits', overwrite=True)

# In[10]:


SzdE = [[Sz[i][j]*(E_plus[i]-E_minus[i]) for j in range(nstep)] for i in range(mstep)]
SzdE = np.array(SzdE).reshape(mstep,nstep)

fits.PrimaryHDU(SzdE).writeto('SzdE.fits', overwrite=True)

# In[20]:


def back_Euler(): # solve dn/dz = Mn + S using backward Euler method, M is square matrix, S is column vector
    n = np.zeros((mstep, 1))
    I = np.identity(mstep)
    
    for j in range(1, nstep): # initialize the first column to 0 with z[0], start from z[1]
        #print('iter', j, 'of', nstep)
        n_jm = np.zeros((mstep,1)) # partition column vector with redshift z[j-1]
        for i in range(mstep):
            n_jm[i] = n[i][j-1]
        
        dz = z[j] - z[j-1]
        
        S_j = np.zeros((mstep, 1))
        for i in range(mstep):
            S_j[i] = SzdE[i][j] # under same redshift, vari from different energy
        
        dzS = dz*S_j
        n_jm_plus_dzS = np.zeros((mstep,))
        for i in range(mstep):
            n_jm_plus_dzS[i] = n_jm[i] + dzS[i]
            
        M_j = get_M(z[j], Delta[j], theta[j], H[j]) # M matrix only depend on redshift
            
        dzM = dz*M_j
        I_minus_dzM = np.zeros((mstep,mstep))
        for i in range(mstep):
            for k in range(mstep):
                I_minus_dzM[i][k] = I[i][k]-dzM[i][k]
                
        I_m_inv = np.linalg.inv(I_minus_dzM)
        
        n_j = I_m_inv@n_jm_plus_dzS
    
        n = np.hstack((n,n_j.reshape(mstep,1)))
        
    return n

n = back_Euler()
fits.PrimaryHDU(n).writeto('n.fits', overwrite=True)
E__dE = E/(E_plus-E_minus).flatten()
fits.PrimaryHDU(np.log10(1e-49+n*E__dE[:-1,None])).writeto('lognE.fits', overwrite=True)
utot = np.sum(n*E[:-1,None], axis=0)
for j in range(nstep):
  nbary = omega_b_0 * (1 + z_list[j])**3 * Delta[j] * rho_crit / m_p
  print('{:4d} {:7.4f} {:9.5f} {:11.5E} {:11.5E}'.format(j, z_list[j], Delta[j], utot[j], utot[j]/nbary/1.602e-12))
exit()

# In[ ]:





# In[21]:


I = np.identity(mstep)

j = 1
        
dz = z[j] - z[j-1]
            
M_j = get_M(z[j], Delta[j], theta[j], H[j]) # M matrix only depend on redshift
eigen_M = np.linalg.eigvals(M_j)
real = eigen_M.real
img = eigen_M.imag
plt.plot(real, img, '.')
plt.show()

eigen_val, eigen_vec = np.linalg.eig(M_j)

eigen_1600 = eigen_vec[:,0]
print(eigen_1600)

            
dzM = np.dot(dz, M_j)

I_minus_dzM = np.zeros((mstep,mstep))
for i in range(mstep):
    for k in range(mstep):
        I_minus_dzM[i][k] = I[i][k] - dzM[i][k]

print(np.linalg.det(I_minus_dzM))
print('---')

I_m_inv = np.linalg.pinv(I_minus_dzM)
print(np.dot(I_m_inv, I_minus_dzM))


# In[13]:


print(M_j[3:10])


# In[14]:


M_data = pd.DataFrame(data=M_j, index=E_mid, columns=E_mid)
plt.figure(figsize = (12,12))
sns.heatmap(M_data)


# In[22]:


n = back_Euler()


# In[23]:


N = np.zeros((mstep, nstep))
for i in range(mstep):
    for j in range(nstep):
        N[i][j] = n[i][j] / (E_plus[i] - E_minus[i])


# In[24]:


N_data = pd.DataFrame(data=N, index=E_mid, columns=z)
plt.figure(figsize = (12,12))
sns.heatmap(N_data)


# In[13]:


E_density = np.zeros((nstep, ))
for i in range(mstep):
    for j in range(nstep):
        E_density[j] += N[i][j] * E[i] * E[i]
plt.plot(z, E_density, '.')
plt.show()


# In[25]:


count = 0
for i in range(mstep):
    for j in range(nstep):
        if N[i][j] < 0:
            count += 1
print(count)


# In[ ]:




