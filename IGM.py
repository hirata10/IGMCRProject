#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
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
# deltalin0 = 0


# In[2]:


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
    
E_minus[0] = E_minus[1] / (E[1] / E[0])


# In[3]:

"""
To use this class, call IGM_N(deltalin0, model)
where deltalin0 is overdensity factor (<0 means underdensity, 0 means mean density, >0 means overdensity),
and model is either 0 or 1 (0 using Khaire's source model and 1 using Haardt's source model)

all variables should be individual values rather than array or list

.get_utot() will return to the energy density utot and energy per baryon in eV E_eV
.get_P() will return to the number of electron per volume per erg N, pressure in erg cm^-3 P_e and pressure in eV cm^-3 P_eV
"""


class IGM_N:
    def __init__(self, deltalin0, model):
        if model == 0: # Khaire's model
            self.filename = "source_term_Khaire.txt"
        if model == 1: # Haardt's model
            self.filename = "source_term_Haardt.txt"
        
        # calculate overdensity evolution
        C = LCDMSphere.XCosmo(0.674, 0.315)
        C.overdensity_evol(15., deltalin0, nstep+1)
        self.z_list = C.z_grid # redshift grid
        delta_b = C.Delta_grid # overdensity of baryon
        Delta_list = np.array([x for x in delta_b]) # relative overdensity of baryon - grid
        ln_Delta = [np.log(x) for x in Delta_list]

        dDdz = np.zeros((nstep, )) # d ln(Delta)/dz
        for i in range(nstep):
            dDdz[i] = (ln_Delta[i+1] - ln_Delta[i]) / (self.z_list[i+1] - self.z_list[i])

        self.z = (self.z_list[:-1] + self.z_list[1:]) / 2 # midpoint of z grid, correspond to d ln(Delta)/dz
    
        self.Delta = (Delta_list[:-1] + Delta_list[1:]) / 2 # midpoint of overdensity grid, correspond to d ln(Delta)/dz
    
        self.H = np.zeros((nstep, ))
        for i in range(nstep):
            self.H[i] = C.H(self.z[i])
    
        self.theta = np.zeros((nstep, ))
        for i in range(nstep):
            self.theta[i] = -3 / (1 + self.z[i]) + dDdz[i]
            
        self.S = np.loadtxt(self.filename) # in 'same columns, same redshift; same row, same energy' format 
        self.S = np.delete(self.S, 0, axis=1) 
        self.Sz = [[self.S[i][j] / (-(1 + self.z[j]) * self.H[j]) * self.Delta[j] for j in range(nstep)] for i in range(mstep)]
        
        self.SzdE = [[self.Sz[i][j] * (E_plus[i] - E_minus[i]) for j in range(nstep)] for i in range(mstep)]
        self.SzdE = np.array(self.SzdE).reshape(mstep,nstep)
            
    # A1 = -theta, coefficient of N(E, z)
    
    # coefficient of dN(E, z)/dE
    def A2(self, E, z, Delta_i, theta_i, H):
        loss = Energy_loss_class.Loss(z, E, Delta_i)
        co1 = theta_i / 3 * E * (E + 2 * E_e) / (E + E_e)
        co2 = loss.E_loss() / (-(1 + z) * H)
        co = co1 - co2
        return co
    
    def A3(self, E, E_p, z, H): # E_p means E_prime
        omega_b = omega_b_0 * (1 + z)**3
        n_H = 0.76 * omega_b * rho_crit / m_p
        n_etot = n_H * (1 + 2 * f_He)
        beta_p = np.sqrt(E_p * (E_p + 2 * E_e)) / (E_p + E_e)
        co1 = 2 * np.pi * r0 * r0 * E_e * (E_e + E_p)**2 / E_p / (2 * E_e + E_p) / E / E
        co2 = 1 + E * E / (E_p - E)**2 + E * E / (E_e + E_p)**2 - E_e * (E_e + 2 * E_p) * E/ (E_e + E_p)**2 / (E_p - E)
        return n_etot * c * beta_p * co1 * co2 / (-(1 + z) * H)
    
    def get_M(self, z, Delta, theta, H): # parameters that only depend on redshift
        M = np.zeros((mstep, mstep))

        for i in range(mstep):
            for j in range(mstep):
                if j == i or j == i+1:
                    A1_ij = -theta
        
                    A2_ij = self.A2(E[j], z, Delta, theta, H)
                    A2_ij_p = self.A2(E_plus[j], z, Delta, theta, H)
                    A2_ij_m = self.A2(E_minus[j], z, Delta, theta, H)
        
                    dE = E[j+1] - E[j]
                    dE_p = E_plus[j+1] - E_plus[j]
                    dE_m = E_minus[j+1] - E_minus[j]
        
                    if j == i: # subcribe of n is equal to j
                        M[i][j] += A1_ij - A2_ij_m / dE_m
                    if j == i+1:
                        M[i][j] += A2_ij_m / dE_m
                
                if E[j] >= 2 * E[i]:
                    M[i][j] += self.A3(E[i], E[j], z, H) * (E_plus[i] - E_minus[i]) # integrate E_p from 2E
                
        return M
    
    def back_Euler(self): # solve dn/dz = Mn + S using backward Euler method, M is square matrix, S is column vector
        n = np.zeros((mstep, 1))
        I = np.identity(mstep)
    
        for j in range(1, nstep): # initialize the first column to 0 with z[0], start from z[1]
            n_jm = np.zeros((mstep,1)) # partition column vector with redshift z[j-1]
            for i in range(mstep):
                n_jm[i] = n[i][j-1]
        
            dz = self.z[j] - self.z[j-1]
        
            S_j = np.zeros((mstep, 1))
            for i in range(mstep):
                S_j[i] = self.SzdE[i][j] # under same redshift, vari from different energy
        
            dzS = dz*S_j
            n_jm_plus_dzS = np.zeros((mstep,))
            for i in range(mstep):
                n_jm_plus_dzS[i] = n_jm[i] + dzS[i]
            
            M_j = self.get_M(self.z[j], self.Delta[j], self.theta[j], self.H[j]) # M matrix only depend on redshift
            
            dzM = dz*M_j
            I_minus_dzM = np.zeros((mstep,mstep))
            for i in range(mstep):
                for k in range(mstep):
                    I_minus_dzM[i][k] = I[i][k]-dzM[i][k]
                
            I_m_inv = np.linalg.inv(I_minus_dzM)
        
            n_j = I_m_inv@n_jm_plus_dzS
    
            n = np.hstack((n,n_j.reshape(mstep,1)))
        
        return n
    
    def get_utot(self):
        n = self.back_Euler()
        utot = np.sum(n * E[:-1,None], axis=0) # erg cm^-3
        E_eV = np.zeros((nstep, ))
        nbary = np.zeros((nstep, ))
        for j in range(nstep):
            nbary[j] = omega_b_0 * (1 + self.z_list[j])**3 * self.Delta[j] * rho_crit / m_p # number density of baryon
            E_eV[j] = utot[j] / nbary[j] / 1.602e-12 # energy per baryon in eV
        return utot, E_eV
    
    def get_P(self):
        n = self.back_Euler()
        
        N = np.zeros((mstep, nstep))
        for i in range(mstep):
            for j in range(nstep):
                N[i][j] = n[i][j] / (E_plus[i] - E_minus[i])
                
        P_e = np.zeros((nstep, ))
        for j in range(nstep):    
            for i in range(mstep):
                P_e[j] = P_e[j] + N[i][j] * (E_plus[i] * (E_plus[i]+2*E_e)/(E_plus[i]+E_e)) * (E[i+1]-E[i]) / 3
                
        P_eV = [x / 1.602e-12 for x in P_e]
        return N, P_e, P_eV

