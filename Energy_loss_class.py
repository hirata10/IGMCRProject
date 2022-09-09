#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

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

z_reion_He = 3 # redshift that He is fully ionized
z_reion_H = 11 # redshift that H is fully ionized

aR = 8 * np.pi**5 * k**4 / (15 * c**3 * h**3) # radiation constant


# In[10]:


"""
To use this class, call Loss(z, E, overdensity)
where z is redshift, E is energy in erg, overdensity is mass overdensity

all variables should be individual values rather than array or list

.E_loss_inv_Compton() will return to the energy loss due to inverse Compton cooling
.E_loss_coll_free() will return to the collisional losses including both free and bound collisions
.E_loss() will return to the total energy loss
"""

class Loss:
    def __init__(self, z, E, overdensity):
        self.z = z
        self.E = E
        self.overdensity = overdensity
        
    def E_loss_inv_Compton(self):
        T_CMB_z = T_CMB * (1 + self.z)
        return -32 * np.pi * r0 * r0 * aR * T_CMB_z**4 / (9 * m_e * c) * (2 + self.E / m_e / c / c) * self.E
    
    def E_loss_coll_free(self):
        E_loss_free = 0 # initialize
        E_loss_coll = 0 # initialize
    
        omega_b = omega_b_0 * (1 + self.z)**3
        mean_n_b = 3 * omega_b * H_0 * H_0 / (8 * np.pi * G * m_p) # mean number density of baryons at the redshift
        n_b = mean_n_b * self.overdensity # local number density of baryons at the redshitf
    
        n_H = n_b * (1 - Y_He) # nuclei in hydrogen
        n_He = n_H * f_He # nuclei in helium, assume He is all 4He
    
        N_ex = [1, 1, 2] # number of bound electrons, H0, He+, He0 accordingly
        I_x_eV = [15.0, 59.9, 42.3] # geometric mean excitation energy for H0, He+, He0 accordingly
        I_x = [x * 1.602e-12 for x in I_x_eV] # 1eV = 1.602e-12 erg
    
        if self.z <= z_reion_He: # fully ionized
            n_e = n_H + 2 * n_He
            n_x = [0, 0, 0] # number density of H0, He+, He0 accordingly
        if z_reion_He < self.z <= z_reion_H: # He is singly ionized
            n_e = n_H + n_He
            n_x = [0, n_He, 0]
        if self.z > z_reion_H: # neutral
            n_e = 0
            n_x = [n_H, 0, n_He]
    
        gamma = self.E / (m_e * c * c) + 1
        beta = np.sqrt(self.E * (self.E + 2 * m_e * c * c)) / (self.E + m_e * c * c)
        omega_p = c * np.sqrt(4 * np.pi * r0 * n_e) # plasma frequency
        I = h * omega_p # mean excitation energy
        tau = self.E / (m_e * c * c)
        F = (1 - beta * beta) * (1 + tau * tau / 8 - (2 * tau + 1) * np.log(2)) # Moller scattering formula
        zeta = 2 * np.log(gamma) - beta * beta # density correction
    
        if self.z <= z_reion_H:
            E_loss_free = -2 * np.pi * n_e * r0 * r0 * m_e * c**3 / beta * (np.log(self.E * self.E / I / I) + np.log(1 + tau / 2) + F - zeta)
        if self.z > z_reion_He:
            for i in range(3):
                E_loss_coll += -2 * np.pi * r0 * r0 * m_e * c**3 / beta * (N_ex[i] * n_x[i] * (np.log(self.E * self.E / I_x[i] / I_x[i]) + np.log(1 + tau / 2) + F))
    
        return E_loss_free + E_loss_coll
    
    def E_loss(self):
        E1 = self.E_loss_inv_Compton() # energy loss due to inverse Compton
        E2 = self.E_loss_coll_free() # energy loss due to collision
        return E1 + E2

