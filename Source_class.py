#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sympy as smp
import scipy.interpolate
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import math

r0 = 2.818e-13 # classical electron radius in cm
m_e = 9.10938e-28 # electron mass in gram
c = 2.998e10 # speed of light in cm/s
h = 6.6261e-27 # planck constant in cm^2 g s-1

f_He = 0.079
H_0 = 2.184e-18 # current hubble constant in s^-1
H_r = 3.24076e-18 # 100 km/s/Mpc to 1/s
h_0 = H_0 / H_r
omega_b_0 = 0.0224 / h_0 / h_0  # current baryon abundance
m_p = 1.67262193269e-24 # proton mass in g
G = 6.6743e-8 # gravitational constant in cm^3 g^-1 s^-2

rho_crit = 3 * H_0 * H_0 / 8 / np.pi / G


# In[2]:


z_list = []
wavelength = []
EBL = []

def read_file(filename, z):
    with open(filename) as f:
        lines = f.readlines()[3:]
        for line in lines:
            data = line.split()
            for x in range(0, len(data), 2):
                z_list.append(float(z))
                wavelength.append(float(data[x]))
                EBL.append(float(data[x+1]))
                
def read_data():
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 0.0.txt',0)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 0.1.txt',0.1)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 0.2.txt',0.2)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 0.3.txt',0.3)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 0.4.txt',0.4)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 0.5.txt',0.5)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 0.6.txt',0.6)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 0.7.txt',0.7)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 0.8.txt',0.8)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 0.9.txt',0.9)

    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 1.0.txt',1.0)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 1.1.txt',1.1)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 1.2.txt',1.2)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 1.3.txt',1.3)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 1.4.txt',1.4)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 1.5.txt',1.5)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 1.6.txt',1.6)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 1.7.txt',1.7)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 1.8.txt',1.8)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 1.9.txt',1.9)

    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 2.0.txt',2.0)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 2.1.txt',2.1)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 2.2.txt',2.2)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 2.3.txt',2.3)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 2.4.txt',2.4)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 2.5.txt',2.5)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 2.6.txt',2.6)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 2.7.txt',2.7)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 2.8.txt',2.8)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 2.9.txt',2.9)

    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 3.0.txt',3.0)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 3.1.txt',3.1)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 3.2.txt',3.2)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 3.3.txt',3.3)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 3.4.txt',3.4)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 3.5.txt',3.5)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 3.6.txt',3.6)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 3.7.txt',3.7)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 3.8.txt',3.8)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 3.9.txt',3.9)

    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 4.0.txt',4.0)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 4.1.txt',4.1)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 4.2.txt',4.2)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 4.3.txt',4.3)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 4.4.txt',4.4)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 4.5.txt',4.5)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 4.6.txt',4.6)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 4.7.txt',4.7)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 4.8.txt',4.8)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 4.9.txt',4.9)

    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 5.0.txt',5.0)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 5.1.txt',5.1)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 5.2.txt',5.2)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 5.3.txt',5.3)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 5.4.txt',5.4)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 5.5.txt',5.5)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 5.6.txt',5.6)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 5.7.txt',5.7)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 5.8.txt',5.8)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 5.9.txt',5.9)

    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 6.0.txt',6.0)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 6.2.txt',6.2)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 6.4.txt',6.4)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 6.6.txt',6.6)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 6.8.txt',6.8)

    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 7.0.txt',7.0)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 7.2.txt',7.2)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 7.4.txt',7.4)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 7.6.txt',7.6)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 7.8.txt',7.8)

    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 8.0.txt',8.0)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 8.2.txt',8.2)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 8.4.txt',8.4)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 8.6.txt',8.6)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 8.8.txt',8.8)

    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 9.0.txt',9.0)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 9.2.txt',9.2)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 9.4.txt',9.4)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 9.6.txt',9.6)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 9.8.txt',9.8)

    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_10.0.txt',10.0)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_10.5.txt',10.5)

    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_11.0.txt',11.0)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_11.5.txt',11.5)

    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_12.0.txt',12.0)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_12.5.txt',12.5)

    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_13.0.txt',13.0)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_13.5.txt',13.5)

    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_14.0.txt',14.0)
    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_14.5.txt',14.5)

    read_file('./KS_2018_EBL/Q20/EBL_KS18_Q20_z_15.0.txt',15.0)
    
read_data()


# In[3]:


"""
To use this class, call Source(z, E, Source_class.z_list, Source_class.wavelength, Source_class.EBL)
where z is redshift, E is energy in erg

both variables should be individual values rather than array or list

.Source_term() will return to the rate of production of Compton-scattered electrons per unit physical volume per unit energy
in unit of number/cm^3/erg

the required data file can be downloaded at http://www.iucaa.in/projects/cobra/
"""

class Source:
    def __init__(self, z, E, z_list, wavelength, EBL):
        self.z = z
        self.E = E
        self.z_list = z_list # redshift
        self.wavelength = wavelength # in Angstrom, 10^-8 cm
        self.EBL = EBL # specific intensity J_nu in erg/s/cm^2/Hz/Sr
    
    def EBL_fit(self, z, wavelength_num):
        if z < 0.1:
            index_zj = 0
            index_zj1 = 695
        elif z > 14.5:
            index_zj = 61855
            index_zj1 = 62550
        else:
            index_zj = bisect.bisect_left(self.z_list, z) - 1
            index_zj1 = index_zj + 695
    
        index_a = self.z_list.index(self.z_list[index_zj])
        index_b = self.z_list.index(self.z_list[index_zj1])
    
        if wavelength_num <= 1e-08:
            jk = bisect.bisect_left(self.wavelength, wavelength_num, index_a, index_a+694)    
            j1k = bisect.bisect_left(self.wavelength, wavelength_num, index_b, index_b+694)
        else:
            jk = bisect.bisect_left(self.wavelength, wavelength_num, index_a, index_a+694) - 1
            j1k = bisect.bisect_left(self.wavelength, wavelength_num, index_b, index_b+694) - 1

        jk1 = jk + 1
        j1k1 = j1k + 1
    
        f_A = self.EBL[jk] + (z - self.z_list[index_zj]) / (self.z_list[index_zj1] - self.z_list[index_zj]) * (self.EBL[j1k] - self.EBL[jk])
        f_B = self.EBL[jk1] + (z - self.z_list[index_zj]) / (self.z_list[index_zj1] - self.z_list[index_zj]) * (self.EBL[j1k1] - self.EBL[jk1])
        f_xy = f_A + (wavelength_num - self.wavelength[jk]) / (self.wavelength[jk1] - self.wavelength[jk]) * (f_B - f_A)
    
        return f_xy
    
    def Source_equ(self, z, E_gamma, E):
        wave = h * c / E_gamma * 1e8 # E_gamma to wavelength in Angstrom
        J_nu = self.EBL_fit(z, wave)
        omega_b = omega_b_0 * (1 + z)**3
        n_H = 0.76 * omega_b * rho_crit / m_p
        n_etot = n_H * (1 + 2 * f_He)
        J = 4 * np.pi * n_etot * J_nu / h / E_gamma
        d_sig = np.pi*r0*r0 * m_e*c*c/(E_gamma*(E_gamma-E)) * (2 + E*E/E_gamma/E_gamma - 2*E/E_gamma - 2*m_e*c*c*E/E_gamma/E_gamma + (m_e*c*c)**2*E*E/E_gamma**3/(E_gamma-E))
        return d_sig*J
    
    def Source_term(self):
        # the rate of production of Compton-scattered electrons per unit physical volume per unit energy
        # number/cm^3/erg
        E_gamma_min = (self.E + np.sqrt(self.E * (self.E + 2*m_e*c*c))) / 2
        E_gamma_array = np.logspace(np.log10(E_gamma_min), np.log10(2), 10000)
        source_int = 0
        for i in range(len(E_gamma_array)-1):
            source_int_i = self.Source_equ(self.z, E_gamma_array[i], self.E) * (E_gamma_array[i+1] - E_gamma_array[i])
            source_int += source_int_i
        return source_int

