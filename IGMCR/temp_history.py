import numpy as np
import LCDMSphere
from scipy import integrate
from scipy import interpolate


f_He = 0.079
H_0 = 2.184e-18 # current hubble constant in s^-1
H_r = 3.24076e-18 # 100 km/s/Mpc to 1/s
h_0 = H_0 / H_r
omega_b_0 = 0.0224 / h_0 / h_0  # current baryon abundance
Omega_m = 0.319
Omega_lambda = 0.691
m_p = 1.67262193269e-24 # proton mass in g
G = 6.6743e-8 # gravitational constant in cm^3 g^-1 s^-2

rho_crit = 3 * H_0 * H_0 / 8 / np.pi / G
Y_H = (1-f_He)/(1+3*f_He)

r0 = 2.818e-13 # classical electron radius in cm
m_e = 9.10938e-28 # electron mass in gram
c = 2.998e10 # speed of light in cm/s
h = 6.6261e-27 # planck constant in cm^2 g s-1
eV_to_erg = 1.6022e-12

T_TR = {}
T_TR['HeII'] = 285335
T_TR['HeIII'] = 631515
T_TR['HII'] = 157807

gamma = {}
gamma['HeII'] = 2.8
gamma['HeIII'] = 1.7
gamma['HII'] = 2.8

ion_energy = {}
ion_energy['HeIII'] = 24.6 * eV_to_erg
ion_energy['HeII'] = 54.4 * eV_to_erg
ion_energy['HII'] = 13.6 * eV_to_erg

kB = 1.380649e-16 # bolzmann constant in erg/K

# z_H_reion = 8
# z_HeII_reion = 3

class IGM_temperature:
	def __init__(self, z_H_reion, z_HeII_reion, alpha_bkg, alpha_QSO, C_HeIII, Delta_0, z_0):
		self.alpha_bkg = alpha_bkg
		self.z_H_reion = z_H_reion
		self.z_HeII_reion = z_HeII_reion
		self.alpha_QSO = alpha_QSO
		self.C_HeIII = C_HeIII
		self.Delta_0 = Delta_0
		self.z_0 = z_0

		C = LCDMSphere.XCosmo(Omega_lambda, Omega_m)
		C.overdensity_evol(15., Delta_0, 200+1)
		delta_b = C.Delta_grid # overdensity of baryon
		self.delta_func = interpolate.interp1d(C.z_grid, delta_b)

	def n_b(self, z):
		'''
		unit: cm^-3
		'''
		omega_b = omega_b_0 * (1 + z)**3
		n_b =  omega_b * rho_crit / m_p * self.Delta_gas(z, self.Delta_0, self.z_0)
		# n_b =  omega_b * rho_crit / m_p

		return n_b

	def n_H(self, z):
		n_H = Y_H * self.n_b(z)
		
		return n_H

	def n_HI(self, z):
		if z> self.z_H_reion:
			n_HI = self.n_H(z)
		elif z <= self.z_H_reion:
			n_HI = 0

		return n_HI

	def n_HII(self, z):
		if z> self.z_H_reion:
			n_HII = 0
		elif z <= self.z_H_reion:
			n_HII = self.n_H(z)

		return n_HII

	def n_He(self, z):   
		n_He = (1-Y_H)/4 * self.n_b(z)
		
		return n_He


	def n_HeI(self, z):

		return self.n_He(z) - self.n_HeII(z) - self.n_HeIII(z)

	def n_HeII(self, z):
		if z> self.z_H_reion:
			n_HeII = 0
		elif self.z_HeII_reion <= z <= self.z_H_reion:
			n_HeII = self.n_He(z)
		elif z < self.z_HeII_reion:
			n_HeII = 0
		
		return n_HeII

	def n_HeIII(self, z):
		if z > self.z_HeII_reion:
			n_HeIII = 0
		elif z <= self.z_HeII_reion:
			n_HeIII = self.n_He(z)
		
		return n_HeIII


	def n_e(self, z):
		if z > self.z_H_reion:
			n_e = 0
		elif self.z_HeII_reion <= z <= self.z_H_reion:
			n_e = self.n_H(z) + self.n_He(z)
		elif z < self.z_HeII_reion:
			n_e = self.n_H(z) + 2 * self.n_He(z)
		
		return n_e

	def n_tot(self, z):
		n_tot = self.n_H(z) + self.n_He(z) + self.n_e(z)

		return n_tot

	# Zeldovich evolution
	# def Delta_gas(self, z, Delta_0, z0):

	# 	# smooth_factor = 1.e-6
	# 	a0 = 1./(1+z0)
	# 	l = (1-1./(Delta_0))/a0

	# 	a = 1./(1+z)
	# 	return 1./(1-l*a)

	# spherical collapse
	def Delta_gas(self, z, Delta_0, z0):

		return self.delta_func(z)

	def H(self, z):
		'''
		unit: s^-1
		'''
		return H_0 * np.sqrt(Omega_lambda + Omega_m*(1+z)**3)

	def t(self, z):
		'''
		unit: s
		'''
		a = 1 / (1+z)
		t = 2 / H_0 / 3 / Omega_m**(1/2) * a**(3/2)

		return t

	def dt_dz(self, z):
		'''
		unit: s
		'''
		return -1 / H_0 / Omega_m**(1/2) / (1+z)**(5/2)

	def sigma_photo_ion(self, E, species):
		'''
		unit cm^2
		'''
		if species == 'HI':
			E0, sigma0, P, ya, yw, y0, y1 = [4.298e-1, 5.475e-14, 2.963, 32.88, 0, 0, 0]
		if species == 'HeI':
			E0, sigma0, P, ya, yw, y0, y1 = [13.61, 9.492e-16, 3.188, 1.469, 2.039, 0.4434, 2.136]
		if species == 'HeII':
			E0, sigma0, P, ya, yw, y0, y1 = [1.720, 1.369e-14, 2.963, 32.88, 0, 0, 0]

		x = E/E0 - y0
		y = np.sqrt(x**2 + y1**2)
		
		sigma = sigma0 * ((x-1)**2 +yw**2) * y**(0.5*P - 5.5)/ (1+np.sqrt(y/ya))**P

		return sigma

	def coefs_caseA(self, T, species, coef):
		'''
		recombination coefficients unit: cm^3/s
		cooling rate unit: erg cm^3 sec^-1 K^-1 
		'''
		lambda_ = 2*T_TR[species] / T 
		if species == 'HII':
			coef_recomb = 1.269e-13 * (lambda_**1.503) / (1+(lambda_/0.522)**0.470)**1.923
			cool_rate = 1.778e-29 * lambda_**1.965 * T / (1+(lambda_/0.541)**0.502)**2.697
		if species == 'HeII':
			coef_recomb = 3.0e-14 * lambda_**0.654
			cool_rate = kB * T *coef_recomb
		if species == 'HeIII':
			coef_recomb = 2.0 * 1.269e-13 * (lambda_**1.503) / (1+(lambda_/0.522)**0.470)**1.923
			cool_rate = 8 * 1.778e-29 * lambda_**1.965 *T / (1+(lambda_/0.541)**0.502)**2.697

		if coef == 'coef':
			return coef_recomb
		elif coef == 'cool':
			return cool_rate

	def coefs_caseB(self, T, species, coef):
		lambda_ = 2*T_TR[species] / T 
		if species == 'HII':
			coef_recomb =2.753e-14 * (lambda_**1.500) / (1+(lambda_/2.740)**0.470)**2.242
			cool_rate = 3.435e-30 * lambda_**1.970 * T / (1+(lambda_/2.25)**0.376)**3.720
		if species == 'HeII':
			coef_recomb = 1.26e-14 * lambda_**0.750
			cool_rate = kB * T *coef_recomb
		if species == 'HeIII':
			coef_recomb = 2.0 * 2.753e-14 * (lambda_**1.500) / (1+(lambda_/2.740)**0.470)**2.242
			cool_rate = 8 * 3.435e-30 * lambda_**1.970 * T / (1+(lambda_/2.25)**0.376)**3.720

		if coef == 'coef':
			return coef_recomb
		elif coef == 'cool':
			return cool_rate
        
	def coefs_coll_ion(self, T, species, coef):
		lambda_ = 2*T_TR[species] / T 
		if species == 'HII':
			coef_recomb = 21.11 / T **(3/2) * np.exp(-lambda_/2) * (lambda_**(-1.089)) / (1+(lambda_/0.354)**0.874)**1.101
			cool_rate = kB * T_TR[species] * coef_recomb
		if species == 'HeII':
			coef_recomb = 32.38 / T **(3/2) * np.exp(-lambda_/2) * (lambda_**(-1.146)) / (1+(lambda_/0.416)**0.987)**1.056
			cool_rate = kB * T_TR[species] *coef_recomb
		if species == 'HeIII':
			coef_recomb = 19.95 / T **(3/2) * np.exp(-lambda_/2) * (lambda_**(-1.089)) / (1+(lambda_/0.553)**0.735)**1.275
			cool_rate = kB * T_TR[species] *coef_recomb

		if coef == 'coef':
			return coef_recomb
		elif coef == 'cool':
			return cool_rate


	def free_free_cooling(self, z, T, species):
		'''
		unit: erg/cm^3/s
		'''
		g_B = 1.2
		if species == 'HII':
			ni = self.n_H(z)
		if species == 'HeII' or species == 'HeIII':
			ni = self.n_He(z) * 4

		return 1.4e-27 * T**(1/2) * self.n_e(z) * ni * g_B

    
	def tau_HeII_integrand(self, z0, z, E):
		'''
			unit: cm^-1
		'''
		if z0 < self.z_HeII_reion:
			d = 0
		else:
			d = c / self.H(z) / (1+z) * self.sigma_photo_ion(E*(1+z)/(1+z0) * self.n_He(z), 'HeII')

		return d

	def tau_HeII(self, z0, z_em, E):
		'''
			unit: cm^-1
		'''
		if z0 >= z_em:
			return 0
		else:
			z_range = np.linspace(z0, z_em, 200)
			tau = integrate.simps(np.array([self.tau_HeII_integrand(z0, z, E) for z in z_range]), z_range)

			return tau

	def J_E_integrand(self, var_list):
		'''
			unit: eV/cm^2/s
		'''

		T = var_list[0]
		z0 = var_list[1]
		z = var_list[2]
		E = var_list[3]
		alpha_QSO = var_list[4]
		C_HeIII = var_list[5]

		E0 = 54.4
		alphaB_HeIII = self.coefs_caseB(T, 'HeIII', 'coef')
		tau = self.tau_HeII(z0, z, E)
		d = c/4/np.pi * np.abs(self.dt_dz(z)) * (1+z0)**3 / (1+z)**3 * E**alpha_QSO * np.exp(-tau) \
		* (1-alpha_QSO) * E0**(1-alpha_QSO) * self.n_He(z) * self.n_e(z) * C_HeIII * alphaB_HeIII

		return d

	def J_E_term1(self, z0, E, alpha_QSO):
		'''
			unit: eV/cm^2/s
			? erg/cm^2/s/eV
		'''
		tau = self.tau_HeII(z0, self.z_HeII_reion, E)
		E0 = 54.4
		term1 = -c/4/np.pi * (1+z0)**3/(1 + self.z_HeII_reion)**3 * E**alpha_QSO * np.exp(-tau) \
		* (1-alpha_QSO) * E0**(1-alpha_QSO) * self.n_He(z0)

		return term1

	def J_E_term2(self, T, z0, E, alpha_QSO, C_HeIII):
		'''
			unit: eV/cm^2/s
		'''

		E0 = 54.4
		z_range = np.linspace(z0, self.z_HeII_reion, 30)
		var_arr = np.zeros([6, z_range.shape[0]])
		var_arr[0] = T
		var_arr[1] = z0
		var_arr[2] = z_range
		var_arr[3] = E
		var_arr[4] = alpha_QSO
		var_arr[5] = C_HeIII
		# J_E_arr = np.array([self.J_E_integrand(T, z0, z, E, alpha_QSO, C_HeIII) for z in z_range])
		J_E_arr = np.array(list(map(self.J_E_integrand, var_arr.T.tolist())))
		term2 = integrate.simps(J_E_arr, z_range)

		return term2


	def J_E(self, T, z, E, alpha_QSO,  C_HeIII):
		J_E = self.J_E_term1(z, E, alpha_QSO) + self.J_E_term2(T, z, E, alpha_QSO, C_HeIII)

		return J_E

	def dQ_integrand(self, var_list):
		'''
			unit: eV/s
			? erg/s/eV
		'''
		T = var_list[0]
		E = var_list[1]
		z = var_list[2]

		E_HeII = 54.4
		
		return (E - E_HeII)/E * self.J_E(T, z, E, self.alpha_QSO, self.C_HeIII) * self.sigma_photo_ion(E, 'HeII')

	def dQ_hard_HeII_dt(self, z, T):
		'''
			unit: erg/cm^3/s
		'''
		E_max = 150 # eV
		E_range = np.linspace(E_max, 1000, 200)

		var_arr = np.zeros([3, E_range.shape[0]])
		var_arr[0] = T
		var_arr[1] = E_range
		var_arr[1] = z

		# 4*pi factor
		d = 4 * np.pi * self.n_HeII(z) * integrate.simps(np.array(list(map(self.dQ_integrand, var_arr.T.tolist()))), E_range)

		return d


	def dQ_photo_dt(self, z, T, species, alpha):
		'''
		unit: erg/cm^3/s
		'''
		if species == 'HII':
			nX = self.n_HII(z)
		elif species == 'HeII':
			nX = self.n_HeII(z)
		elif species == 'HeIII':
			nX = self.n_HeIII(z)
		# print('nX = %f' % nX)

		alphaA = self.coefs_caseA(T, species, 'coef')

		rate = ion_energy[species] / (gamma[species] - 1 + alpha) * alphaA * nX * self.n_e(z)
		
		return rate

	def Q_Compton_dt(self, z, T):
		'''
		unit: erg/cm^3/s
		'''
		Obh2 = omega_b_0 * h_0 * h_0
		X_e = self.n_e(z) / self.n_b(z)
		
		return 6.35e-41 * Obh2 * X_e * (1+z)**7 * (2.726*(1+z) - T)
    
	def Q_cooling(self, z, T):
		'''
			Negative value
		'''
		# case A recombination cooling 
		Q_r_HeII = self.n_e(z) * self.n_HeII(z) * self.coefs_caseA(T, 'HeII', 'cool')
		Q_r_HeIII = self.n_e(z) * self.n_HeIII(z) * self.coefs_caseA(T, 'HeIII', 'cool')
		Q_r_HII = self.n_e(z) * self.n_HII(z) * self.coefs_caseA(T, 'HII', 'cool')

		# free-free cooling
		Q_ff_HII = self.n_e(z) * self.n_HII(z) * self.free_free_cooling(z, T, 'HII')
		Q_ff_HeII = self.n_e(z) * self.n_HeII(z) * self.free_free_cooling(z, T, 'HeII')
		Q_ff_HeIII = self.n_e(z) * self.n_HeIII(z) * self.free_free_cooling(z, T, 'HeIII')

		# Compton cooling
		Q_compton = self.Q_Compton_dt(z, T)

		# collisional cooling
		Q_c_HII = self.n_e(z) * self.n_HI(z) * self.coefs_coll_ion(T, 'HII', 'cool')
		Q_c_HeII = self.n_e(z) * self.n_HeI(z) * self.coefs_coll_ion(T, 'HeII', 'cool')
		Q_c_HeIII = self.n_e(z) * self.n_HeII(z) * self.coefs_coll_ion(T, 'HeIII', 'cool')

		Q = - (Q_r_HeII + Q_r_HeIII + Q_r_HII + Q_ff_HeII + Q_ff_HeIII + Q_ff_HII + Q_c_HeII + Q_c_HeIII + Q_c_HII) + Q_compton

		return Q, Q_r_HeII, Q_r_HeIII, Q_r_HII, Q_ff_HeII, Q_ff_HeIII, Q_ff_HII, Q_c_HeII, Q_c_HeIII, Q_c_HII, Q_compton


	def dQ_dt(self, T, z):
		dQ_photo = self.dQ_photo_dt(z, T, 'HII', self.alpha_bkg) + self.dQ_photo_dt(z, T, 'HeII', self.alpha_bkg) + self.dQ_photo_dt(z, T, 'HeIII', self.alpha_bkg)

		Q_cool, Q_r_HeII, Q_r_HeIII, Q_r_HII, Q_ff_HeII, Q_ff_HeIII, Q_ff_HII, Q_c_HeII, Q_c_HeIII, Q_c_HII, Q_compton = self.Q_cooling(z, T)

		dQ_HeII = 0
		# if z > self.z_HeII_reion:
		# 	dQ_HeII = 0
		# else:
		# 	dQ_HeII = self.dQ_hard_HeII_dt(z, T)

		return dQ_photo + Q_cool + dQ_HeII

	def dT_dt_hubble(self, T, z):
		return -2 * self.H(z) * T

	def dT_dz_adb(self, T, z):
		dz = 0.01
		return 2*T/3/self.Delta_gas(z, self.Delta_0, self.z_0) * (self.Delta_gas(z+dz/2, self.Delta_0, self.z_0) - self.Delta_gas(z-dz/2, self.Delta_0, self.z_0)) / dz 

	def dT_dt_n(self, T):
		T -= 0.04*T
		return T

	def dT_dt_heat_cool(self, T, z):

		return 2/3/kB/self.n_tot(z) * self.dQ_dt(T, z)

	def dT_dz(self, T, z):
		dT_dz_hubble = self.dT_dt_hubble(T,z) * self.dt_dz(z)
		dT_dz_adb = self.dT_dz_adb(T, z) 
		dT_dz_heat_cool = self.dT_dt_heat_cool(T,z) * self.dt_dz(z)

		dT_dz_total = dT_dz_hubble + dT_dz_adb + dT_dz_heat_cool
		print('At z = %.1f, T = %.2f, dT_dz_hubble = %.2f, dT_dz_adb = %.2f, dT_dz_heat_cool = %.2f' % (z, T, dT_dz_hubble, dT_dz_adb, dT_dz_heat_cool))
		return [dT_dz_total, dT_dz_hubble, dT_dz_adb, dT_dz_heat_cool]


	def T_evolution(self, T_init, z_init, z_end, dz):
		# z_range_1 = np.arange(z_init - dz, self.z_HeII_reion, -dz)
		# z_range_2 = np.arange(self.z_HeII_reion - dz, z_end, -dz)
		z_range = np.arange(z_init - dz, z_end, -dz)

		T = T_init
		T_arr = [T]
		dTs_arr = []
		Q_cooling_arr = []
		for z in z_range:
			print('z = %.1f' % z)
			Q_cooling_arr.append(self.Q_cooling(z, T))
			dT_dz = self.dT_dz(T, z) 
			dTs_arr.append(dT_dz)
			dT = dT_dz[0] * (-dz)
			T += dT
			if round(z,1) == round(self.z_HeII_reion,1):
				T = self.dT_dt_n(T)
				if self.alpha_QSO==1.7:
					T += 8100
				elif self.alpha_QSO==1.5:
					T += 8500
				elif self.alpha_QSO==1.3:
					T += 8900
			T_arr.append(T)

		# HeII reionization
		

		# T_arr.append(T)
		# for z in z_range_2:
		# 	print('z = %.1f' % z)
		# 	Q_cooling_arr.append(self.Q_cooling(z, T))
		# 	dT_dz = self.dT_dz(T, z) 
		# 	dTs_arr.append(dT_dz)
		# 	dT = dT_dz[0] * (-dz)
		# 	T += dT
		# 	T_arr.append(T)

		return np.array(T_arr), np.array(dTs_arr), np.array(Q_cooling_arr)




















