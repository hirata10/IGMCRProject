import numpy as np
from scipy import integrate
from scipy import interpolate
# import matplotlib.pyplot as plt
from multiprocessing import Pool
import pickle as pkl
import sys
from functools import partial
import itertools
import timeit
import argparse



# import Source_class

z_Mrk421 = 0.031


r0 = 2.818e-13 # classical electron radius in cm
m_e = 9.10938e-28 # electron mass in gram
c = 2.998e10 # speed of light in cm/s
h = 6.6261e-27 # planck constant in cm^2 g s-1
# z = 2
eV_to_erg = 1.6022e-12
parsec_to_cm=3.085677581e18 # cm per parsec

# J_nu_2 = np.loadtxt('./EBL_KS18_Q20_z_ 2.0.txt')
# J_nu_2_func = interpolate.interp1d(c*1.e8/J_nu_2.T[0], J_nu_2.T[1])


def D(z):
	'''
	in units of Mpc
	'''
	h = 0.674
	OM = 0.315

	zs = np.linspace(0,z,1000,endpoint=True)
	chi = 3.e5/(100*h)*integrate.simps(1./np.sqrt(1-OM+OM*(1+zs)**3),zs)

	return chi


class beam(object):
	def __init__(self, EBL_filepath, readPATH):
		'''
		E_photo_arr in eV
		'''
		self.EBL_filepath = EBL_filepath

		E_photo_arr = np.logspace(10,14, 200) / 1.e12
		self.get_alpha_eff_func(f'{readPATH}alpha_eff_full.pkl', E_photo_arr)
		self.J_nu_func = self.get_Jnu_func(EBL_filepath)

		# theta_e_arr = np.logspace(-8,0, 400)
		# self.gamma_1_arr = E_photo_arr/m_e/c**2

	def load_Jnu_z(self, path):
		J_nu = np.loadtxt(path)

		return J_nu

	def get_Jnu_func(self, path):
		J_nu = self.load_Jnu_z(path)
		J_nu_func = interpolate.interp1d(c*1.e8/J_nu.T[0], J_nu.T[1])

		return J_nu_func

	def J_integrand(self, nu):

		return (m_e*c**2)**2/(h*nu)**3 * self.J_nu_func(nu)

	def dN_dt_dOmega_dgammae(self, gamma_e, gamma_1, theta_e):
		smooth = 1.e-8
	#     gamma_e, gamma_1, theta_e = var_list
		nu_lower = gamma_1 * (1+ gamma_e**2*theta_e**2)/4/gamma_e/(gamma_1-gamma_e+smooth)*m_e*c**2/h
		if nu_lower > 2.99e26:
			return 0
		else:
			nu_arr = np.logspace(np.log10(nu_lower),26, 1000)
		#     z_nu_list = np.zeros([2, nu_arr.shape[0]])
		#     z_nu_list[0] = z
		#     z_nu_list[1] = nu_arr
		J_int = integrate.simps(np.array(list(map(lambda nu: self.J_integrand(nu), nu_arr))), nu_arr)

		return 2 * np.pi * r0**2 * gamma_e/gamma_1/(gamma_1-gamma_e) * (-1 + gamma_1**2/2/gamma_e/(gamma_1-gamma_e)+2*gamma_e**2*theta_e**2/(1+gamma_e**2 * theta_e**2)**2) * J_int

	def dN_dt_dgammae(self, gamma_e, gamma_1, theta_e_arr):
	#     var_arr= generate_var_arr(gamma_e, gamma_1, theta_e_arr)
		print('gamma_e= %.5e, gamma_1=%.1f' % (gamma_e, gamma_1))
	    # with Pool() as pool:
		# 	integrand = 2 * np.pi * np.array(pool.map(partial(dN_dt_dOmega_dgammae, gamma_e=gamma_e, gamma_1=gamma_1), theta_e_arr))

		integrand = 2 * np.pi * np.array(list(map(lambda theta_e: self.dN_dt_dOmega_dgammae(gamma_e, gamma_1, theta_e), theta_e_arr)))
		I = integrate.simps(integrand * theta_e_arr, theta_e_arr)

		return I

	def alpha_eff(self, gamma_1, theta_e_arr):
		E_1 = gamma_1 * m_e * c**2  / eV_to_erg
		gamma_e_arr = np.logspace(8, np.log10(E_1), int((np.log10(E_1)-8)/np.log10(1.05)), endpoint=False)*eV_to_erg/m_e/c**2
		print('num_bins = %d' % int((np.log10(E_1)-8)/np.log10(1.05)))
		with Pool() as pool:
			integrand = np.array(pool.map(partial(self.dN_dt_dgammae,gamma_1=gamma_1, theta_e_arr=theta_e_arr) , gamma_e_arr))

		# integrand = np.array(list(map(lambda gamma_e: dN_dt_dgammae(gamma_e, gamma_1, theta_e_arr), gamma_e_arr)))
		I = integrate.simps(1./c * integrand, gamma_e_arr)

		return I

	def get_alpha_eff_func(self, file, E_photo_arr):
		'''
		E_photo_arr: in units of TeV
		'''
		with open(file, 'rb') as f:
			alpha_arr = pkl.load(f)

		# E_photon_arr = self.E_photo_arr
		self.alpha_eff_func = interpolate.interp1d(E_photo_arr, np.array(alpha_arr))

		# return alpha_eff_func

	def dNdE_PL(self, E, N0 = 6.6e-12, E0 = 1, alpha = 2.61):
		'''
		Mrk 501
		N0: 6.6 * 10^-12 TeV^-1 cm^-2 s^-1
		alpha: 2.61
		E: units TeV
		'''

		return N0 * (E/E0)**(-alpha) #* np.exp(-tau(E,z))

	def dNdE_PL_CO(self, E, N0 = 4.e-11, E0=1, alpha = 2.26, E_c = 5.1):
		'''
		Mrk 421
		N0: 4.0 * 10^-11 TeV^-1 cm^-2 s^-1
		alpha: 2.26
		E_c: 5.1 TeV
		E: units TeV
		'''

		return N0 * (E/E0)**(-alpha) * np.exp(-E/E_c) # * np.exp(-tau(E,z))


	def prod_rate_integrand(self, E, gamma_e, theta_e, r):
		'''
		E: [erg]
		r_fid = 1 Mpc
		'''
		# r = r * 1.e6 * parsec_to_cm
		gamma_1 = E / m_e / c**2
		E = E / eV_to_erg / 1.e12 # TeV
		return self.dNdE_PL_CO(E) * E * D(z_Mrk421)**2 / r**2 / c * np.exp(-self.alpha_eff_func(E) * r) * self.dN_dt_dOmega_dgammae(gamma_e, gamma_1, theta_e)

	def prod_rate(self, gamma_e, theta_e, r):
		#print('gamma_e = %.2f, theta_e = %.3e' % (gamma_e, theta_e))
	#     with Pool() as pool:
	#         integrand = np.array(pool.map(partial(prod_rate_integrand, r=r, gamma_e=gamma_e, theta_e = theta_e), E_photon_arr))
		E_e = gamma_e * m_e * c**2 / eV_to_erg

		E_photon_arr = np.logspace(np.log10(np.where(E_e>1.e10, E_e, 1.e10)), 14, 200) * eV_to_erg
		integrand = np.array(list(map(lambda E: self.prod_rate_integrand(E, gamma_e, theta_e, r), E_photon_arr)))
		I = integrate.simps(integrand, E_photon_arr)

		return I


def main(args):
	z = args.z
	r = args.r
	path = args.source_file

	beam_obj = beam(path, args.readPATH)
	prod_rate = partial(beam_obj.prod_rate, r = r)

	gamma_e_arr = np.logspace(8, 14, 400) * eV_to_erg / m_e / c**2
	theta_e_arr = np.logspace(-8, 0, 400)

	# All combinations of gamma_e and theta_e
	gamma_theta_combinations = list(itertools.product(gamma_e_arr, theta_e_arr))

	start = timeit.default_timer()
	with Pool() as pool:
		rate = pool.starmap(prod_rate, gamma_theta_combinations)

	print('Time: {:.2f} min'.format((timeit.default_timer() - start) / 60))

	with open(f'{args.savePATH}rate_z%.1f_%dMpc.pkl' % (z, r), 'wb') as f:
		pkl.dump(rate, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--z', type=float, help='desired redshift', default=2)
    parser.add_argument('--r', type=float, help='distance to the blazar in Mpc', default=10)
    parser.add_argument('--source_file', type=str, help='source file corresponding to the redshift', default='KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 2.0.txt')    
    parser.add_argument('--readPATH', type=str, help='reading path of attenuation coefficient', default='')
    parser.add_argument('--savePATH', type=str, help='saving path for the result', default='')

    args = parser.parse_args()
    
    main(args)




