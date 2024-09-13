import numpy as np
import bisect
import argparse
import LCDMSphere
import Energy_loss_class

"""
This script will generator the source file, run

python get_source.py --nstep --mstep --E_min --E_max --source_model

the default setting is reshift bins nstep = 399, energy bins mstep = 399, E_min = 1e-11 erg, E_max = 1e-3 erg,

and source_model = '0' for Khaire's model, '1' for Haardt's model
"""

r0 = 2.818e-13 # classical electron radius in cm
m_e = 9.10938e-28 # electron mass in gram
c = 2.998e10 # speed of light in cm/s
h = 6.6261e-27 # planck constant in cm^2 g s-1
k = 1.3808e-16 # boltzmann constant in cm^2 g s^-2 K^-1
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


# # This class loads Khaire's model

class Khaire:
    def __init__(self):
        self.z_for_fit_p = []
        self.wavelength_for_fit_p = []
        self.EBL_for_fit_p = []

    def read_file(self, filename, z):
        with open(filename) as f:
            lines = f.readlines()[3:]
            for line in lines:
                data = line.split()
                for x in range(0, len(data), 2):
                    self.z_for_fit_p.append(float(z))
                    self.wavelength_for_fit_p.append(float(data[x]))
                    self.EBL_for_fit_p.append(float(data[x+1]))
                
    def read_data(self):
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 0.0.txt',0)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 0.1.txt',0.1)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 0.2.txt',0.2)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 0.3.txt',0.3)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 0.4.txt',0.4)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 0.5.txt',0.5)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 0.6.txt',0.6)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 0.7.txt',0.7)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 0.8.txt',0.8)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 0.9.txt',0.9)

        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 1.0.txt',1.0)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 1.1.txt',1.1)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 1.2.txt',1.2)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 1.3.txt',1.3)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 1.4.txt',1.4)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 1.5.txt',1.5)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 1.6.txt',1.6)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 1.7.txt',1.7)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 1.8.txt',1.8)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 1.9.txt',1.9)

        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 2.0.txt',2.0)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 2.1.txt',2.1)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 2.2.txt',2.2)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 2.3.txt',2.3)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 2.4.txt',2.4)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 2.5.txt',2.5)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 2.6.txt',2.6)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 2.7.txt',2.7)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 2.8.txt',2.8)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 2.9.txt',2.9)

        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 3.0.txt',3.0)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 3.1.txt',3.1)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 3.2.txt',3.2)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 3.3.txt',3.3)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 3.4.txt',3.4)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 3.5.txt',3.5)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 3.6.txt',3.6)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 3.7.txt',3.7)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 3.8.txt',3.8)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 3.9.txt',3.9)

        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 4.0.txt',4.0)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 4.1.txt',4.1)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 4.2.txt',4.2)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 4.3.txt',4.3)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 4.4.txt',4.4)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 4.5.txt',4.5)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 4.6.txt',4.6)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 4.7.txt',4.7)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 4.8.txt',4.8)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 4.9.txt',4.9)

        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 5.0.txt',5.0)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 5.1.txt',5.1)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 5.2.txt',5.2)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 5.3.txt',5.3)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 5.4.txt',5.4)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 5.5.txt',5.5)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 5.6.txt',5.6)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 5.7.txt',5.7)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 5.8.txt',5.8)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 5.9.txt',5.9)

        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 6.0.txt',6.0)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 6.2.txt',6.2)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 6.4.txt',6.4)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 6.6.txt',6.6)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 6.8.txt',6.8)

        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 7.0.txt',7.0)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 7.2.txt',7.2)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 7.4.txt',7.4)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 7.6.txt',7.6)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 7.8.txt',7.8)

        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 8.0.txt',8.0)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 8.2.txt',8.2)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 8.4.txt',8.4)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 8.6.txt',8.6)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 8.8.txt',8.8)

        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 9.0.txt',9.0)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 9.2.txt',9.2)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 9.4.txt',9.4)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 9.6.txt',9.6)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_ 9.8.txt',9.8)

        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_10.0.txt',10.0)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_10.5.txt',10.5)

        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_11.0.txt',11.0)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_11.5.txt',11.5)

        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_12.0.txt',12.0)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_12.5.txt',12.5)

        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_13.0.txt',13.0)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_13.5.txt',13.5)

        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_14.0.txt',14.0)
        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_14.5.txt',14.5)

        self.read_file('./KS_2018_EBL/Fiducial_Q18/EBL_KS18_Q18_z_15.0.txt',15.0)

    def get_data(self):
        self.read_data()

        z_for_fit = np.array(self.z_for_fit_p[::695])
        wavelength_for_fit = np.array(self.wavelength_for_fit_p[:695])
        EBL_for_fit_p = np.array(self.EBL_for_fit_p)
        EBL_for_fit = EBL_for_fit_p.reshape(91, 695)

        return z_for_fit, wavelength_for_fit, EBL_for_fit


# # This class loads Haardt's model

class Haardt:
    def __init__(self):
        # read data from file
        data = np.loadtxt('UVB.out.txt', skiprows=20)

        # extract z values from the first row
        self.z_for_fit = data[0][:60] # (60, )

        # extract w values from the first column of the rest of the rows
        self.wavelength_for_fit = data[1:,:1].flatten() # (575, )

        # extract J values from the rest of the rows
        self.EBL_for_fit = data[1:,1:] 

        # transpose J to match the order of w and z
        self.EBL_for_fit = self.EBL_for_fit.T # (60, 575)


# # Main class for source term class

class Source:
    def __init__(self, z, E, z_list, wavelength, EBL):
        self.z = z
        self.E = E
        self.z_list = z_list # redshift
        self.wavelength = wavelength # in Angstrom, 10^-8 cm
        self.EBL = EBL # specific intensity J_nu in erg/s/cm^2/Hz/Sr
        
    def find_closest_indices(self, X, x):
        i = bisect.bisect_left(X, x)
        if i == 0:
            return i, i+1
        elif i >= len(X)-1:
            return i-1, i
        else:
            if x - X[i-1] < X[i] - x:
                return i-1, i
            else:
                return i, i+1
    
    def EBL_fit(self, z_input, w_input):
        j, j1 = self.find_closest_indices(self.z_list, z_input)
        k, k1 = self.find_closest_indices(self.wavelength, w_input)
    
        f_A = self.EBL[j][k] + (self.z - self.z_list[j]) / (self.z_list[j1] - self.z_list[j]) * (self.EBL[j1][k] - self.EBL[j][k])
        f_B = self.EBL[j][k1] + (self.z - self.z_list[j]) / (self.z_list[j1] - self.z_list[j]) * (self.EBL[j1][k1] - self.EBL[j][k1])
        f_xy = f_A + (w_input - self.wavelength[k]) / (self.wavelength[k1] - self.wavelength[k]) * (f_B - f_A)
    
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


# # Calculate the source term and write into file

def main(args):
    nstep = args.nstep
    mstep = args.mstep

    E_min = args.E_min
    E_max = args.E_max

    C = LCDMSphere.XCosmo(0.674, 0.315)
    C.overdensity_evol(15., -10., nstep+1)
    z_list = C.z_grid # redshift grid
    delta_b = C.Delta_grid # overdensity of baryon
    Delta_list = np.array([1 + x for x in delta_b]) # relative overdensity of baryon - grid
    ln_Delta = [np.log(x) for x in Delta_list]

    dDdz = np.zeros((len(z_list) - 1, )) # d ln(Delta)/dz
    for i in range(len(z_list) - 1):
        dDdz[i] = (ln_Delta[i+1] - ln_Delta[i]) / (z_list[i+1] - z_list[i])

    z = (z_list[:-1] + z_list[1:]) / 2 # midpoint of z grid, correspond to d ln(Delta)/dz
    
    Delta = (Delta_list[:-1] + Delta_list[1:]) / 2 # midpoint of overdensity grid, correspond to d ln(Delta)/dz
    
    H = np.zeros((len(z), ))
    for i in range(len(z)):
        H[i] = C.H(z[i])
    
    theta = np.zeros((len(z), ))
    for i in range(len(theta)):
        theta[i] = 3 * H[i] - dDdz[i]

    E = np.logspace(np.log10(E_min), np.log10(E_max), mstep + 1)

    E_plus = np.zeros((mstep+1, 1)) # E_{i+1/2} = sqrt(E[i] * E[i+1])
    E_minus = np.zeros((mstep+1, 1)) # E_{i-1/2} = sqrt(E[i-1] * E[i])

    for i in range(mstep):
        E_plus[i] = np.sqrt(E[i] * E[i+1]) # length len(E) -1

    E_plus[-1] = E_plus[-2] * E[-1] / E[-2] # E[i+1]/E[i] is constant, so in E_plus and E_minus

    for i in range(1, mstep+1):
        E_minus[i] = np.sqrt(E[i] * E[i-1])
    
    E_minus[0] = E_minus[1] / (E[1] / E[0])

    if args.source_model == '0':
        output_file = 'source_term_Khaire.txt'
        source = Khaire()
        z_for_fit, wavelength_for_fit, EBL_for_fit = source.get_data()
    elif args.source_model == '1':
        output_file = 'source_term_Haardt.txt'
        source = Haardt()
        z_for_fit, wavelength_for_fit, EBL_for_fit = source.z_for_fit, source.wavelength_for_fit, source.EBL_for_fit
    else:
        raise ValueError("choose 0 for Khaire's model, and 1 for Haardt's model")

    for i in range(mstep):
        with open(output_file, 'a') as f:
            f.write(str(i)+' ')
        f.close()
    
        for j in range(nstep):
            print('i=', i, ' j=', j)
            S_class = Source(z[j], E[i], z_for_fit, wavelength_for_fit, EBL_for_fit)
            # result in "same z, same columns", "same E, same raw" format
            S_ij = S_class.Source_term()
            with open(output_file, 'a') as f:
                f.write(str(S_ij)+' ')
                if j == mstep - 1:
                    f.write('\n')
            f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nstep', type=int, help='number of redshift bins', default=399)
    parser.add_argument('--mstep', type=int, help='number of Energy bins', default=399)

    parser.add_argument('--E_min', type=float, help='minimum energy in erg', default=1e-11)
    parser.add_argument('--E_max', type=float, help='maximum energy in erg', default=1e-3)

    parser.add_argument('--source_model', type=str, help="0 for Khaire's model, 1 for Haardt's model", default='0')

    args = parser.parse_args()
    
    main(args)
