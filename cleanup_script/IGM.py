
import numpy as np
import argparse
import LCDMSphere
import Energy_loss_class

"""
This script will generator 5 npy file: energy density utot, energy per baryon in eV E_eV,
the number of electron per volume per erg N, pressure in erg/cm^3 P_e, and pressure in eV/cm^3 P_eV

run python IGM.py --nstep --mstep --E_min --E_max --source_model --deltalin0

the default setting is reshift bins nstep = 399, energy bins mstep = 399, E_min = 1e-11 erg, E_max = 1e-3 erg,

source_model = '0' for Khaire's model ('1' for Haardt's mode)

and overdensity evolution deltalin0 = 0 for mean density (> 0 for overdensity, and < 0 for underdensity)

Please make sure the parameters are consistent with get_source.py
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


"""
To use this class, call IGM_N(deltalin0, model)
where deltalin0 is overdensity factor (<0 means underdensity, 0 means mean density, >0 means overdensity),
and model is either 0 or 1 (0 using Khaire's EBL model and 1 using Haardt's EBL model)

all variables should be individual values rather than array or list

.get_utot() will return to the energy density utot and energy per baryon in eV E_eV
.get_P() will return to the number of electron per volume per erg N, pressure in erg cm^-3 P_e and pressure in eV cm^-3 P_eV
"""


class IGM_N:
    def __init__(self, deltalin0, model, nstep, mstep, E_min, E_max):
        if model == '0': # Khaire's model
            self.filename = "source_term_Khaire.txt"
        elif model == '1': # Haardt's model
            self.filename = "source_term_Haardt.txt"
        else:
            raise ValueError("choose 0 for Khaire's model, and 1 for Haardt's model")
            
        self.nstep = nstep
        self.mstep = mstep
        
        self.E = np.logspace(np.log10(E_min), np.log10(E_max), self.mstep + 1)

        E_mid = (self.E[:-1] + self.E[1:]) / 2 # midpoint for reference
        self.E_plus = np.zeros((self.mstep+1, 1)) # E_{i+1/2} = sqrt(E[i] * E[i+1])
        self.E_minus = np.zeros((self.mstep+1, 1)) # E_{i-1/2} = sqrt(E[i-1] * E[i])

        for i in range(self.mstep):
            self.E_plus[i] = np.sqrt(self.E[i] * self.E[i+1]) # length len(E) -1

        self.E_plus[-1] = self.E_plus[-2] * self.E[-1] / self.E[-2] # E[i+1]/E[i] is constant, so in E_plus and E_minus

        for i in range(1, self.mstep+1):
            self.E_minus[i] = np.sqrt(self.E[i] * self.E[i-1])
    
        self.E_minus[0] = self.E_minus[1] / (self.E[1] / self.E[0])
        
        # calculate overdensity evolution
        C = LCDMSphere.XCosmo(0.674, 0.315)
        C.overdensity_evol(15., deltalin0, self.nstep+1)
        self.z_list = C.z_grid # redshift grid
        delta_b = C.Delta_grid # overdensity of baryon
        Delta_list = np.array([x for x in delta_b]) # relative overdensity of baryon - grid
        ln_Delta = [np.log(x) for x in Delta_list]

        dDdz = np.zeros((self.nstep, 1)) # d ln(Delta)/dz
        for i in range(self.nstep):
            dDdz[i] = (ln_Delta[i+1] - ln_Delta[i]) / (self.z_list[i+1] - self.z_list[i])

        self.z = (self.z_list[:-1] + self.z_list[1:]) / 2 # midpoint of z grid, correspond to d ln(Delta)/dz
    
        self.Delta = (Delta_list[:-1] + Delta_list[1:]) / 2 # midpoint of overdensity grid, correspond to d ln(Delta)/dz
    
        self.H = np.zeros((self.nstep, 1))
        for i in range(self.nstep):
            self.H[i] = C.H(self.z[i])
    
        self.theta = np.zeros((self.nstep, 1))
        for i in range(self.nstep):
            self.theta[i] = -3 / (1 + self.z[i]) + dDdz[i]
            
        self.S = np.loadtxt(self.filename) # in 'same columns, same redshift; same row, same energy' format 
        self.S = np.delete(self.S, 0, axis=1) 
        self.Sz = [[self.S[i][j] / (-(1 + self.z[j]) * self.H[j]) * self.Delta[j] for j in range(self.nstep)] for i in range(self.mstep)]
        
        self.SzdE = [[self.Sz[i][j] * (self.E_plus[i] - self.E_minus[i]) for j in range(self.nstep)] for i in range(self.mstep)]
        self.SzdE = np.array(self.SzdE).reshape(self.mstep,self.nstep)
            
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
        M = np.zeros((self.mstep, self.mstep))

        for i in range(self.mstep):
            for j in range(self.mstep):
                if j == i or j == i+1:
                    A1_ij = -theta
        
                    A2_ij = self.A2(self.E[j], z, Delta, theta, H)
                    A2_ij_p = self.A2(self.E_plus[j], z, Delta, theta, H)
                    A2_ij_m = self.A2(self.E_minus[j], z, Delta, theta, H)
        
                    dE = self.E[j+1] - self.E[j]
                    dE_p = self.E_plus[j+1] - self.E_plus[j]
                    dE_m = self.E_minus[j+1] - self.E_minus[j]
        
                    if j == i: # subcribe of n is equal to j
                        M[i][j] += (A1_ij - A2_ij_m / dE_m).item()
                    if j == i+1:
                        M[i][j] += (A2_ij_m / dE_m).item()
                
                if self.E[j] >= 2 * self.E[i]:
                    M[i][j] += (self.A3(self.E[i], self.E[j], z, H) * (self.E_plus[i] - self.E_minus[i])).item() # integrate E_p from 2E
                
        return M
    
    def back_Euler(self): # solve dn/dz = Mn + S using backward Euler method, M is square matrix, S is column vector
        n = np.zeros((self.mstep, 1))
        I = np.identity(self.mstep)
    
        for j in range(1, self.nstep): # initialize the first column to 0 with z[0], start from z[1]
            n_jm = np.zeros((self.mstep,1)) # partition column vector with redshift z[j-1]
            for i in range(self.mstep):
                n_jm[i] = n[i][j-1]
        
            dz = self.z[j] - self.z[j-1]
        
            S_j = np.zeros((self.mstep, 1))
            for i in range(self.mstep):
                S_j[i] = self.SzdE[i][j] # under same redshift, vari from different energy
        
            dzS = dz*S_j
            n_jm_plus_dzS = np.zeros((self.mstep, 1))
            for i in range(self.mstep):
                n_jm_plus_dzS[i] = n_jm[i] + dzS[i]
            
            M_j = self.get_M(self.z[j], self.Delta[j], self.theta[j], self.H[j]) # M matrix only depend on redshift
            
            dzM = dz*M_j
            I_minus_dzM = np.zeros((self.mstep,self.mstep))
            for i in range(self.mstep):
                for k in range(self.mstep):
                    I_minus_dzM[i][k] = I[i][k]-dzM[i][k]
                
            I_m_inv = np.linalg.inv(I_minus_dzM)
        
            n_j = I_m_inv@n_jm_plus_dzS
    
            n = np.hstack((n,n_j.reshape(self.mstep,1)))
        
        return n
    
    def get_utot(self, n):
        if n is None:
            n = self.back_Euler()
        utot = np.sum(n * self.E[:-1,None], axis=0) # erg cm^-3
        E_eV = np.zeros((self.nstep, 1))
        nbary = np.zeros((self.nstep, 1))
        for j in range(self.nstep):
            nbary[j] = omega_b_0 * (1 + self.z_list[j])**3 * self.Delta[j] * rho_crit / m_p # number density of baryon
            E_eV[j] = utot[j] / nbary[j] / 1.602e-12 # energy per baryon in eV
        return utot, E_eV
    
    def get_P(self, n):
        if n is None:
            n = self.back_Euler()
        
        N = np.zeros((self.mstep, self.nstep))
        for i in range(self.mstep):
            for j in range(self.nstep):
                N[i][j] = (n[i][j] / (self.E_plus[i] - self.E_minus[i])).item()
                
        P_e = np.zeros((self.nstep, 1))
        for j in range(self.nstep):    
            for i in range(self.mstep):
                P_e[j] = P_e[j] + N[i][j] * (self.E_plus[i] * (self.E_plus[i]+2*E_e)/(self.E_plus[i]+E_e)) * (self.E[i+1]-self.E[i]) / 3
                
        P_eV = [x / 1.602e-12 for x in P_e]
        return N, P_e, P_eV

def main(args):
    IGM = IGM_N(args.deltalin0, args.source_model, args.nstep, args.mstep, args.E_min, args.E_max)

    n = IGM.back_Euler()

    utot, E_eV = IGM.get_utot(n)
    N, P_e, P_eV = IGM.get_P(n)

    np.save(f'utot_{args.deltalin0}_{args.source_model}.npy', utot)
    np.save(f'E_eV_{args.deltalin0}_{args.source_model}.npy', E_eV)
    np.save(f'N_{args.deltalin0}_{args.source_model}.npy', N)
    np.save(f'P_e_{args.deltalin0}_{args.source_model}.npy', P_e)
    np.save(f'P_eV_{args.deltalin0}_{args.source_model}.npy', P_eV)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nstep', type=int, help='number of redshift bins', default=399)
    parser.add_argument('--mstep', type=int, help='number of Energy bins', default=399)

    parser.add_argument('--E_min', type=float, help='minimum energy in erg', default=1e-11)
    parser.add_argument('--E_max', type=float, help='maximum energy in erg', default=1e-3)

    parser.add_argument('--source_model', type=str, help="0 for Khaire's model, 1 for Haardt's model", default='0')

    parser.add_argument('--deltalin0', type=float, help='overdensity evolution, 0 for mean density, > 0 for overdensity, and < 0 for underdensity', default=0)

    args = parser.parse_args()
    
    main(args)
