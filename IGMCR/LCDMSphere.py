import numpy

# makes a cosmology class for spherical overdensity perturbations
class XCosmo:
  def __init__(self, h, OmM):
    self.H0 = h/3.08567758e17 # Hubble constant in s^-1
    self.OmM = OmM
    self.OmMH2 = OmM*self.H0**2
    self.OmLH2 = (1.-OmM)*self.H0**2

  # Hubble constant in units of s^-1
  def H(self, z):
    return(self.H0*numpy.sqrt(self.OmM*(1+z)**3 + 1.-self.OmM))

  # overdensity evolution ODE
  # step from z1 to z2<z1
  # R = a*xi, delta = xi^{-3} - 1
  # xi_dotdot = -2*H*xi_dot + OmMH2/(2a^3)*(xi-1/xi^2)
  #
  # input is xv = [xi, d(xi)/d(ln a)] at z1; output is deltat (for step) and xv at z2
  def Rdd(self, z1, z2, xv):
    zave = (z1+z2)/2.
    dt = numpy.log((1+z1)/(1+z2))/self.H(zave)
    # step to average
    Hz1 = self.H(z1)
    xi = xv[0] + xv[1]*Hz1*dt/2.
    xi_dot = xv[1]*Hz1
    xi_dot *= (1+z2)/(1+z1)
    xi_dot += self.OmMH2/2.*(1+zave)**3*(xi-1./xi**2) * dt
    xi_dot *= (1+z2)/(1+z1)
    return dt, [xi+xi_dot*dt/2., xi_dot / self.H(z2)]

  # compute overdensities and time steps for evolution
  # from zi --> 0 with nstep steps, log spaced
  # linear overdensity delta_lin
  # Delta_grid becomes the densities (scaled to mean=1, not subtracted) at each grid point
  def overdensity_evol(self, zi, delta_lin, nstep):
    self.z_grid = numpy.exp(numpy.linspace(0, numpy.log(1.+zi), nstep))[::-1]-1.
    self.Delta_grid = numpy.zeros((nstep,))
    xv = [1.-delta_lin/3./(1+zi), -delta_lin/3./(1+zi)]
    self.Delta_grid[0] = 1./xv[0]**3
    for i in range(nstep-1):
      dt,xv = self.Rdd(self.z_grid[i], self.z_grid[i+1], xv)
      self.Delta_grid[i+1] = 1./xv[0]**3

def test():
  c = XCosmo(0.6774, 0.319)
  print(c.H(0), c.H(1), c.H(8))

  # overdensity grid
  c.overdensity_evol(50., -10., 200)
  print(c.z_grid)
  print(c.Delta_grid)

#test()
