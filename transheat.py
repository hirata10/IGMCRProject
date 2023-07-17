import numpy

# constants
Tcmb = 2.725*3
kB = 1.380649e-16
h = 6.62607015e-27
c = 2.99792458e10
me = 9.1093837015e-28
eV_erg = 1.602176634e-12

def compton_f(x):
  if x>1e-2:
    return ((-24-144*x-290*x**2-194*x**3+20*x**4+28*x**5)/3/x**2/(1+2*x)**3+(4+4*x-x**2)/x**3*numpy.log(1+2*x))
  return 8/5*x**2 - 32/5*x**3 + 2248/105*x**4 - 1376/21*x**5 + 11840/63*x**6

def compton_FF(x):
  n = 1024
  y = numpy.linspace(x/(2*n),x-x/(2*n),n)
  result = 0.
  for i in range(n): result += compton_f(y[i])*y[i]
  result *= x/n
  return result

jtable = numpy.loadtxt('KS_2018_EBL/Q20/EBL_KS18_Q20_z_ 2.0.txt')
nstep = numpy.shape(jtable)[0]-1
jtable[:,0] = 2.99792458e18/jtable[:,0] # convert to Hz
nu = (jtable[:-1,0]+jtable[1:,0])/2
dnu = (jtable[:-1,0]-jtable[1:,0])
JnuEBL = (jtable[:-1,1]+jtable[1:,1])/2

xx = h*nu/kB/Tcmb
Bnu = 2*h*nu**3/c**2/numpy.expm1(xx)
Jnu = JnuEBL+Bnu
numpy.savetxt('spec.dat', numpy.concatenate((nu, Bnu, JnuEBL)).reshape(3,nstep).T)
print('#', numpy.sum(Bnu*dnu*4*numpy.pi/c), numpy.sum(JnuEBL*dnu*4*numpy.pi/c))

gamma = numpy.logspace(8,14,400)/(me*c**2/eV_erg)
gamma = gamma[0] * (gamma[1]/gamma[0])**numpy.linspace(0,499,500)
sumsq = 0.
for i in range(500)[::-1]:
  all_int = 0
  factors = Bnu/nu**3*dnu
  for j in range(nstep):
    all_int += compton_FF(2*gamma[i]*h*nu[j]/me/c**2)*factors[j]
  tp2 = 135*me*c**2/128/numpy.pi**4/gamma[i]**3 * (me*c**2/kB/Tcmb)**4 * all_int
  sumsq += tp2/(me*c*gamma[i])**2 * numpy.log(gamma[1]/gamma[0])

  print('{:11.5E} {:11.5E} {:11.5E} {:11.5E}'.format(gamma[i], tp2/(me*c*gamma[i])**2, tp2/(me*c)**2, sumsq**.5))
