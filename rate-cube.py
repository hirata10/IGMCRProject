import numpy
import pickle
from astropy.io import fits

with open('rate_10Mpc.pkl', 'rb') as f:
  rate_arr = pickle.load(f)

gamma_e_arr = numpy.logspace(8, 14, 400) / 5.11e5
theta_e_arr = numpy.logspace(-8,0, 400)
dlng = numpy.log(gamma_e_arr[1]/gamma_e_arr[0])
dlnt = numpy.log(theta_e_arr[1]/theta_e_arr[0])
print(dlng,dlnt)

rate_arr = rate_arr.reshape((400,400))
rate_arr[-1,:] = 0.
rate_arr = numpy.clip(rate_arr,1e-100,1e100)
fits.PrimaryHDU(numpy.log10(rate_arr)).writeto('rate-image_log10.fits', overwrite=True)
rate_arr_scale = rate_arr * gamma_e_arr[:,None] * 2.*numpy.pi*theta_e_arr[None,:]**2
fits.PrimaryHDU(rate_arr_scale).writeto('rate-image-scale.fits', overwrite=True)

print('total rate =', numpy.sum(rate_arr_scale)*2*dlng*dlnt)

