import numpy as np
import scipy
from scipy import interpolate
from ct_scan import ct_scan
from ct_detect import ct_detect

def ct_calibrate(photons, material, sinogram, scale, correct=True):

	""" ct_calibrate convert CT detections to linearised attenuation
	sinogram = ct_calibrate(photons, material, sinogram, scale) takes the CT detection sinogram
	in x (angles x samples) and returns a linear attenuation sinogram
	(angles x samples). photons is the source energy distribution, material is the
	material structure containing names, linear attenuation coefficients and
	energies in mev, and scale is the size of each pixel in x, in cm."""

	# Get dimensions and work out detection for just air of twice the side
	# length (has to be the same as in ct_scan.py)
	if len(sinogram.shape) > 1:
		n = sinogram.shape[1]
		angles = sinogram.shape[0]
	else:
		n = sinogram.shape[0]

	# perform calibration
	calibration_value = ct_detect(photons, material.coeff('Air'), 2*scale*n)
	# Use log to turn intensities into attenuations
	calibrated_sinogram = -np.log(sinogram / calibration_value)

	depths = np.logspace(-3,3,1000)

	water_calibration = ct_detect(photons, material.coeff('Water'), depths)
	water_calibration_air = ct_detect(photons, material.coeff('Air'), depths)

	water_coeff = -np.log(water_calibration/water_calibration_air)
	fn = scipy.interpolate.interp1d(water_coeff, depths, fill_value = 'extrapolate')

	return fn(calibrated_sinogram)


