import numpy as np
from attenuate import *
from ct_calibrate import *
from ramp_filter import ramp_filter
from back_project import back_project


def hu(p, material, reconstruction, scale):
    """ convert CT reconstruction output to Hounsfield Units
        calibrated = hu(p, material, reconstruction, scale) converts the reconstruction into Hounsfield
        Units, using the material coefficients, photon energy p and scale given."""
    # use water to calibrate
    water = ct_detect(p, material.coeff('Water'), 1)
    water = ct_calibrate(p, material, water, scale)[0]

    reconstruction = 1000 * (reconstruction - water) / water

    # use result to convert to hounsfield units
    # limit minimum to -1024, which is normal for CT data.

    return reconstruction


