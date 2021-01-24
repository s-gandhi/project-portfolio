import math
import numpy as np
import numpy.matlib
import sys
from ct_lib import *


def ramp_filter(sinogram, scale, alpha=0.001):
    """ Ram-Lak filter with raised-cosine for CT reconstruction

    fs = ramp_filter(sinogram, scale) filters the input in sinogram (angles x samples)
    using a Ram-Lak filter.

    fs = ramp_filter(sinogram, scale, alpha) can be used to modify the Ram-Lak filter by a
    cosine raised to the power given by alpha."""

    # get input dimensions
    angles = sinogram.shape[0]
    n = sinogram.shape[1]

    # Set up filter to be at least twice as long as input
    m = np.ceil(np.log(2 * n - 1) / np.log(2))
    m = int(2 ** m)

    # apply filter to all angles
    print('Ramp filtering')

    ft_p = np.fft.fft(sinogram, n=m, axis=1)

    if alpha:
        filtered_ft = ft_p * raised_cosine(alpha, m, scale)
    else:
        filtered_ft = ft_p * ram_lak(m, scale)

    filtered_sinogram = np.fft.ifft(filtered_ft)
    filtered_sinogram = np.delete(filtered_sinogram, np.arange(m - n, m, dtype=int), axis=1)
    return np.real(filtered_sinogram)


def ram_lak(n, scale):
    w = np.fft.fftfreq(n, scale)
    w = np.abs(w)
    w[0] = w[1] / 6
    return w


def raised_cosine(alpha, n, scale):
    w = np.fft.fftfreq(n, scale)
    cos = np.maximum(np.cos(scale * w * np.pi), np.zeros(n))
    response = np.abs(w) * np.power(cos, alpha)
    response[0] = response[1] / 6
    return response
