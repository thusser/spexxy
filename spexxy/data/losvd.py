import math
from typing import Tuple
import numpy as np
import scipy.misc
from numpy.polynomial import hermite

from spexxy.data.spectrum import Spectrum


class LOSVD:
    """Simple class for handling LOSVD convolution of a spectrum."""

    def __init__(self, params: Tuple):
        """Inits new LOSVD.

        Args:
            params: Parameters as (v, sig, h3, h4, h5, h6).
        """

        # init
        self._losvd = list(params)

    def kernel(self, x: np.ndarray) -> np.ndarray:
        """Creates convolution kernel for LOSVD.

        Args:
            x: x Array to create kernel on.

        Returns:
            Kernel on given x values.
        """

        # get values
        v, sig = self._losvd[:2]
        herm = self._losvd[2:]

        # helpers
        t = (x - v) / sig
        t2 = t * t

        # calculate Gaussian kernel
        k = np.exp(-0.5 * t2) / (np.sqrt(2. * np.pi) * sig)

        # Hermite polynomials normalized as in Appendix A of van der Marel &
        # Franx (1993). They are given e.g. in Appendix C of Cappellari et al. (2002)
        n = np.arange(3 + len(herm))
        nrm = np.sqrt(scipy.misc.factorial(n) * 2 ** n)

        # normalize coefficients
        c = np.array([1., 0., 0.] + herm) / nrm

        # add hermites
        k *= hermite.hermval(t, c)

        # normalize kernel
        return k / np.sum(k)

    def x(self, wave_step: float) -> np.ndarray:
        """Calculates optimal X array for kernel for given spectrum.

        Args:
            wave_step: Wavelength step size in log.

        Returns:
            Numpy array containing X array.
        """

        # get values
        v, sig = self._losvd[:2]

        # wave step in km/s
        step_kms = 299792.458 * (np.exp(wave_step) - 1.)

        # get range in pixels that the kernel should cover
        pix_range = math.floor((abs(v) + 5. * sig) / step_kms)

        # create array in pixels and convert to km/s
        return np.arange(-pix_range, pix_range + 1) * step_kms

    def __call__(self, spec: Spectrum) -> np.ndarray:
        """Convolves a given spectrum with the LOSVD kernel.

        Args:
            spec: Spectrum to convolve.

        Returns:
            Convolved flux.
        """

        # spectrum in log mode?
        if spec.wave_mode != Spectrum.Mode.LOGLAMBDA:
            raise ValueError("Spectrum must be on log wavelength.")

        # get optimal X array
        x = self.x(spec.wave_step)

        # get kernel and its half-length
        k = self.kernel(x)

        # if kernel is smaller than spectrum, we can use 'same' mode, otherwise we need to cut
        if len(k) < len(spec):
            return np.convolve(spec.flux, k, mode='same')
        else:
            # convolve
            c = np.convolve(spec.flux, k, mode='full')

            # cut original range from output
            kl = len(k) // 2
            return c[kl:-kl]


__all__ = ['LOSVD']
