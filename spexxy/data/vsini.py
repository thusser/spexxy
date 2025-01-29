import numpy as np
from typing import Tuple
from PyAstronomy import pyasl

from spexxy.data.spectrum import Spectrum


class Vsini:

    def __init__(self, params: Tuple, fast: bool = True):

        self._vsini = list(params)
        self._fast = fast

    def __call__(self, spec: Spectrum) -> np.ndarray:
        """Convolves a spectrum with the given Vsini kernel

        Args:
            spec (Spectrum): Spectrum to convolve

        Returns
            np.ndarray: Convolved spectrum
        """
        # Problem is that Spexxy requires log-sampled templates when applying broadening kernel, while PyAstronomy
        # currently only works with spectra regularly sampled in linear space.

        # spectrum in log mode?
        if spec.wave_mode != Spectrum.Mode.LOGLAMBDA:
            raise ValueError("Spectrum must be on log wavelength.")

        # create copy of original spectrum for following operations
        tmp = spec.copy()

        # apply velocity shift
        tmp.redshift(vrad=self._vsini[0])

        # create regular sampling in linear space
        tmp.mode(Spectrum.Mode.LAMBDA)
        spec_lin = tmp.resample_const()

        # apply broadening
        if self._fast:
            func = pyasl.fastRotBroad
        else:
            func = pyasl.rotBroad
        spec_lin.flux = func(spec_lin.wave, spec_lin.flux, epsilon=self._vsini[2], vsini=self._vsini[1])

        # recover original sampling & return
        spec_lin.mode(spec.wave_mode)
        final = spec_lin.resample(spec=spec)

        # make sure return value has no NaNs
        return np.nan_to_num(final.flux)


__all__ = ['Vsini']


# if __name__ == "__main__":
#     import time
#     import matplotlib.pyplot as plt
#     from spexxy.data import FitsSpectrum
#     from spexxy.data import LOSVD
#
#     # fs = FitsSpectrum('/Users/ariskama/Downloads/lte11200-4.50-0.5.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')
#     fs =  FitsSpectrum('/Users/ariskama/Downloads/lsfspec_teff8000_logg4.4_feh-0.35.fits')
#     t0 = time.time()
#
#     kernel_fast = Vsini(params=[100., 200, 0.5], fast=True)
#     fast = kernel_fast(fs.spectrum)
#     t1 = time.time()
#
#     kernel_slow = Vsini(params=[100., 200, 0.5], fast=False)
#     slow = kernel_slow(fs.spectrum)
#     t2 = time.time()
#
#     losvd = LOSVD(params=[100., 100, 0., 0., 0.])
#     alt = losvd(fs.spectrum)
#     t3 = time.time()
#     print(t1-t0, t2-t1, t3-t2)
#
#     fig, ax = plt.subplots()
#     ax.plot(fs.spectrum.wave, fs.spectrum.flux, 'g-')
#     ax.plot(fs.spectrum.wave, fast, 'b-')
#     ax.plot(fs.spectrum.wave, slow, 'g:')
#     ax.plot(fs.spectrum.wave, alt, 'r--')
#     plt.show()
