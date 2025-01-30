import numpy as np
from typing import Tuple
from PyAstronomy import modelSuite as ms

from spexxy.data.spectrum import Spectrum


class Vsini:

    def __init__(self, params: Tuple):

        self._vsini = list(params)

    def __call__(self, spec: Spectrum) -> np.ndarray:
        """Convolves a spectrum with the given Vsini kernel

        Args:
            spec (Spectrum): Spectrum to convolve

        Returns
            np.ndarray: Convolved spectrum
        """
        # spectrum in log mode?
        if spec.wave_mode != Spectrum.Mode.LOGLAMBDA:
            raise ValueError("Spectrum must be on log wavelength.")

        # spectrum regularly sampled?
        if spec.wave_step is None:
            raise NotImplementedError

        # get sampling in velocity space
        delta_v = 299792.458 * spec.wave_step

        # The line-of sight velocity is split up into the nearest smaller value that corresponds to a shift
        # by an interger number of pixels, and the remaining fractional pixel shift. The latter is applied
        # when convolving with the Vsini kernel, while the former is applied by shifting the convolved
        # spectrum by an integer number of pixels
        int_shift = int(np.floor(self._vsini[0]/delta_v))  # no. of pixels by which convolved array needs to be shifted
        frac_shift = np.mod(self._vsini[0], delta_v)  # velocity shift relative to previous pixel (>0)

        # set up broadening kernel
        n_pix_kernel = 2*(self._vsini[1]//delta_v) + 1  # no. of pixels in kernel always odd
        profile = ms.RotBroadProfile()
        profile["xmax"] = self._vsini[1]
        profile["A"] = delta_v  # so that profile is normalised to one
        profile["eps"] = self._vsini[2]
        profile["off"] = 0
        profile["mu"] = frac_shift

        # set up wavelength array of kernel, LOS velocity will be between pixels [n_pix_kernel//2, n_pix_kernel//2+1)
        vel = delta_v*(np.arange(n_pix_kernel, dtype=np.float64) - n_pix_kernel//2)

        # evaluate broadening kernel & apply to spectrum
        kernel = profile.evaluate(vel)
        out = np.convolve(spec.flux, kernel, mode="same")

        # return final spectrum
        if int_shift == 0:
            # make sure return value has no NaNs
            return np.nan_to_num(out)
        elif int_shift > 0:
            return np.nan_to_num(np.pad(out, (int_shift, 0), mode="edge"))[:-int_shift]
        else:
            return np.nan_to_num(np.pad(out, (0, -int_shift), mode="edge"))[-int_shift:]


__all__ = ['Vsini']


# if __name__ == "__main__":
#     import time
#     import matplotlib.pyplot as plt
#     from spexxy.data import FitsSpectrum
#     from spexxy.data import LOSVD
#
#     fs = FitsSpectrum('/Users/ariskama/Downloads/lte11200-4.50-0.5.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')
#     # fs =  FitsSpectrum('/Users/ariskama/Downloads/lsfspec_teff8000_logg4.4_feh-0.35.fits')
#     t0 = time.time()
#
#     kernel = Vsini(params=[144., 200, 0.5])
#     new = kernel(fs.spectrum)
#     t1 = time.time()
#
#     losvd = LOSVD(params=[144., 100, 0., 0., 0.])
#     alt = losvd(fs.spectrum)
#     t2 = time.time()
#     print(t1-t0, t2-t1)
#
#     fig, ax = plt.subplots()
#     ax.plot(fs.spectrum.wave, fs.spectrum.flux, 'g-')
#     ax.plot(fs.spectrum.wave, new, 'm-.')
#     ax.plot(fs.spectrum.wave, alt, 'r--')
#     plt.show()
