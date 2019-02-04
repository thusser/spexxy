import os
import logging
from astropy.io import fits
import numpy as np

from .weight import Weight
from ..data import Spectrum


class WeightFromSNR(Weight):
    """Reads the S/N ratio from the given file and creates weights from it as 1/SQRT(FLUX/SNR).

    This class, when called, loads the SNR from the given file and returns weights from it as 1/SQRT(FLUX/SNR).
    """

    def __init__(self, keyword: str = 'HIERARCH SPECTRUM SNRATIO', *args, **kwargs):
        """Initializes a new weight.

        Args:
            keyword: FITS header keyword containing S/N.
        """
        Weight.__init__(self, *args, **kwargs)
        self._keyword = keyword

    def __call__(self, spectrum: Spectrum, filename: str) -> np.ndarray:
        """Creates a new weight for a spectrum from its S/N.

        Args:
            spectrum: Spectrum to create weight for.
            filename: Name of file containing spectrum to create weight for.

        Returns:
            Array containing weight for given spectrum.
        """

        self.log.info('Creating weights from S/N...')

        # get S/N
        snr = fits.getval(filename, self._keyword)

        # calculate weights
        return 1. / np.sqrt(spectrum.flux / snr)


__all__ = ['WeightFromSNR']
