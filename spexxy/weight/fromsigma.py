import os
import logging
from astropy.io import fits
import numpy as np

from .weight import Weight
from ..data import Spectrum


class WeightFromSigma(Weight):
    """Reads the SIGMA extension from the given file and creates weights from it as 1/SIGMA.

    This class, when called, loads the SIGMA extension from the given file and returns the weights as 1/SIGMA.
    """

    def __init__(self, squared: bool = False, *args, **kwargs):
        """Initializes a new weight.

        Args:
            squared: Return 1/SIGMA**2 instead of 1/SIGMA.
        """
        Weight.__init__(self, *args, **kwargs)
        self._squared = squared

    def __call__(self, spectrum: Spectrum, filename: str) -> np.ndarray:
        """Creates a new weight for a spectrum from its SIGMA extension.

        Args:
            spectrum: Spectrum to create weight for.
            filename: Name of file containing spectrum to create weight for.

        Returns:
            Array containing weight for given spectrum.
        """

        self.log.info('Creating weights from SIGMA extension...')

        # load SIGMA
        sigma = fits.getdata(filename, 'SIGMA')

        # calculate weights
        weights = 1. / sigma

        # square it?
        if self._squared:
            weights = weights * weights

        # finished
        return weights


__all__ = ['WeightFromSigma']
