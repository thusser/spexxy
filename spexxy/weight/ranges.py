import os
import logging
from astropy.io import fits
import numpy as np
from typing import List, Tuple

from .weight import Weight
from ..data import Spectrum


class WeightRanges(Weight):
    """Creates a weights array from given ranges.

    This class, when called, creates a weights array from the given wavelength ranges.
    """

    def __init__(self, ranges: List[Tuple[float, float, float]], initial: float = 1., *args, **kwargs):
        """Initializes a new weight.

        Args:
            ranges: List of tuples of (wave start, wave end, weight).
            initial: Initial value for whole array.
        """
        Weight.__init__(self, *args, **kwargs)
        self._ranges = ranges
        self._initial = initial

    def __call__(self, spectrum: Spectrum, filename: str) -> np.ndarray:
        """Creates a new weight for a spectrum from the ranges given in the configuration.

        Args:
            spectrum: Spectrum to create weight for.
            filename: Name of file containing spectrum to create weight for.

        Returns:
            Array containing weight for given spectrum.
        """

        # init array
        self.log.info('Creating weights for spectrum from given ranges...')
        weights = np.zeros((len(spectrum))) + self._initial

        # loop ranges
        for start, end, weight in self._ranges:
            # get pixels in range
            w = (spectrum.wave >= start) & (spectrum.wave <= end)
            # set weight
            weights[w] = weight

        # finished
        return weights


__all__ = ['WeightRanges']
