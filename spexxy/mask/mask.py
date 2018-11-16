import numpy as np

from ..object import spexxyObject
from ..data import Spectrum


class Mask(spexxyObject):
    """Mask is the base class for all objects that can create a good pixel mask for spectra."""

    def __init__(self, *args, **kwargs):
        """Initialize a new mask."""
        spexxyObject.__init__(self, *args, **kwargs)

    def __call__(self, spectrum: Spectrum, filename: str) -> np.ndarray:
        """Creates a new mask for a spectrum.

        Args:
            spectrum: Spectrum to create mask for.
            filename: Name of file containing spectrum to create mask for.

        Returns:
            Boolean array containing good pixel mask for given spectrum.

        """
        raise NotImplementedError


__all__ = ['Mask']
