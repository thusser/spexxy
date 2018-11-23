import numpy as np

from ..object import spexxyObject
from ..data import Spectrum


class Weight(spexxyObject):
    """Weight is the base class for all objects that can create a weight array for spectra."""

    def __init__(self, *args, **kwargs):
        """Initialize a new weight."""
        spexxyObject.__init__(self, *args, **kwargs)

    def __call__(self, spectrum: Spectrum, filename: str) -> np.ndarray:
        """Creates a new weight for a spectrum.

        Args:
            spectrum: Spectrum to create weight for.
            filename: Name of file containing spectrum to create weight for.

        Returns:
            Array containing weight for given spectrum.
        """
        raise NotImplementedError


__all__ = ['Weight']
