import numpy as np

from .mask import Mask
from ..data import Spectrum


class MaskNegative(Mask):
    """Masks negative fluxes in a spectrum.

    This class, when called, creates a mask that masks all negative pixels in the given spectrum.
    """

    def __init__(self, *args, **kwargs):
        """Initializes a new mask for masking negative pixels in a spectrum."""
        Mask.__init__(self, *args, **kwargs)

    def __call__(self, spectrum: Spectrum, filename: str = None) -> np.ndarray:
        """Creates a new mask for the given spectrum masking all negative pixels.

        Args:
            spectrum: Spectrum to create mask for.
            filename: Name of file containing spectrum to create mask for (unused).

        Returns;
            Boolean array containing good pixel mask for given spectrum.
        """

        # mask all negative pixels, i.e. create a mask with only positive fluxes, and return it
        self.log.info('Masking all negative pixels in spectrum...')
        return spectrum.flux > 0


__all__ = ['MaskNegative']
