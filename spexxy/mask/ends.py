import numpy as np

from .mask import Mask
from ..data import Spectrum


class MaskEnds(Mask):
    """Masks the ends of a spectrum.

    This class, when called, creates a mask that masks the N pixels at each end of the given spectrum.
    """

    def __init__(self, npixels=10, *args, **kwargs):
        """Initializes a new mask masking the ends of a spectrum.

        Args:
            npixels: Number of pixels to mask at each end of the spectrum.
        """
        Mask.__init__(self, *args, **kwargs)
        self._npixels = npixels

        # npixels cannot be negative
        if npixels < 0:
            raise ValueError('Number of pixels to mask at each end of the spectrum cannot be negative.')

    def __call__(self, spectrum: Spectrum, filename: str = None) -> np.ndarray:
        """Creates a new mask for the given spectrum masking only the N pixels at each end.

        Args:
            spectrum: Spectrum to create mask for.
            filename: Name of file containing spectrum to create mask for (unused).

        Returns;
            Boolean array containing good pixel mask for given spectrum.
        """

        # create empty mask
        mask = np.ones((len(spectrum.wave)), dtype=bool)

        # if npixels is zero, just return mask
        if self._npixels == 0:
            return mask

        # mask ends and return mask
        mask[:self._npixels] = False
        mask[-self._npixels:] = False
        self.log.info('Masked %d pixels at both ends of the spectrum.', self._npixels)
        return mask


__all__ = ['MaskEnds']
