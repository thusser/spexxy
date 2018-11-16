import os
import logging
from astropy.io import fits
import numpy as np

from .mask import Mask
from ..data import Spectrum


class MaskFromPath(Mask):
    """Reads a pre-calculated mask from another file of a given name.

    This class, when called, searches for a file of the given name in the directory specified in the configuration.
    If it exists, the extension of the given name (defaults to "GOODPIXELS") is read, converted into a Boolean array,
    and returned. If it doesn't exist, an empty mask is returned.
    """

    def __init__(self, path: str, fits_extension: str = 'GOODPIXELS', *args, **kwargs):
        """Initializes a new mask from a file in a given path.

        Args:
            path: Path to search in for file containing mask.
            fits_extension: FITS extension to read mask from.
        """
        Mask.__init__(self, *args, **kwargs)
        self._path = path
        self._fits_extension = fits_extension

    def __call__(self, spectrum: Spectrum, filename: str) -> np.ndarray:
        """Creates a new mask for a spectrum from a file of the same name in a given directory.

        Args:
            spectrum: Spectrum to create mask for.
            filename: Name of file containing spectrum to create mask for.

        Returns:
            Boolean array containing good pixel mask for given spectrum.
        """

        # build filename
        filename = os.path.join(self._path, os.path.basename(filename))

        # does it exist?
        if os.path.exists(filename):
            # read from file and return it as boolean array
            self.log.info('Reading additional mask from file %s...', filename)
            try:
                mask = fits.getdata(filename, self._fits_extension).astype(np.bool)
            except KeyError:
                self.log.warning('Could not find extension %s in FITS file, skipping.', self._fits_extension)
                return np.ones((len(spectrum.wave)), dtype=np.bool)

            # check length
            if len(spectrum) != len(mask):
                self.log.warning('Length of mask does not match length of spectum. Using empty mask.')
                return np.ones((len(spectrum.wave)), dtype=np.bool)

            # return mask
            return mask

        else:
            # return empty mask...
            self.log.warning('Could not find addition file for mask: %s', filename)
            return np.ones((len(spectrum.wave)), dtype=np.bool)


__all__ = ['MaskFromPath']
