from typing import Union
from astropy.io import fits
import numpy as np

from .mask import Mask
from ..data import Spectrum


class MaskRanges(Mask):
    """Masks ranges in spectra.

    This class, when called, creates a mask on the given wavelength ranges
    """

    def __init__(self, ranges: list, vrad: Union[str, float] = None,
                 component: str = None, vrad_parameter: str = None,
                 *args, **kwargs):
        """Initializes a new mask.

        Args:
            ranges: List of tuples defining (start, end) of wavelength ranges to mask.
            vrad: Radial velocity to shift by, either a number or the name of a FITS header entry, in which
                case "-<name>" negates the value.
            component: Name of component to read the radial velocity from.
            vrad_parameter: Name of parameter in given component to use as radial velocity.
        """
        Mask.__init__(self, *args, **kwargs)
        self._ranges = ranges
        self._vrad = vrad
        self._component = component
        self._vrad_parameter = vrad_parameter

    @staticmethod
    def _redshift(wave: float, vrad: float, mode: Spectrum.Mode = Spectrum.Mode.LAMBDA) -> float:
        """Redshifts a given wavelengths by the given radial velocity.

        Args:
            wave: Wavelength to redshift.
            vrad: Radial velocity to shift by.
            mode: Reference frame of wavelength (LAMBDA or LOG).

        Returns:
            Redhifted wavelength.
        """

        # calculate redshift depending on mode
        if mode == Spectrum.Mode.LAMBDA:
            return wave * (1. + vrad / 299792.458)
        else:
            return wave + np.log(1. + vrad / 299792.458)

    def __call__(self, spectrum: Spectrum, filename: str = None) -> np.ndarray:
        """Creates a new dynamic mask for a spectrum in the wavelength ranges given in the configuration and
        shifted by the current radial velocity of the given component.

        Args:
            spectrum: Spectrum to create mask for.
            filename: Name of file containing spectrum to create mask for.

        Returns:
            Boolean array containing good pixel mask for given spectrum.
        """

        # by default, there is no vrad shift
        vrad = 0.

        # get vrad from component
        if self._component is not None and self._vrad_parameter is not None:
            vrad = self.objects['components'][self._component][self._vrad_parameter]

        # additional vrad?
        if self._vrad is not None:
            # str or float?
            if isinstance(self._vrad, float):
                # absolute value
                self.log.info('Shifting all dynamic masks by constant %.2f km/s.', self._vrad)
                vrad += self._vrad
            elif isinstance(self._vrad, str):
                # FITS header
                mask_sign = 1.
                mask_shift = self._vrad
                # negative?
                if self._vrad[0] == '-':
                    mask_sign = -1
                    mask_shift = self._vrad[1:]
                # get fits header
                hdr = fits.getheader(filename)
                try:
                    # get shift and add it
                    val = mask_sign * hdr[mask_shift]
                    self.log.info('Shifting all masks by %.3f km/s.', val)
                    vrad += val
                except KeyError:
                    self.log.error('Could not find FITS header "%s" in file.')

        # init mask
        mask = np.ones((len(spectrum.wave)), dtype=np.bool)

        # loop ranges
        for rng in self._ranges:
            # need to swap?
            if rng[0] > rng[1]:
                rng = (rng[1], rng[0])

            # shift start/end
            rng = (MaskRanges._redshift(rng[0], vrad), MaskRanges._redshift(rng[1], vrad))

            # create (inverted) mask
            m = (spectrum.wave >= rng[0]) & (spectrum.wave <= rng[1])

            # apply it
            mask &= ~m

            # log it
            self.log.info('Masked range %.2f-%.2f AA (%d pixels).', rng[0], rng[1], np.sum(m))

        # finished
        return mask


__all__ = ['MaskRanges']
