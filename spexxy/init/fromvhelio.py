import os
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
import astropy.units as u

from .init import Init
from ..component import Component


class InitFromVhelio(Init):
    """Initializes a component from the heliocentric correction calculated for RA/Dec given in the FITS headers.

    This class, when called, initializes a single parameter (the radial velocity, default to "v") of the given
    component with the heliocentric correction calculated from RA/DEC/DATE-OBS in the given file.
    """

    def __init__(self, negative: bool = False, parameter: str = 'v', scale: str = 'utc', obs: str = 'paranal',
                 kind: str = 'barycentric', *args, **kwargs):
        """Initializes a new Init object.

        Args:
            negative: If True, uses the negative of the calculated correction.
            parameter: Name of the parameter in the component to set.
            scale: Time scale to use for DATE-OBS.
            obs: Observatory to use for calculating correction.
            kind: Either barycentric or heliocentric.
        """
        Init.__init__(self, *args, **kwargs)
        self._negative = negative
        self._parameter = parameter
        self._scale = scale
        self._obs = EarthLocation.of_site(obs)
        self._kind = kind

    def __call__(self, cmp: Component, filename: str):
        """Initializes the radial velocity parameter of a given component to the (negative) heliocentric correction.

        Args:
            cmp: Component to initialize.
            filename: Name of file containing RA/DEC/DATE-OBS in FITS header to calculate correction from.
        """
        # file doesn't exist?
        if not os.path.exists(filename):
            self.log.error('Give file %s for extracting coordinates does not exist, skipping.', filename)
            return

        # get fits header
        header = fits.getheader(filename)

        # got coordinates in header?
        if 'RA' in header and 'DEC' in header and 'DATE-OBS' in header:
            # construct SkyCoord and time
            sc = SkyCoord(ra=header['RA'] * u.deg, dec=header['DEC'] * u.deg)
            time = Time(header['DATE-OBS'], scale=self._scale)

            # calculate correction
            vhelio = sc.radial_velocity_correction(self._kind, obstime=time, location=self._obs).to(u.km/u.s).value
            if self._negative:
                vhelio *= -1

            # set it
            self.log.info('Initializing "%s" of component "%s" to %sv_helio=%f...',
                          self._parameter, cmp.prefix, '-' if self._negative else '', vhelio)
            cmp[self._parameter] = vhelio


__all__ = ['InitFromVhelio']
