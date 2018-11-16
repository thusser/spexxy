from astropy.io import fits
import numpy as np
import os
from typing import List, Tuple

from . import Interpolator
from ..grid import GridAxis, Grid
from ..data import Spectrum


class TelluricsInterpolator(Interpolator):
    """A tellurics interpolator"""

    def __init__(self, path: str, molecules: list, *args, **kwargs):
        """Initializes a new tellurics interpolator.

        Args:
            path: Path to files containing telluric grids
            molecules: Name of telluric grid files without .fits extension.

        Examples:
            Initialize the interpolator in the given path ($SPEXXYPATH/Tellurics) with H2O and O2 as the fitted
            molecules, provided by two different files (TelluricsH2O_R10000.fits, TelluricsO2_R10000.fits) in
            that directory:
                class: spexxy.interpolator.TelluricsInterpolator
                path: $SPEXXYPATH/Tellurics
                molecules: [TelluricsH2O_R10000, TelluricsO2_R10000]
        """

        Interpolator.__init__(self, *args, **kwargs)

        # load grids
        self._grids = []
        self._axes = []

        # load grids/axes
        self.log.info('Initializing tellurics interpolator...')
        for i, molec in enumerate(molecules):
            # get filename
            filename = os.path.join(os.path.expandvars(path), molec + '.fits')

            # load coefficients
            self.log.info("Loading tellurics grid from " + filename)
            self._grids += [fits.getdata(filename)]

            # get some header infos
            hdr = fits.getheader(filename)

            # add axis
            self._axes.append(
                GridAxis(name=hdr['MOLECULE'], min=hdr['VALIDMIN'], max=hdr['VALIDMAX'], initial=0.)
            )

            # get info from first grid
            if i == 0:
                self._wStart = hdr["CRVAL1"]
                self._wStep = hdr["CDELT1"]
                self._waveMode = Spectrum.Mode.LAMBDA
                if hdr["CTYPE1"] == "AWAV-LOG":
                    self._waveMode = Spectrum.Mode.LOGLAMBDA

    def axes(self) -> List[GridAxis]:
        """Returns information about the axes.

        Returns:
            List of GridAxis objects describing the grid's axes.
        """
        return self._axes

    def __call__(self, params: Tuple) -> Spectrum:
        """Interpolates at the given parameter set.

        Args:
            params: Parameter set to interpolate at.

        Returns:
            Interpolated spectrum at given position.
        """

        # loop params
        flux = None
        for i in range(len(params)):
            f = self._interpol(self._grids[i], params[i])
            if flux is None:
                flux = f
            else:
                flux *= f

        # return spec
        return Spectrum(flux=flux, wave_start=self._wStart, wave_step=self._wStep, wave_mode=self._waveMode)

    def _interpol(self, grid: Grid, value: Tuple) -> np.ndarray:
        """Do the actual interpolation in the given grid at the given position.

        Args:
            grid: Grid to interpolate in.
            params: Parameter to interpolate at.

        Returns:
            Interpolated tellurics at given position.
        """

        # matrix
        mat = np.empty((grid.shape[0]))
        for i in range(grid.shape[0]):
            if i == 0:
                mat[0] = 1.
            else:
                mat[i] = mat[i - 1] * value

        # evaluate
        return np.dot(mat, grid)


__all__ = ['TelluricsInterpolator']
