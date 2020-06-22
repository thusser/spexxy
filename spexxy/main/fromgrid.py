from typing import List, Tuple, Dict
import numpy as np

from .base import MainRoutine
from ..data import SpectrumFits, Spectrum, LSF
from ..grid import Grid


class FromGrid(MainRoutine):
    """Grid is a main routine for fetching a spectrum from a grid."""

    def __init__(self, grid: Grid, params: Dict[str, Tuple], prefix: str = 'spectrum_', lsf: str = None,
                 *args, **kwargs):
        """Initialize a new ParamsFit object

        Args:
            grid: Grid to extract from.
            param: Dict with filename->parameters pairs with spectra to extract.
            prefix: Prefix for spectra to write.
            lsf: Name of file containing LSF.
        """
        MainRoutine.__init__(self, *args, **kwargs)
        self._grid = grid
        self._params = params
        self._prefix = prefix
        self._lsf = lsf

    def __call__(self):
        """Start the routine."""

        # get grid
        grid = self.get_objects(self._grid, Grid, 'grids', self.log, single=True)

        # loop params
        for i, (filename, params) in enumerate(self._params.items(), 1):
            # get spectrum
            self.log.info('Fetching spectrum at %s...', params)
            spec: Spectrum = grid(tuple(params))

            # lsf?
            if self._lsf is not None:
                # resample const
                spec = spec.resample_const()

                # load lsf
                lsf = LSF.load(self._lsf).extract_at_wave(np.mean(spec.wave))
                lsf.resample(spec)

                # apply it
                spec = lsf(spec)

            # write it
            self.log.info('Saving as %s...', filename)
            SpectrumFits(spec=spec).save(filename)


__all__ = ['FromGrid']
