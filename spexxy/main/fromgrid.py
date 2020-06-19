from typing import List, Tuple
from .base import MainRoutine
from ..data import SpectrumFits, Spectrum
from ..grid import Grid


class FromGrid(MainRoutine):
    """Grid is a main routine for fetching a spectrum from a grid."""

    def __init__(self, grid: Grid, params: List[Tuple], prefix: str = 'spectrum_', *args, **kwargs):
        """Initialize a new ParamsFit object

        Args:
            grid: Grid to extract from.
            param: List of parameter tuples with spectra to extract.
            prefix: Prefix for spectra to write.
        """
        MainRoutine.__init__(self, *args, **kwargs)
        self._grid = grid
        self._params = params
        self._prefix = prefix

    def __call__(self):
        """Start the routine."""

        # get grid
        grid = self.get_objects(self._grid, Grid, 'grids', self.log, single=True)

        # loop params
        for i, params in enumerate(self._params, 1):
            # get spectrum
            self.log.info('Fetching spectrum at %s...', params)
            spec: Spectrum = grid(tuple(params))

            # get filename
            filename = '%s%04d.fits' % (self._prefix, i)
            self.log.info('Saving as %s...', filename)

            # write it
            SpectrumFits(spec=spec).save(filename)


__all__ = ['FromGrid']
