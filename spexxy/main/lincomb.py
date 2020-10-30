from typing import List
import numpy as np
import pandas as pd
import scipy.special
import scipy.linalg
import scipy.optimize
from spexxy.data import Spectrum, FitsSpectrum
import matplotlib.pyplot as plt

from .base import FilesRoutine
from ..grid import FitsGrid


class LinearCombinationFit(FilesRoutine):
    """A routine that tries to fit a given spectrum by finding a linear combination of spectra in a grid."""

    def __init__(self, grid: FitsGrid, *args, **kwargs):
        """Initialize a new LinearCombinationFit object

        Args:
            grid: Grid to use for fitting.
        """
        FilesRoutine.__init__(self, *args, **kwargs)

        # create grid
        self._grid: FitsGrid = self.get_objects(grid, FitsGrid, 'grids', single=True)
        self._poly_degree = 5

    def parameters(self) -> List[str]:
        """Get list of parameters fitted by this routine.

        Returns:
            List of parameter names (including prefix) fitted by this routine.
        """

        # just a list of grid axes
        return ['STAR %s' % n for n in self._grid.axis_names()]

    def __call__(self, filename: str) -> List[float]:
        """Start the fitting procedure on the given file.

        Args:
            filename: Name of file to fit.

        Returns:
            List of final values of parameters, ordered in the same way as the return value of parameters()
        """

        # get data from grid
        grid = self._grid.data

        # create Legendre polynomials
        legendre = np.empty((self._poly_degree, grid.shape[1]))
        x = np.linspace(-1, 1, grid.shape[1])
        for n in range(self._poly_degree):
            leg = scipy.special.legendre(n)
            legendre[n, :] = leg(x)

        # concatenate arrays
        data = np.concatenate((grid, legendre), axis=0)

        # load spectrum
        spec = Spectrum.load(filename)
        mask = ~np.isnan(spec.flux)

        # do fit
        x, _ = scipy.optimize.nnls(data[:, mask].T, spec.flux[mask])

        # get bestfit
        best_fit = Spectrum(spec=spec)
        best_fit.flux = data.T @ x

        # get table with parameters and weights
        params = dict(zip(self._grid.axis_names(), list(map(list, zip(*self._grid.all())))))
        params['weight'] = x[:-self._poly_degree] / np.sum(x)
        weights_table = pd.DataFrame(params)

        # write results
        self._write_results_to_file(filename, spec, weights_table, best_fit)

        # do weighting in table
        results = []
        for col in self._grid.axis_names():
            # calculate weighted mean and append it and error
            mean = np.sum(weights_table[col] * weights_table['weight'])
            results.extend([mean, 0.])

        # finished
        return results

    def _write_results_to_file(self, filename: str, spec: Spectrum, weights: pd.DataFrame, best_fit: Spectrum):
        """Writes results of fit back to file.

        Args:
            filename: Name of file to write results into.
            weights: Weights array.
            best_fit: Best fit model.
        """

        # Write fits results back to file
        self.log.info("Writing results to file.")
        with FitsSpectrum(filename, 'rw') as fs:
            # write spectra best fit and residuals
            if best_fit is not None:
                fs.best_fit = best_fit
                fs.residuals = spec.flux - best_fit.flux

        # write weights (as csv for now)
        weights.to_csv(filename.replace('.fits', '.csv'), index=False)


__all__ = ['LinearCombinationFit']
