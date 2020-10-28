from typing import List, Any, Tuple
import pandas as pd
import os
import numpy as np
from astropy.io import fits
from astropy.table import Table

from .grid import Grid, GridAxis
from ..data import Spectrum


class FitsGrid(Grid):
    """Spectrum grid in a single FITS file."""

    def __init__(self, filename: str, norm_to_mean: bool = False, *args, **kwargs):
        """Constructs a new Grid.

        Args:
            filename: Filename of CSV file.
            norm_to_mean: Normalize spectra to their mean.
        """

        # store
        self._norm_to_mean = norm_to_mean

        # expand filename
        filename = os.path.expandvars(filename)

        # load data
        self._spectra, hdr = fits.getdata(filename, header=True)

        # get spec info
        self._wave_start = hdr['CRVAL1']
        self._wave_step = hdr['CDELT1']
        self._wave_mode = Spectrum.Mode.LAMBDA if hdr['CTYPE1'] == 'AWAV' else Spectrum.Mode.LOGLAMBDA

        # load parameters into a pandas table
        self._data = Table(fits.getdata(filename, 'PARAMS')).to_pandas()
        idx_columns = list(self._data.columns.values)

        # add row
        self._data['row'] = range(len(self._data))

        # create axes and init grid
        values = {name: sorted([float(v) for v in self._data[name].unique()]) for name in idx_columns}
        axes = [GridAxis(name=name, values=values[name]) for name in idx_columns]
        Grid.__init__(self, axes, *args, **kwargs)

        # set index
        self._data.set_index(idx_columns, inplace=True)

    def all(self) -> List[Tuple]:
        """Return all possible parameter combinations.

        Returns:
            All possible parameter combinations.
        """
        return self._data.index.values

    def __contains__(self, params: Tuple) -> bool:
        """Checks, whether the grid contains a given parameter set.

        Args:
            params: Parameter set to check.

        Returns:
            Whether or not the given parameter set exists in the grid.
        """
        return tuple(params) in self._data

    def __call__(self, params: Tuple) -> Any:
        """Fetches the value for the given parameter set.

        Args:
            params: Parameter set to catch value for.

        Returns:
            Grid value at given position.
        """

        # get index of parameters
        idx = self._data.loc[tuple(params)].row

        # get flux
        flux = self._spectra[idx, :]

        # create spectrum
        spec = Spectrum(flux=flux, wave_start=self._wave_start, wave_step=self._wave_step, wave_mode=self._wave_mode)

        # normalize?
        if self._norm_to_mean:
            spec.norm_to_mean()

        # return it
        return spec


__all__ = ['FitsGrid']
