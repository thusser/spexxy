import os
import logging

from astropy.io import fits
import numpy as np
import pandas as pd
from typing import List, Tuple

from .weight import Weight
from ..data import Spectrum


class WeightFromGridNearest(Weight):
    """
    This class loads the weights from a grid depending on the initial values of the fit parameters by choosing the
     nearest neighbour in the grid. It returns an array containing the weights.
    """

    def __init__(self, filename, initial: float = 0., max_line_depth: float = 0.1, center_weight: float = 1.,
                 max_step: int = 1, *args, **kwargs):
        """
        Initializes a new weight.

        Args:
            filename: Name of grid file.
            initial: Initial value for the whole weight array.
            max_line_depth: Central pixel for lines with larger line depth are masked out.
            center_weight: Factor that increases the weight of the central pixel of each line.
            max_step: In iteration steps <= max_step new weights are loaded from the grid.
        """

        Weight.__init__(self, *args, **kwargs)

        self._initial = initial
        self._max_line_depth = max_line_depth
        self._center_weight = center_weight
        self._max_step = max_step

        # expand filename
        filename = os.path.expandvars(filename)

        # get grid's root path
        self._root = os.path.dirname(filename)

        # load CSV
        self._data = pd.read_csv(filename, index_col=False)

        # get all parameters, by removing 'Filename' from list of columns
        self._parameters = list(self._data.columns)
        self._parameters.remove('Filename')

        # we assume that all parameters are floats, so treat them as such
        for name in self._parameters:
            self._data[name] = self._data[name].apply(lambda x: float(x))

        # values set by main routine
        self.step = None
        self.init_values = {}

        self._filename = None

        # weights will be stored for next iterations
        self._weights = None

        self.new_weights = False
        self.return_dict = False

    def __call__(self, spectrum: Spectrum, filename: str):
        """
        Creates and returns weight array.

        Args:
            spectrum: Spectrum to create weight for.
            filename: Name of spectrum file.

        Returns:
             Array containing the weight for given spectrum.
        """

        # load new weights if max_step has not been reached
        if (self.step <= self._max_step) or self.new_weights:
            grid = self._data.copy()
            # loop over fit parameters
            for param_name, value in self.init_values.items():
                # remove component prefix from parameter name
                param_name = param_name.split()[-1]
                # parameter in given grid?
                if param_name not in self._parameters:
                    continue

                # get closest value on grid axis
                axis = np.array(sorted(self._data[param_name].unique()))
                grid = grid[grid[param_name] == axis[np.argmin(np.abs(axis - value))]]

            self._filename = grid.Filename.values[0]
            # save weights for all iteration steps
            self._weights = {step: self._load_weights(self._filename, spectrum, step) for step in range(1, 4)}

        # print(self.weights)

        if self.return_dict:
            return self._weights

        # return weight array for proper iteration step
        if self.step <= 3:
            return self._weights[self.step]

        return self._weights[3]

    def _load_weights(self, filename: str, spectrum: Spectrum, step: int):
        """
        Load CSV file from grid and create weight array.

        Args:
            filename: Filename of CSV file that contains the weights.
            spectrum: Spectrum to create the weight for.
            step: Iteration step.

        Returns:
             Weight array.
        """

        # load table containing the weights
        df = pd.read_csv(os.path.join(self._root, filename))

        # consider only weights for iteration steps lower/equal than the given one
        df = df[df['step'] <= step]

        # initialize weight array
        weights = np.zeros(spectrum.wave.shape) + self._initial

        # write weights to array
        for i, row in df.iterrows():
            weights[(spectrum.wave >= row['wave_start']) & (spectrum.wave <= row['wave_end'])] = row['weights']

            # if line depth larger than given threshold mask out the central region otherwise increase weight of
            # central pixel by given factor
            if row['line_depth'] > self._max_line_depth:
                # if region spans more than 10 wavelength pixel mask out the 3 central pixel otherwise only the central
                # one
                if (row['wave_end'] - row['wave_start']) // spectrum.wave_step >= 10:
                    i = np.argmin(np.abs(spectrum.wave - row['wave_center']))
                    weights[i - 1:i + 2] = 0
                else:
                    weights[np.argmin(np.abs(spectrum.wave - row['wave_center']))] = 0
            else:
                weights[np.argmin(np.abs(spectrum.wave - row['wave_center']))] *= self._center_weight

        return weights


__all__ = ['WeightFromGridNearest']
