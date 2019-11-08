import os
import logging

from astropy.io import fits
import numpy as np
import pandas as pd
from typing import List, Tuple

from .weight import Weight
from .fromgridnearest import WeightFromGridNearest
from ..data import Spectrum


class WeightFromGrid(Weight):
    """
    This class loads the weights from a grid depending on the initial values of the fit parameters by linear
    interpolation. It returns an array containing the weights.
    """

    def __init__(self, filename, initial: float = 0., max_line_depth: float = 0.1, center_weight: float = 1.,
                 max_step: int = 1, *args, **kwargs):
        """
        Initializes a new weight.

        Args:
            filename: Name of grid file.
            initial: Initial value for the whole weight array.
            max_line_depth:
            max_line_depth: Central pixel for lines with larger line depth are masked out.
            center_weight: Factor that increases the weight of the central pixel of each line.
            max_step: In iteration steps <= max_step new weights are loaded from the grid.
        """

        Weight.__init__(self, *args, **kwargs)

        self._initial = initial
        self._max_line_depth = max_line_depth
        self._center_weight = center_weight
        self._max_step = max_step

        self._args = args
        self._kwargs = kwargs

        # expand filename
        self._filename = os.path.expandvars(filename)

        # get grid's root path
        self._root = os.path.dirname(self._filename)

        # load CSV
        self._data = pd.read_csv(self._filename, index_col=False)

        # get all parameters, by removing 'Filename' from list of columns
        self._parameters = list(self._data.columns)
        self._parameters.remove('Filename')

        # we assume that all parameters are floats, so treat them as such
        for name in self._parameters:
            self._data[name] = self._data[name].apply(lambda x: float(x))

        # get grid axes
        self._axes = [sorted(self._data[p].unique()) for p in self._parameters]

        # remove axes that contain only a single value
        for i, p in enumerate(self._parameters):
            if len(self._axes[i]) <= 1:
                del self._axes[i]
                self._parameters.remove(p)

        # values set by main routine
        self.step = None
        self.init_values = {}

        # weights will be stored for next iterations
        self._weights = None

        # stores values of fit parameters from two iterations before
        self._previous_values = None

    def __call__(self, spectrum: Spectrum, filename: str) -> np.ndarray:
        """
        Creates and returns weight array.

        Args:
            spectrum: Spectrum to create weight for.
            filename: Name of spectrum file.

        Returns:
             Array containing the weight for given spectrum.
        """

        # load new weights if the fit parameters changed significantly
        new_weights = False
        if self._previous_values is not None:
            params = []
            for param in self._parameters:
                # fit parameter in component?
                if param in map(lambda x: x.split()[-1], list(self.init_values.keys())):
                    p = list(filter(lambda x: param == x.split()[-1], list(self.init_values.keys())))[0]
                    params.append(self.init_values[p])
                    i = self._parameters.index(param)
                    if param.lower() == 'teff':
                        if abs(self._previous_values[i] - self.init_values[p]) >= 250:
                            new_weights = True
                    else:
                        if abs(self._previous_values[i] - self.init_values[p]) >= 0.25:
                            new_weights = True

            self._previous_values = params.copy()

        # load new weights if max_step has not been reached or fit parameters changed significantly
        if (self.step <= self._max_step) or new_weights:
            params = []
            for param in self._parameters:
                # fit parameter in component
                if param in map(lambda x: x.split()[-1], list(self.init_values.keys())):
                    # write parameter value to list
                    p = list(filter(lambda x: param == x.split()[-1], list(self.init_values.keys())))[0]
                    params.append(self.init_values[p])

            self._previous_values = params.copy()

            # interpolate weight for given values, use nearest neighbour if values are outside of the grid
            try:
                w = self._interpolate(tuple(params))
            except KeyError:
                w = WeightFromGridNearest(self._filename, self._initial, self._max_line_depth, self._center_weight,
                                          self._max_step, *self._args, **self._kwargs)
                w.init_values = self.init_values
                w.step = self.step
                w.new_weights = new_weights
                w.return_dict = True
                self._weights = w(spectrum, filename)

                if self.step <= 3:
                    return self._weights[self.step]

                return self._weights[3]

            # consider only the regions with the highest weights
            if len(w[w['step'] == 1]) >= 20:
                w = w[w['step'] == 1]
                w = w.sort_values('weights', ascending=False).iloc[:20]

            # create weight array for each iteration step
            self._weights = {step: self._get_weight_array(
                w[w['step'] <= step], spectrum) for step in range(1, 4)}

        # return weight array for proper iteration step
        if self.step <= 3:
            return self._weights[self.step]

        return self._weights[3]

    def _interpolate(self, params: Tuple, axis: int = None) -> pd.DataFrame:
        # no axis given, start at latest
        if axis is None:
            axis = len(self._axes) - 1

        if params[axis] < min(self._axes[axis]) or params[axis] > max(self._axes[axis]):
            raise KeyError('Requested parameters are outside the grid.')

        # let's get all possible values for the given axis
        axisValues = self._axes[axis].copy()

        # if params[axis] is on axis; return it directly
        if params[axis] in axisValues:
            if axis == 0:
                return self._load_weight_table(params)
            else:
                return self._interpolate(tuple(params), axis - 1)

        # find the next lower and the next higher axis value
        p_lower = self._neighbour(tuple(params), axis, 0)
        p_higher = self._neighbour(tuple(params), axis, 1)

        if p_lower is None or p_higher is None:
            raise KeyError('No direct neighbours found in grid.')

        # get axis values
        x_lower = p_lower[axis]
        x_higher = p_higher[axis]

        # get data for p_lower and p_higher
        if axis == 0:
            lower_data = self._load_weight_table(p_lower)
            higher_data = self._load_weight_table(p_higher)
        else:
            lower_data = self._interpolate(p_lower)
            higher_data = self._interpolate(p_higher)

        # interpolate
        f = (params[axis] - x_lower) / (x_higher - x_lower)

        # add interpolation weight to table
        lower_data['w'] = 1. - f
        higher_data['w'] = f

        df = pd.concat([lower_data, higher_data])
        df = df.sort_values(by=['wave_center'])
        cols = list(df.columns)
        cols.remove('wave_center')

        # assign identical values to wave centers less than 0.3 Angstrom apart
        centers = df['wave_center'].values
        while True:
            delta = np.ediff1d(centers)
            mask = (np.abs(delta) <= 0.3) & (delta != 0.)
            centers[1:][mask] = centers[:-1][mask]

            if not np.any(mask):
                break

        # average lines with identical wave centers together
        df['wave_center'] = centers
        df_grouped = df.groupby('wave_center')
        df = df_grouped.filter(lambda x: len(x) == 1)

        # average lines showing up in both tables
        df2 = df_grouped.filter(lambda x: (len(x) == 2) & (x['w'].sum() == 1)).copy()
        w = df2['w'].values[:, None]
        df2.loc[:, cols] *= w
        df2 = df2.groupby('wave_center').sum()
        df2 = df2.reset_index()
        cols.insert(1, 'wave_center')
        df2 = df2.loc[:, cols]

        df = pd.concat([df, df2]).sort_values('wave_center')
        df = df.drop(columns=['w'])

        df['step'] = df['step'].apply(lambda x: np.around(x, decimals=0))

        return df

    def _load_weight_table(self, params: Tuple) -> pd.DataFrame:
        """
        Load CSV file containing the weights for the given set of parameters.
        """

        grid = self._data.copy()
        for i, p in enumerate(self._parameters):
            grid = grid[grid[p] == params[i]]

        filename = grid.Filename.values[0]

        return pd.read_csv(os.path.join(self._root, filename))

    def _get_weight_array(self, df: pd.DataFrame, spectrum: Spectrum):
        """
        Create weight array from the given table for the given spectrum.

        Args:
            df: Table containing the weights for each absorption line considered in that fitting mask.
            spectrum: Spectrum for which the weight array is created.

        Returns:
             Weight array.
        """

        # consider only weights for iteration steps lower/equal than the given one
        df = df[df['step'] <= self.step]

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
                    weights[i-1:i+2] = 0
                else:
                    weights[np.argmin(np.abs(spectrum.wave - row['wave_center']))] = 0
            else:
                weights[np.argmin(np.abs(spectrum.wave - row['wave_center']))] *= self._center_weight

        return weights

    def _neighbour(self, params: Tuple, axis: int, distance: int = 1):
        """Finds a neighbour on the given axis for the given value in the given distance.

        Args:
            params: Paremeter tuple to search neighbour from.
            axis: Axis to search for
            distance: Distance in which to find neighbour.
                >0:  Find larger neighbours, i.e. 0 next larger value, 1 the one after that, etc
                <=0:  Find smaller neighbouars, i.e. 0 next smaller value (or value itself), -1 the before that, etc

        Returns:
            New parameter tuple with neighbour on the given axis.
        """

        # find neighbour in axis
        values = self._axes[axis]
        value = None
        # loop all values
        for i in range(len(values)):
            # found value?
            if values[i] <= params[axis] < values[i + 1]:
                # index of neighbour
                ii = i + distance
                # does it exist?
                if 0 <= ii < len(values):
                    value = values[ii]

        if value is None:
            return None

        # create new tuple
        p = list(params)
        p[axis] = value

        return tuple(p)


__all__ = ['WeightFromGrid']
