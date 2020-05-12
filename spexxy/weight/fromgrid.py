import os

import numpy as np
import pandas as pd
from typing import List, Union, Tuple

from .weight import Weight
from .fromgridnearest import WeightFromGridNearest
from ..data import Spectrum


class WeightFromGrid(Weight):
    """
    This class loads the weights from a grid depending on the initial values of the fit parameters by linear
    interpolation. It returns an array containing the weights.
    """

    def __init__(self, filename, initial: float = 0., max_line_depth: float = 0.5, center_weight: float = 1.,
                 max_step: int = 1, mask_lines: Union[bool, str, List] = True, max_change=(300, 0.3), *args, **kwargs):
        """
        Initializes a new weight.

        Args:
            filename: Name of grid file.
            initial: Initial value for the whole weight array.
            max_line_depth: Central pixel for lines with larger line depth are masked out.
            center_weight: Factor that increases the weight of the central pixel of each line.
            max_step: In iteration steps <= max_step new weights are loaded from the grid.
            mask_lines: List of absorption lines that are always masked out in their centers.
        """

        Weight.__init__(self, *args, **kwargs)

        # expand filename
        filename = os.path.expandvars(filename)

        self.filename = filename

        self._initial = initial
        self._max_line_depth = max_line_depth
        self._center_weight = center_weight
        self._max_step = max_step
        self._max_change = sorted(max_change, reverse=True)

        if mask_lines:
            if isinstance(mask_lines, bool):
                self._mask_lines = 'default'
            elif isinstance(mask_lines, list):
                self._mask_lines = []
                for line in mask_lines:
                    if len(line) == 2:
                        self._mask_lines.append(line + [-0.5, 6.5])
                    else:
                        self._mask_lines.append(line)
            elif isinstance(mask_lines, str):
                df = pd.read_csv(os.path.expandvars(mask_lines))
                df.loc[df['logg_min'].isna(), 'logg_min'] = -0.5
                df.loc[df['logg_max'].isna(), 'logg_max'] = 6.5

                self._mask_lines = df.to_numpy()
        else:
            self._mask_lines = mask_lines

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

        # get grid axes
        self._axes = [sorted(self._data[p].unique()) for p in self._parameters]

        # remove axes that contain only a single value
        for i, p in enumerate(self._parameters):
            if len(self._axes[i]) <= 1:
                del self._axes[i]
                self._parameters.remove(p)

        # initialize step counter
        self._step = 1

        # values of the fit parameter from previous iteration step
        self._previous_values = None

        # save weight array
        self._weights = None

        # save initial parameters
        self._initial_values = None

        self._logg = None

    def __call__(self, spectrum: Spectrum, filename: str) -> np.ndarray:
        """
        Creates and returns weight array.

        Args:
            spectrum: Spectrum to create weight for.
            filename: Name of spectrum file.

        Returns:
             Array containing the weight for given spectrum.
        """

        # save initial values
        if self._initial_values is None:
            self._initial_values = {}
            for cmp in self.objects['init_iter'].values():
                for param_name in cmp.param_names:
                    self._initial_values[param_name] = cmp[param_name]

                    if param_name == 'logg' and self._logg is None:
                        self._logg = cmp[param_name]

                break

        # load new weights if the fit parameters changed significantly
        new_weights = False
        if self._previous_values is not None:
            for param in self._parameters:
                if new_weights:
                    break
                for cmp in self.objects['init_iter'].values():
                    for param_name in cmp.param_names:
                        if param.lower() != param_name.lower():
                            continue

                        if param.lower() == 'teff':
                            # did Teff change significantly?
                            new_weights = abs(
                                self._previous_values[self._parameters.index(param)] - cmp[param_name]) > self._max_change[0]
                        else:
                            # did FeH, Alpha or logg change significantly?
                            new_weights = abs(
                                self._previous_values[self._parameters.index(param)] - cmp[param_name]) > self._max_change[1]

        # are current parameter values identical with initial values?
        if self._step > 1:
            tmp = []
            for cmp in self.objects['init_iter'].values():
                for param_name in cmp.param_names:
                    tmp.append(cmp[param_name] == self._initial_values[param_name])

                break

            # component is reset to initial values if the fit restarts with a damping factor, in that case the iteration
            #  step needs to be reset as well
            if np.all(tmp):
                self._step = 1

        # load new weights if max_step has not been reached or fit parameters changed significantly
        if (self._step <= self._max_step) or new_weights:
            if new_weights:
                self._step = 1

            # get parameters from component
            params = []
            for param in self._parameters:
                for cmp in self.objects['init_iter'].values():
                    for param_name in cmp.param_names:
                        if param.lower() != param_name.lower():
                            continue

                        params.append(cmp[param_name])

                    break

            # save current parameters for next step
            self._previous_values = params.copy()

            # interpolate weight for given values, use nearest neighbour if values are outside of the grid
            try:
                self._weight_table = self._interpolate(tuple(params))
            except KeyError:
                self._weight_table = None

                if self._mask_lines == 'default':
                    w = WeightFromGridNearest(self.filename, self._initial, self._max_line_depth, self._center_weight,
                                              self._max_step, objects=self.objects)
                else:
                    w = WeightFromGridNearest(self.filename, self._initial, self._max_line_depth, self._center_weight,
                                              self._max_step, self._mask_lines, objects=self.objects)

                self._weights = {step: w(spectrum, filename) for step in range(1, 6)}

        if self._weight_table is None:
            if self._step <= 5:
                w = self._weights[self._step]
                self._step += 1

                return w

            return self._weights[5]

        w = self._get_weight_array(self._weight_table, spectrum)
        self._step += 1

        return w

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
        df = df_grouped.filter(lambda x: len(x) == 1).copy()
        df['weights'] *= df['w'].values

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
        df = df[df['step'] <= self._step]
        df = df[df['weights'] > 10.]

        # initialize weight array
        weights = np.zeros(spectrum.wave.shape) + self._initial

        # write weights to array
        for i, row in df.iterrows():
            if isinstance(self._mask_lines, list) or isinstance(self._mask_lines, np.ndarray):
                if self._mask_centers(row, self._mask_lines, weights, spectrum):
                    continue
            elif self._mask_lines == 'default':
                if self._mask_default_lines(row, weights, spectrum):
                    continue

            weights[(spectrum.wave >= row['wave_start']) & (spectrum.wave <= row['wave_end'])] += row['weights']

            # if line depth larger than given threshold mask out the central region otherwise increase weight of
            # central pixel by given factor
            if row['line_depth'] > self._max_line_depth:
                # if region spans more than 10 wavelength pixel mask out the 3 central pixel otherwise only the central
                # one
                if (row['wave_end'] - row['wave_start']) >= 12:
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

    def _mask_default_lines(self, row: pd.Series, weights: np.ndarray, spectrum: Spectrum):
        # Halpha
        if (row['wave_center'] < 6566.) & (row['wave_center'] > 6557.) & (self._logg <= 3.5):
            weights[(spectrum.wave >= row['wave_start']) & (spectrum.wave <= row['wave_end'])] += row['weights']

            i = np.argmin(np.abs(spectrum.wave - row['wave_center']))
            weights[i - 1:i + 2] = 0
            return True
        elif (row['wave_center'] < 6566.) & (row['wave_center'] > 6557.) & (self._logg > 3.5):
            weights[(spectrum.wave >= row['wave_start']) & (spectrum.wave <= row['wave_end'])] += row['weights']

            i = np.argmin(np.abs(spectrum.wave - row['wave_center']))
            weights[i] = 0
            return True

        # Hbeta
        if (row['wave_center'] < 4867.) & (row['wave_center'] > 4857.):
            weights[(spectrum.wave >= row['wave_start']) & (spectrum.wave <= row['wave_end'])] += row['weights']

            i = np.argmin(np.abs(spectrum.wave - row['wave_center']))
            weights[i] = 0
            return True

        # FeI line
        if (row['wave_center'] < 5272.) and (row['wave_center'] > 5267.):
            weights[(spectrum.wave >= row['wave_start']) & (spectrum.wave <= row['wave_end'])] += row['weights']

            i = np.argmin(np.abs(spectrum.wave - row['wave_center']))
            weights[i - 1:i + 2] = 0
            return True

        # Ca triplet
        if (row['wave_center'] < 8508.) and (row['wave_center'] > 8490.):
            weights[(spectrum.wave >= row['wave_start']) & (spectrum.wave <= row['wave_end'])] += row['weights']

            i = np.argmin(np.abs(spectrum.wave - row['wave_center']))
            weights[i - 2:i + 3] = 0
            return True

        if (row['wave_center'] < 8553.) and (row['wave_center'] > 8530.):
            weights[(spectrum.wave >= row['wave_start']) & (spectrum.wave <= row['wave_end'])] += row['weights']

            i = np.argmin(np.abs(spectrum.wave - row['wave_center']))
            weights[i - 2:i + 3] = 0
            return True

        if (row['wave_center'] < 8672.) and (row['wave_center'] > 8651.):
            weights[(spectrum.wave >= row['wave_start']) & (spectrum.wave <= row['wave_end'])] += row['weights']

            i = np.argmin(np.abs(spectrum.wave - row['wave_center']))
            weights[i - 2:i + 3] = 0
            return True

        return False

    def _mask_centers(self, row: pd.Series, lines: Union[list, np.ndarray], weights: np.ndarray, spectrum: Spectrum):
        for center, npix, logg_min, logg_max in lines:
            if (row['wave_start'] < center) and (row['wave_end'] > center) and (self._logg < logg_max) and (
                    self._logg >= logg_min):
                weights[(spectrum.wave >= row['wave_start']) & (spectrum.wave <= row['wave_end'])] += row['weights']

                i = np.argmin(np.abs(spectrum.wave - row['wave_center']))

                if npix % 2 == 0:
                    weights[int(i-npix//2):int(i+npix//2)] = 0
                else:
                    weights[int(i-npix//2):int(i+npix//2+1)] = 0

                return True

        return False


__all__ = ['WeightFromGrid']
