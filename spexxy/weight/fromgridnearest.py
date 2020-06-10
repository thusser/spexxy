import os

import numpy as np
import pandas as pd
from typing import List, Union

from .weight import Weight
from ..data import Spectrum


class WeightFromGridNearest(Weight):
    """
    This class loads the weights from a grid depending on the initial values of the fit parameters by choosing the
     nearest neighbour in the grid. It returns an array containing the weights.
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

        if 'objects' in kwargs:
            self.objects = kwargs['objects']

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

        # get grid axes
        self._axes = [np.array(sorted(self._data[p].unique())) for p in self._parameters]

        # remove axes that contain only a single value
        for i, p in enumerate(self._parameters):
            if len(self._axes[i]) <= 1:
                del self._axes[i]
                self._parameters.remove(p)

        self._data.set_index(keys=self._parameters, inplace=True)

        # initialize step counter
        self._step = 1

        # values of the fit parameter from previous iteration step
        self._previous_values = None

        self._filename = None

        # weights will be stored for next iterations
        self._weights = None
        self._neighbour = None

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
                            # did Teff change by more than 300K?
                            new_weights = abs(
                                self._previous_values[self._parameters.index(param)] - cmp[param_name]) > self._max_change[0]
                        else:
                            # did FeH, Alpha or logg change by more than 0.3 dex?
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

            # find nearest neighbour
            self._neighbour = []
            for i, p in enumerate(params):
                self._neighbour.append(self._axes[i][np.argmin(np.abs(self._axes[i] - p))])

            # save filename of weight table
            self._filename = self._data.loc[tuple(self._neighbour)].Filename

        # load weights
        w = self._load_weights(spectrum)

        # increase step counter
        self._step += 1

        return w

    def _load_weights(self, spectrum: Spectrum):
        """
        Load CSV file from grid and create weight array.

        Args:
            filename: Filename of CSV file that contains the weights.
            spectrum: Spectrum to create the weight for.

        Returns:
             Weight array.
        """

        # load table containing the weights
        df = pd.read_csv(os.path.join(self._root, self._filename))

        # consider only weights for iteration steps lower/equal than the given one
        df = df[df['step'] <= self._step]

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


__all__ = ['WeightFromGridNearest']
