import numpy as np
from typing import List, Tuple

from . import Interpolator
from ..grid import Grid, GridAxis
from ..data import Spectrum


class SplineInterpolator(Interpolator):
    """A cubic spline interpolator that operates on a given grid."""

    def __init__(self, grid: Grid, derivs: Grid = None, *args, **kwargs):
        """Initializes a new linear interpolator.

        Args:
            grid: Grid to interpolate on.
            derivs: If given, contains a second grid at the same parameters as grid, but containg 2nd derivatives for
                the first axis of the grid.
        """
        Interpolator.__init__(self, *args, **kwargs)

        # grids
        self.log.info('Initializing spline interpolator...')
        self._grid = self.get_objects(grid, Grid, 'grids', single=True)
        self._derivs = self.get_objects(derivs, Grid, 'grids', single=True)

        # init
        self._axes = self._grid.axes()

    @property
    def grid(self) -> Grid:
        """Returns grid used in this interpolator.

        Returns:
            Grid used for this interpolator
        """
        return self._grid

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
        return self._interpolate(tuple(params))

    def _interpolate(self, params: Tuple, axis: int = None):
        """Do the actual interpolation at the given axis, then continue for axis-1.

        Args:
            params: Parameter set to interpolate at.
            axis: Axis to interpolate in - if None, we start at last available axis.

        Returns:
            Interpolated spectrum at given position.
        """

        # no axis given, start at latest
        if axis is None:
            axis = len(self._axes) - 1

        # check boundaries
        if params[axis] < self._axes[axis].min or params[axis] > self._axes[axis].max:
            raise KeyError('Requested parameters are outside the grid.')

        # let's get all possible values for the given axis
        axisValues = self._grid.axis_values(axis)

        # if params[axis] is on axis; return it directly
        if params[axis] in axisValues:
            if axis == 0:
                return self._grid(tuple(params))
            else:
                return self._interpolate(tuple(params), axis - 1)

        # find the next lower and the next higher axis value
        p_lower = self._grid.neighbour(tuple(params), axis, 0)
        p_higher = self._grid.neighbour(tuple(params), axis, 1)
        if p_lower is None or p_higher is None:
            raise KeyError('No direct neighbours found in grid.')

        # get axis values
        x_lower = p_lower[axis]
        x_higher = p_higher[axis]

        # get data for p_lower and p_higher
        if axis == 0:
            lower_data = self._grid(p_lower)
            higher_data = self._grid(p_higher)
        else:
            lower_data = self._interpolate(p_lower)
            higher_data = self._interpolate(p_higher)

        # do we have pre-calculated 2nd derivatives?
        if axis == 0 and self._derivs is not None:
            lower_deriv = self._derivs(p_lower)
            higher_deriv = self._derivs(p_higher)
        else:
            # find the next lower and the next higher axis value
            try:
                p2_lower = self._grid.neighbour(tuple(params), axis, -1)
                p2_higher = self._grid.neighbour(tuple(params), axis, 2)

                # get x and y values
                x = [p2_lower[axis], x_lower, x_higher, p2_higher[axis]]
                y = [self._interpolate(p2_lower), lower_data, higher_data, self._interpolate(p2_higher)]

                # calculate 2nd derivatives
                y2 = np.zeros((4, len(lower_data)))
                u = np.zeros((4, len(lower_data)))
                for i in [1, 2]:
                    sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
                    p = sig * y2[i - 1] + 2.0
                    y2[i] = (sig - 1.0) / p
                    u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1])
                    u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p
                for k in [2, 1]:
                    y2[k] = y2[k] * y2[k + 1] + u[k]

                # set them
                lower_deriv = y2[1]
                higher_deriv = y2[2]

            except KeyError:
                # if anything here fails, fall back to a zero (array), which effectively is a linear interpolation
                if hasattr(lower_data, '__iter__'):
                    lower_deriv = np.zeros((len(lower_data)))
                    higher_deriv = np.zeros((len(higher_data)))
                else:
                    lower_deriv = 0.
                    higher_deriv = 0.

        # calculate interpolation
        A = (x_higher - params[axis]) / (x_higher - x_lower)
        B = 1. - A
        C = 1. / 6. * (A * A * A - A) * (x_higher - x_lower) * (x_higher - x_lower)
        D = 1. / 6. * (B * B * B - B) * (x_higher - x_lower) * (x_higher - x_lower)
        ip = lower_data * A + higher_data * B + lower_deriv * C + higher_deriv * D
        return ip


__all__ = ['SplineInterpolator']
