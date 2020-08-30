import numpy as np
from typing import List, Tuple

from . import Interpolator
from ..grid import Grid, GridAxis
from ..data import Spectrum


class SplineInterpolator(Interpolator):
    """A cubic spline interpolator that operates on a given grid."""

    def __init__(self, grid: Grid, derivs: Grid = None, n: int = 1, verbose: bool = False, *args, **kwargs):
        """Initializes a new linear interpolator.

        Args:
            grid: Grid to interpolate on.
            derivs: If given, contains a second grid at the same parameters as grid, but containg 2nd derivatives for
                the first axis of the grid.
            n: Number of points on each side to use for calculating derivatives.
            verbose: If True, output some more logs
        """
        Interpolator.__init__(self, *args, **kwargs)

        # grids
        self.log.info('Initializing spline interpolator...')
        self._grid: Grid = self.get_objects(grid, Grid, 'grids', single=True)
        self._derivs: Grid = self.get_objects(derivs, Grid, 'grids', single=True)

        # init
        self._axes = self._grid.axes()
        self._npoints = n
        self._verbose = verbose

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
            if axis == 0 and tuple(params) in self._grid:
                # if this is the first axis AND params are in grid, return it
                return self._grid(tuple(params))
            elif axis > 0:
                # if it's another axis, interpolate
                return self._interpolate(tuple(params), axis - 1)
            # if it's neither, just continue

        # find the next lower and the next higher axis value
        # on the last axis we enforce that the neighbour must exist
        p_lower = self._grid.neighbour(tuple(params), axis, 0, must_exist=axis == 0)
        p_higher = self._grid.neighbour(tuple(params), axis, 1, must_exist=axis == 0)
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
            # set them
            lower_deriv = self._derivs(p_lower)
            higher_deriv = self._derivs(p_higher)

        else:
            # get data for a maximum of N lower and higher axis values
            x = []
            y = []
            ilower = None
            for i in range(-self._npoints, 2+self._npoints):
                if i == 0:
                    # we got this value already, store it
                    x.append(x_lower)
                    y.append(lower_data)
                    # set ilower index to this index
                    ilower = len(x) - 1
                elif i == 1:
                    # we got this value already, store it
                    x.append(x_higher)
                    y.append(higher_data)
                else:
                    # try to fetch additional data
                    try:
                        p = self._grid.neighbour(tuple(params), axis, i, must_exist=axis == 0)
                        if p is not None:
                            y.append(self._interpolate(p))
                            x.append(p[axis])
                    except KeyError:
                        pass

            # calculate 2nd derivatives
            if self._verbose:
                self.log.info('Creating 2nd derivates for axis %s with values %s.', self._grid.axis_name(axis), x)
            y2 = self._spline(x, y)

            # set them
            lower_deriv = y2[ilower]
            higher_deriv = y2[ilower + 1]

        # log
        if self._verbose:
            self.log.info('Interpolate for axis %s at %.2f from neighbours at %.2f and %.2f.',
                          self._grid.axis_name(axis), params[axis], x_lower, x_higher)
            self.log.info('2nd derivation at 1st point is %.2g for next lower and %.2g for next higher neighbour.',
                          lower_deriv[0], higher_deriv[1])

        # calculate coefficients
        A = (x_higher - params[axis]) / (x_higher - x_lower)
        B = 1. - A
        C = (A**3 - A) * (x_higher - x_lower)**2 / 6.
        D = (B**3 - B) * (x_higher - x_lower)**2 / 6.
        if self._verbose:
            self.log.info('A=%.2f, B=%.2f, C=%.2f, D=%.2f', A, B, C, D)

        # interpolate
        return lower_data * A + higher_data * B + lower_deriv * C + higher_deriv * D

    def _spline(self, x: np.ndarray, y: np.ndarray, yp1=np.inf, ypn=np.inf) -> np.ndarray:
        """Calculates the 2nd derivatives for a spline.
        Python conversion from the C++ code in chapter "Cubic Spline Interpolation" in the
        "Numerical Recipes in C++, 2nd Edition".

        Args:
            x: Input x values.
            y: Input y values.
            yp1: First derivative at point 0. If set to np.inf, use natural boundary condition and set 2nd deriv to 0.
            ypn: First derivative at point n-1. np.inf means the same as for yp1.

        Returns:
            Second derivates for all points given by x and y.
        """

        # get number of elements
        n = len(x)

        # create arrays for u and 2nd derivs
        if hasattr(y[0], '__iter__'):
            y2 = np.zeros((n, len(y[0])))
            u = np.zeros((n, len(y[0])))
        else:
            y2 = np.zeros((n))
            u = np.zeros((n))

        # derivatives for point 0 given?
        if not np.isinf(yp1):
            y2[0] += -0.5
            u[0] += (3. / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1)

        # decomposition loop of the tridiagonal algorithm
        for i in range(1, n-1):
            sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
            p = sig * y2[i - 1] + 2.0
            y2[i] = (sig - 1.0) / p
            u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1])
            u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p

        # derivatives for point n-1 given?
        if not np.isinf(ypn):
            qn = 0.5
            un = (3. / (x[n-1] - x[n-2])) * (ypn - (y[n-1] - y[n-2]) / (x[n-1] - x[n-2]))
            y2[n-1] += (un - qn * u[n-2]) / (qn * y2[n-2] + 1.)

        # backsubstitution loop of the tridiagonal algorithm
        for k in range(n-2, 0, -1):
            y2[k] = y2[k] * y2[k + 1] + u[k]

        # finished
        return y2


__all__ = ['SplineInterpolator']
