from typing import List, Tuple, Dict

from . import Interpolator
from ..grid import GridAxis
from ..data import Spectrum


class SubInterpolator(Interpolator):
    """An interpolator that only gives access to a part of the parent interpolator."""

    def __init__(self, parent: Interpolator, fixed: Dict[str, float], *args, **kwargs):
        """Initializes a new sub interpolator.

        Args:
            parent: Parent interpolator.
            fixed: Dictionary with fixed axis_name->value pairs.
        """
        Interpolator.__init__(self, *args, **kwargs)
        self.log.info('Initializing sub interpolator...')

        # store
        self._parent = parent
        self._fixed = fixed

        # get axes
        self._axes: List[GridAxis] = []
        for ax in parent.axes():
            if ax.name in fixed.keys():
                self.log.info('Fixing axis %s to value %f.', ax.name, fixed[ax.name])
            else:
                self._axes.append(ax)

    def axes(self) -> List[GridAxis]:
        """Returns information about the axes.

        Returns:
            List of GridAxis objects describing the grid's axes
        """
        return self._axes

    def __call__(self, params: Tuple) -> Spectrum:
        """Interpolates at the given parameter set

        Args:
            params: Parameter set to interpolate at

        Returns:
            Interpolated spectrum at given position
        """

        # convert given params to list
        given_params = list(params)

        # loop all parameters
        full_params = []
        for ax in self._parent.axes():
            if ax.name in self._fixed.keys():
                # if axis is fixed, use fixed value
                full_params.append(self._fixed[ax.name])
            else:
                # otherwise pull next value from given_params
                full_params.append(given_params.pop(0))

        # call parent interpolator
        return self._parent(tuple(full_params))


__all__ = ['SubInterpolator']
