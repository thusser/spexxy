from typing import List, Tuple, Dict, Any

from . import Grid
from ..grid import GridAxis


class SubGrid(Grid):
    """A grid that only gives access to a part of the parent grid."""

    def __init__(self, parent: Grid, fixed: Dict[str, float], *args, **kwargs):
        """Initializes a new sub grid.

        Args:
            parent: Parent grid.
            fixed: Dictionary with fixed axis_name->value pairs.
        """

        # store
        self._parent = parent
        self._fixed = {}

        # get axes
        axes: List[GridAxis] = []
        for ax in parent.axes():
            if ax.name in fixed.keys():
                self._fixed[ax.name] = fixed[ax.name]
            else:
                axes.append(ax)

        # init Grid
        Grid.__init__(self, axes, *args, **kwargs)

        # log
        self.log.info('Initializing sub interpolator...')
        for ax, val in self._fixed.items():
            self.log.info('Fixed axis %s to value %f.', ax, val)

    def __call__(self, params: Tuple) -> Any:
        """Fetches the value for the given parameter set

        Args:
            params: Parameter set to catch value for.

        Returns:
            Grid value at given position.
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


__all__ = ['SubGrid']
