from typing import List, Tuple, Any

from ..object import spexxyObject
from ..grid import GridAxis


class Interpolator(spexxyObject):
    """Base class for all interpolators in spexxy."""

    def __init__(self, *args, **kwargs):
        """Initializes a new interpolator."""
        spexxyObject.__init__(self, *args, **kwargs)

    def axes(self) -> List[GridAxis]:
        """Returns information about the axes.

        Returns:
            List of GridAxis objects describing the grid's axes
        """
        raise NotImplementedError

    def __call__(self, params: Tuple) -> Any:
        """Interpolates at the given parameter set

        Args:
            params: Parameter set to interpolate at

        Returns:
            Interpolation result at given position
        """
        raise NotImplementedError


__all__ = ['Interpolator']
