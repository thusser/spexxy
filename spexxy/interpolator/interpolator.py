from typing import List, Tuple, Any

from ..object import spexxyObject
from ..grid import GridAxis


class Interpolator(spexxyObject):
    """Base class for all interpolators in spexxy."""
    def __init__(self, cache_level: int = 0, *args, **kwargs):
        """Initializes a new interpolator.

         Args:
            cache_level: Level of caching: 0 means off, 1 means only last dimension, 2 is last 2 dimensions and so on.
                Interpolation might be faster with higher level, but will consume significantly more memory.
        """
        spexxyObject.__init__(self, *args, **kwargs)
        self.cache_level = cache_level
        self.cache = {}

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

    def clear_cache(self):
        """Clear cache."""
        self.cache = {}


__all__ = ['Interpolator']
