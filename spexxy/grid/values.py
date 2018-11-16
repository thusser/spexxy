from typing import List, Dict, Any, Tuple

from .grid import Grid, GridAxis


class ValuesGrid(Grid):
    """Basic Grid that gets its values from the constructor. Mainly used for testing."""

    def __init__(self, axes: List[GridAxis], values: Dict[Tuple, Any], *args, **kwargs):
        """Constructs a new Grid.

        Args:
            axes: A list of axes to build the grid from.
            values: A dictionary where the keys are the parameters and the values the values at that
                position in the grid.
        """
        Grid.__init__(self, axes, *args, **kwargs)

        # remember values
        self._values = values

    def all(self) -> List[Tuple]:
        """Return all possible parameter combinations.

        Returns:
            All possible parameter combinations.
        """
        return list(self._values.keys())

    def __contains__(self, params: Tuple) -> bool:
        """Checks, whether the grid contains a given parameter set.

        Args:
            params: Parameter set to check.

        Returns:
            Whether or not the given parameter set exists in the grid.
        """
        return tuple(params) in self._values

    def __call__(self, params: Tuple) -> Any:
        """Fetches the value for the given parameter set

        Args:
            params: Parameter set to catch value for

        Returns:
            Grid value at given position
        """
        return self._values[params]


__all__ = ['ValuesGrid']
