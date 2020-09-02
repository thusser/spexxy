import numpy as np
from typing import List, Tuple, Any

from ..object import spexxyObject


class GridAxis(object):
    """Description of a single axis in a grid."""

    def __init__(self, name: str, values: List = None, min: float = None, max: float = None, initial: float = None):
        """Initialize a new grid axis.

        Args:
            name: Name of new axis.
            values: List of all possible values for this axis.
            min: Minimum value for this axis.
            max: Maximum value for this axis.
            initial: Initial guess for this axis.
        """

        # remember name
        self.name = name

        # store values and get min/max
        self.values = None if values is None else sorted(values)
        if (min is None or max is None) and values is not None:
            self.min = self.values[0]
            self.max = self.values[-1]
        else:
            self.min = min
            self.max = max

        # store initial value - if none is given, take mean of values
        self.initial = np.mean(self.values) if initial is None else initial

    def neighbour(self, value: float, distance: int=1) -> float:
        """Finds a neighbour in this axis for the given value in the given distance.

        Args:
            value: Value to find neighbour for.
            distance: Distance in which to find neighbour
                >0:  Find larger neighbours, i.e. 0 next larger value, 1 the one after that, etc
                <=0:  Find smaller neighbours, i.e. 0 next smaller value (or value itself), -1 the before that, etc

        Returns:
            Value on grid in the given distance to the given value. If given value is on grid, distance is
            counted from that value.

        Raises:
            KeyError: If no neighbour has been found.
        """

        # loop all values
        for i in range(len(self.values)):
            # found value?
            if self.values[i] <= value < self.values[i + 1]:
                # index of neighour
                ii = i + distance
                # does it exist?
                if 0 <= ii < len(self.values):
                    return self.values[ii]

        else:
            # nothing found
            raise KeyError('No neighbour found.')


class Grid(spexxyObject):
    """Base class for all grids in spexxy."""

    def __init__(self, axes: List[GridAxis], *args, **kwargs):
        """Initialize a new Grid.

        Args:
            axes: List of axes for this grid.
        """
        spexxyObject.__init__(self, *args, **kwargs)

        # store axes
        self._axes = axes

    def num_axes(self) -> int:
        """Returns number of axes.

        Returns:
            Number of axes in grid.
        """
        return len(self._axes)

    def axes(self) -> List[GridAxis]:
        """Returns information about the axes.

        Returns:
            List of GridAxis objects describing the grid's axes.
        """
        return self._axes

    def axis_values(self, axis: int) -> List[float]:
        """Returns all possible values for the given axis.

        Args:
            axis: Index of axis to return values for.

        Returns:
            All possible values for the given axis.
        """
        return self._axes[axis].values

    def axis_name(self, axis: int) -> str:
        """Returns name of given axis.

        Args:
            axis: Index of axis to return values for.

        Returns:
            Name of given axis.
        """
        return self._axes[axis].name

    def axis_names(self) -> List[str]:
        """Returns names of all axis.

        Returns:
            Names of all axis.
        """
        return [axis.name for axis in self._axes]

    def all(self) -> List[Tuple]:
        """Return all possible parameter combinations.

        Returns:
            All possible parameter combinations.
        """
        raise NotImplementedError

    def __contains__(self, params: Tuple) -> bool:
        """Checks, whether the grid contains a given parameter set.

        Args:
            params: Parameter set to check.

        Returns:
            Whether or not the given parameter set exists in the grid.
        """
        raise NotImplementedError

    def __call__(self, params: Tuple) -> Any:
        """Fetches the value for the given parameter set

        Args:
            params: Parameter set to catch value for.

        Returns:
            Grid value at given position.
        """
        raise NotImplementedError

    def create_array(self) -> Any:
        """In case the values provided by this grid are of an array type, this method creates an empty array that
        can be filled.

        Returns:
            Empty array element of same type as values in the grid.
        """

        # fetch first item from grid
        data = self(next(iter(self.all())))

        # zero and return it
        data *= 0.
        return data

    def neighbour(self, params: Tuple, axis: int, distance: int=1, must_exist: bool = False) -> Tuple:
        """Finds a neighbour on the given axis for the given value in the given distance.

        Args:
            params: Parameter tuple to search neighbour from.
            axis: Axis to search for
            distance: Distance in which to find neighbour.
                >0:  Find larger neighbours, i.e. 0 next larger value, 1 the one after that, etc
                <=0:  Find smaller neighbours, i.e. 0 next smaller value (or value itself), -1 the before that, etc
            must_exist: Grid point with new parameter set must actually exist.

        Returns:
            New parameter tuple with neighbour on the given axis.
        """

        # create new tuple
        p = list(params)

        # if we don't enfore the new point to exist, it's easy
        if not must_exist:
            # find neighbour in axis
            value = self._axes[axis].neighbour(params[axis], distance=distance)
            if value is None:
                return None

            # create new tuple
            p[axis] = value
            return tuple(p)

        # grid point does not exist, get all axis values
        axis_values = np.array(self._axes[axis].values)

        # lower or higher values?
        if distance <= 0:
            # smaller values
            axis_values = sorted(axis_values[axis_values < params[axis]], reverse=True)
            # define steps to take
            steps = abs(distance)
        else:
            # larger values
            axis_values = sorted(axis_values[axis_values > params[axis]], reverse=False)
            # define steps to take, subtract 1, so that 1 (first larger) is index 0
            steps = distance - 1

        # find parameter set that exists
        for v in axis_values:
            # set value
            p[axis] = v

            # check it
            if self.__contains__(tuple(p)):
                # exists, does distance match, i.e. steps=0?
                if steps == 0:
                    # return tuple
                    return tuple(p)
                else:
                    # otherwise, reduce steps and go on
                    steps -= 1
        else:
            # nothing found
            return None

    def nearest(self, params, scales=None) -> Tuple:
        """Finds the nearest point within the grid.

        Calculates distance between given params and all points p in the grid as the sum over all elements in
        the parameter tuple of ((params - grid_point) * scales)**2.

        Args:
            params: Parameters to find nearest point to.
            scales: If given, scales dimensions.

        Returns:
            Parameters for nearest point in grid.
        """

        # check scales
        num_axes = self.num_axes()
        if scales and len(scales) != num_axes:
            raise ValueError('Invalid number of scales, must be identical '
                             'to number of axes.')
        scales = np.array(scales) if scales is not None else np.ones((num_axes))

        # check params
        if len(params) != num_axes:
            raise ValueError('Number of parameters must be equal to '
                             'number of axes.')

        # get all parameters values and convert them to numpy arrays
        all_params = self.all()

        # calc squared distances and scale:
        # - distance: params - p
        # - scale:    * scales
        # - squared:  **2
        # - sum up:   np.sum
        # There is no need to take the square root, since we're just
        # comparing values.
        dist = [np.sum(((params - np.array(p)) * scales)**2) for p in all_params]

        # get best match
        return all_params[np.argmin(dist)]


__all__ = ['GridAxis', 'Grid']
