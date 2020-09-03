from spexxy.interpolator import Interpolator

from spexxy.grid import Grid
from .spectrumcomponent import SpectrumComponent


class GridComponent(SpectrumComponent):
    """A Grid component takes a grid and adds LOSVD parameters."""

    def __init__(self, grid: Grid = None, interpolator: Interpolator = None, name: str = "STAR", *args, **kwargs):
        """Initializes a new Grid component. Either a grid or an interpolator must be given.

        Parameters
        ----------
        grid : Grid
            The grid to use for the component
        interpolator : Interpolator
            The interpolator to use for the component
        name : str
            Name of the component
        """
        SpectrumComponent.__init__(self, name, self._model_func, *args, **kwargs)
        self.log.info('Initializing grid component "%s"...', name)

        # try to get grid
        try:
            self._grid = self.get_objects(grid, Grid, 'grids')
            if isinstance(self._grid, list):
                self._grid = self._grid[0]

            # add parameters of grid
            for a in self._grid.axes():
                self.log.info('Found parameter %s with min=%.2f, max=%.2f, and initial=%.2f.',
                              a.name, a.min, a.max, a.initial)
                self.set(a.name, min=a.min, max=a.max, value=a.initial, values=a.values)
        except (ValueError, IndexError):
            self._grid = None

        # try to get interpolator
        try:
            # try to get interpolator
            self._interpolator = self.get_objects(interpolator, Interpolator, 'interpolators')
            if isinstance(self._interpolator, list):
                self._interpolator = self._interpolator[0]

            # add parameters of grid
            for a in self._interpolator.axes():
                self.log.info('Found parameter %s with min=%.2f, max=%.2f, and initial=%.2f.',
                              a.name, a.min, a.max, a.initial)
                self.set(a.name, min=a.min, max=a.max, value=a.initial)
        except (ValueError, IndexError):
            self._interpolator = None

        # we need either
        if self._grid is None and self._interpolator is None:
            raise ValueError('Neither grid nor interpolator given.')
        if self._grid is not None and self._interpolator is not None:
            raise ValueError('Both grid and interpolator given, while only one should be.')

    @property
    def grid(self):
        """Returns grid."""
        return self._grid

    @property
    def interpolator(self):
        """Returns interpolator."""
        return self._interpolator

    def _model_func(self):
        """Get spectrum with given parameters.

        Returns
        -------
        Any
            Result from grid/interpolator
        """

        # grid or interpolator?
        if self._grid is not None:
            values = tuple([self[a.name] for a in self._grid.axes()])
            return self._grid(values)
        else:
            values = tuple([self[a.name] for a in self._interpolator.axes()])
            return self._interpolator(values)


__all__ = ['GridComponent']
