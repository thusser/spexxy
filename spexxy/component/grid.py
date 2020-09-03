from spexxy.grid import Grid
from .spectrumcomponent import SpectrumComponent


class GridComponent(SpectrumComponent):
    """A Grid component takes a grid and adds LOSVD parameters."""

    def __init__(self, grid: Grid, name: str = "STAR", *args, **kwargs):
        """Initializes a new Grid component.

        Parameters
        ----------
        grid : Grid
            The grid to use for the component
        name : str
            Name of the component
        """
        SpectrumComponent.__init__(self, name, self._model_func, *args, **kwargs)
        self.log.info('Initializing grid component "%s"...', name)

        # get interpolator
        self._grid = self.get_objects(grid, Grid, 'grids')
        if isinstance(self._grid, list):
            self._grid = self._grid[0]

        # add parameters of grid
        for a in self._grid.axes():
            self.log.info('Found parameter %s with min=%.2f, max=%.2f, and initial=%.2f.',
                          a.name, a.min, a.max, a.initial)
            self.set(a.name, min=a.min, max=a.max, value=a.initial, values=a.values)

    def _model_func(self):
        """Get spectrum with given parameters.

        Returns
        -------
        Any
            Result from interpolator
        """

        # get values as tuple
        values = tuple([self[a.name] for a in self._grid.axes()])

        # interpolate
        return self._grid(values)


__all__ = ['GridComponent']
