from typing import List, Any, Tuple
import pandas as pd
import os

from .grid import Grid, GridAxis
from ..data import Spectrum


class FilesGrid(Grid):
    """Grid working on files with a CSV based database."""

    def __init__(self, filename: str, *args, **kwargs):
        """Constructs a new Grid.

        Args:
            filename: Filename of CSV file.
        """

        # expand filename
        filename = os.path.expandvars(filename)

        # get grid's root path
        self._root = os.path.dirname(filename)

        # load CSV
        self._data = pd.read_csv(filename, index_col=False)

        # get all parameters, by removing 'Filename' from list of columns
        parameters = list(self._data.columns)
        parameters.remove('Filename')

        # we assume that all parameters are floats, so treat them as such
        for name in parameters:
            self._data[name] = self._data[name].apply(lambda x: float(x))

        # create axes
        axes = [GridAxis(name=name, values=sorted(self._data[name].unique())) for name in parameters]

        # init grid
        Grid.__init__(self, axes, *args, **kwargs)

        # set index on dataframe
        self._data.set_index(parameters, inplace=True)

    def all(self) -> List[Tuple]:
        """Return all possible parameter combinations.

        Returns:
            All possible parameter combinations.
        """
        return self._data.index.values

    def __contains__(self, params: Tuple) -> bool:
        """Checks, whether the grid contains a given parameter set.

        Args:
            params: Parameter set to check.

        Returns:
            Whether or not the given parameter set exists in the grid.
        """
        return tuple(params) in self._data.index

    def filename(self, params: Tuple) -> str:
        """Returns filename for given parameter set.

        Args:
            params: Parameter set to catch value for.

        Returns:
            Filename.
        """
        return self._data.loc[tuple(params)].Filename

    def __call__(self, params: Tuple) -> Any:
        """Fetches the value for the given parameter set.

        Args:
            params: Parameter set to catch value for.

        Returns:
            Grid value at given position.
        """

        # get filename
        filename = self._data.loc[tuple(params)].Filename

        # return Spectrum
        return Spectrum.load(os.path.join(self._root, filename))


__all__ = ['FilesGrid']
