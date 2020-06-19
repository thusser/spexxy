from typing import List, Any, Tuple
import pandas as pd
import os

from .grid import Grid, GridAxis
from ..data import Spectrum


class SynspecGrid(Grid):
    """Synthesizes a new spectrum with Synspec at given grid positions."""

    def __init__(self, models: Grid, vturbs: Grid, synspec: str, inpmol: str, linelist: str, mollist: str, iat: float,
                 el_sol: float, *args, **kwargs):
        """Constructs a new Grid.

        Args:
            models: Grid with model atmospheres
            vturbs: Grid with micro turbulences
            synspec: Full path to synspec exectuble
            inpmol: Standard input for synspec
            linelist: File with line list
            mollist: File with molecular list
            iat: bla
            el_sol: bla
        """
        Grid.__init__(self, axes=None, *args, **kwargs)

        # store
        self._synspec = synspec
        self._inpmol = inpmol
        self._linelist = linelist
        self._mollist = mollist
        self._iat = iat
        self._el_sol = el_sol

        # load grids
        self._models: Grid = self.get_objects(models, Grid, 'grids', self.log, single=True)
        self._vturbs: Grid = self.get_objects(vturbs, Grid, 'grids', self.log, single=True)

        # init grid
        self._axes = self._models.axes()


    def all(self) -> List[Tuple]:
        """Return all possible parameter combinations.

        Returns:
            All possible parameter combinations.
        """
        return self._models.all()

    def __contains__(self, params: Tuple) -> bool:
        """Checks, whether the grid contains a given parameter set.

        Args:
            params: Parameter set to check.

        Returns:
            Whether or not the given parameter set exists in the grid.
        """
        return tuple(params[:-1]) in self._models

    def filename(self, params: Tuple) -> str:
        """Returns filename for given parameter set.

        Args:
            params: Parameter set to catch value for.

        Returns:
            Filename.
        """
        return None

    def __call__(self, params: Tuple) -> Any:
        """Fetches the value for the given parameter set.

        Args:
            params: Parameter set to catch value for.

        Returns:
            Grid value at given position.
        """
        print(params)

        # find element in models grid
        model_params = self._models.nearest(params[:-1])
        model = self._models.filename(model_params)
        print(model_params, model)



__all__ = ['SynspecGrid']
