import gc
import os
import sqlite3
from spexxy.data import Spectrum
from typing import Tuple, List

from .grid import Grid, GridAxis


class PhoenixGrid(Grid):
    """Class for handling grid of PHOENIX spectra based on an sqlite database."""

    def __init__(self, path: str, *args, **kwargs):
        """Initializes a new PHOENIX grid.

        Args:
            path: Path to grid.
        """

        # try to find database file
        self._db_file = os.path.join(os.path.expandvars(path), 'gridParams.db')

        # get path
        self._dir = self._db_file[:self._db_file.rfind('/')]
        
        # connect
        self.sqlite = sqlite3.connect(self._db_file)

        # get axes infos
        axis_names = self._fetch_axis_names()
        axes = [GridAxis(name=name, values=self._fetch_axis_values(name)) for name in axis_names]

        # init grid
        Grid.__init__(self, axes, *args, **kwargs)
        self.log.info('Initializing Phoenix grid from %s...', self._db_file)
        self.log.info('Found axes: %s', ', '.join(axis_names))

        # fetch filenames
        self._filenames = {}
        self._fetch_filenames()

    def __contains__(self, params: Tuple) -> bool:
        """Checks, whether the grid contains a given parameter set.

        Args:
            params: Parameter set to check.

        Returns:
            Whether or not the given parameter set exists in the grid.
        """

        # convert to tuplt
        p = tuple(params)

        # check in database
        if p not in self._filenames:
            return False

        # does the file exist?
        elif not os.path.exists(self._filenames[p]):
            return False
        else:
            return True

    def all(self) -> List[Tuple]:
        """Return all possible parameter combinations.

        Returns:
            All possible parameter combinations
        """
        return self._filenames.keys()

    def __call__(self, params: Tuple) -> Spectrum:
        """Fetches the value for the given parameter set

        Args:
            params: Parameter set to catch value for.

        Returns:
            Grid value at given position.
        """

        # load spectrum
        return Spectrum.load(self.filename(tuple(params)))

    def filename(self, params: Tuple) -> str:
        """Returns the filename of a spectrum with the given paramaters.

        Args:
            params: Parameters of spectrum to return filename for.

        Returns:
            Filename of spectrum.
        """

        # does it exist?
        if params not in self:
            raise KeyError('No spectrum with given parameter ' + str(params) + 'exists in grid.')

        # return filename
        return self._filenames[tuple(params)]

    def _fetch_axis_names(self) -> List[str]:
        """Fetch information about all axes from the database.

        Returns:
            List of all axis names.
        """

        # query
        c = self.sqlite.cursor()
        c.execute("SELECT Name, Desc FROM Column")

        # loop all axes and fill lists
        names = []
        for row in c:
            names.append(row[0])
        c.close()
        return names

    def _fetch_axis_values(self, axis: int) -> List[float]:
        """Fetches all possible values for a given axis from the database.

        Args:
            axis: Number of axis to return values for.

        Returns:
            List of all possible values for axis.
        """

        # list of values
        values = []

        # query
        c = self.sqlite.cursor()
        sql = "SELECT DISTINCT %s FROM Model ORDER BY %s ASC" % (axis, axis)
        c.execute(sql)

        # add values
        for row in c:
            values.append(float(row[0]))
        c.close()

        # return values
        return values

    def _fetch_filenames(self):
        """Create a dictionary of parameter tuples and filenames from the database."""

        # query
        c = self.sqlite.cursor()
        c.execute("SELECT * FROM Model")
        for row in c:
            # filename
            filename = row[0]

            # loop all columns
            values = []
            for col in row[1:]:
                values.append(float(col))

            # add to filename cache
            self._filenames[tuple(values)] = self._dir + "/" + filename + ".fits"

        # finish up
        c.close()


__all__ = ['PhoenixGrid']
