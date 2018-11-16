import pandas as pd
import numpy as np

from .init import Init


class InitFromCsv(Init):
    """Initializes a component from a line in a CSV file.

    This class, when called, initializes the given parameters (or all if None) of a given component
    to the values in the given CSV file.
    """

    def __init__(self, filename: str = 'initials.csv', filename_col: str = 'Filename',
                 parameters: list = None, cmp_sep: str = ' ',
                 *args, **kwargs):
        """Initializes a new Init object.

        Args:
            filename: Name of CSV file.
            filename_col:  Name of column containing filename.
            parameters: List of parameter names to set from CSV.
            cmp_sep: String separating component and parameter name in CSV.
        """
        Init.__init__(self, *args, **kwargs)
        self._parameters = parameters
        self._cmp_sep = cmp_sep

        # load csv
        self.log.info('Reading CSV file with initial values from %s...', filename)
        self._csv = pd.read_csv(filename, index_col=filename_col)

    def __call__(self, cmp: 'Component', filename: str):
        """Initializes parameters of the given component with values from the CSV given in the configuration.

        Args:
            cmp: Component to initialize.
        filename: Filename of spectrum.
        """

        # got filename?
        try:
            self._csv.loc[filename]
        except KeyError:
            return

        # get lower case columns
        columns = {c.lower(): c for c in self._csv.columns}

        # parameters in component
        cmp_params = {c.lower(): c for c in cmp.param_names}

        # get list of parameters
        params = self._parameters if self._parameters is not None else cmp.param_names

        # and loop them
        for param in params:
            # lower case
            p = param.lower()

            # is it actually a parameter of the given component?
            if p in cmp_params:
                # find column for parameter
                col = None
                if cmp.prefix + self._cmp_sep + p in columns:
                    # <cmp>.<param> matches column name
                    col = columns[cmp.prefix + self._cmp_sep + p]
                elif p in columns:
                    # just <param> matches column name
                    col = columns[p]

                # found column?
                if col is not None:
                    # get value
                    val = self._csv.at[filename, col]
                    if np.isnan(val):
                        val = None

                    # set it, if we got a valid value
                    if val is not None:
                        self.log.info('Setting initial value for "%s" of component "%s" to %f...',
                                      param, cmp.prefix, val)
                        cmp[cmp_params[p]] = val


__all__ = ['InitFromCsv']
