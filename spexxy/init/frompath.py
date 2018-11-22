import os

from .init import Init
from ..data import FitsSpectrum
from ..component import Component


class InitFromPath(Init):
    """Initializes a component from another file in a given directory.

    This class, when called, initializes the parameters of a given component to the values in the header of
    a file with the same name (usually written in a previous fit) in the given directory.
    """

    def __init__(self, path: str, *args, **kwargs):
        """Initializes a new Init object.

        Args:
            path: Path in which to look for the file to read the initial values from.
        """
        Init.__init__(self, *args, **kwargs)
        self._path = path

    def __call__(self, cmp: Component, filename: str):
        """Initializes parameters of the given component with values from another file.

        Args:
            cmp: Component to initialize.
            filename: Name of file (in given path) to read initial values from.
        """

        # get filename
        fn = os.path.join(self._path, filename)
        if not os.path.exists(fn):
            self.log.error('Could not find file %s for reading initials.', fn)
            return

        # open file
        self.log.info('Reading initial values for component "%s" from %s...', cmp.prefix, fn)
        with FitsSpectrum(fn) as fs:
            # get results for this component
            results = fs.results(cmp.prefix)

            # loop parameters of component
            for param in cmp.param_names:
                # is it in results?
                if param in results:
                    # get value
                    value = results[param][0]

                    # set it
                    self.log.info('Setting initial value for "%s" of component "%s" to %f...', param, cmp.prefix, value)
                    cmp[param] = value


__all__ = ['InitFromPath']
