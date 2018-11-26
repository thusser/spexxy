import sys
import lmfit
import numpy as np
import scipy.linalg
from lmfit import Parameters
from lmfit.minimizer import MinimizerResult
from typing import List

from spexxy.data import FitsSpectrum, Spectrum
from spexxy.component import Component
from spexxy.mask import Mask
from spexxy.weight import Weight
from spexxy.data import SpectrumFitsHDU
from spexxy.object import spexxyObject
from .main import MainRoutine


class MultiMain(MainRoutine):
    """MultiRun iterates over a given set of main routines and runs them sequentially."""

    def __init__(self, routines: List = None, iterations: int = 1, *args, **kwargs):
        """Initialize a new MultiRun object

        Args:
            routines: List of main routines to run.
            iterations: Number of iterations for the whole cycle.
        """
        spexxyObject.__init__(self, *args, **kwargs)

        # remember variables
        self._iterations = iterations

        # find main routines
        self._routines = self.get_objects(routines, MainRoutine, 'routines')

    def parameters(self) -> List[str]:
        """Get list of parameters fitted by this routine.

        Returns:
            List of parameter names (including prefix) fitted by this routine.
        """

        # get all parameters
        parameters = []
        for routine in self._routines:
            parameters.extend(routine.parameters())

        # make unique and sort
        return sorted(list(set(parameters)))

    def __call__(self, filename: str) -> List[float]:
        """Process the given file.

        Args:
            filename: Name of file to process.

        Returns:
            List of final values of parameters, ordered in the same way as the return value of parameters()
        """

        # init results dict with Nones
        parameters = self.parameters()
        results = {p: None for p in parameters}

        # loop iterations
        for it in range(self._iterations):
            # loop main routines
            for routine in self._routines:
                # get parameters for this routine
                params = routine.parameters()

                # run routine
                res = routine(filename)

                # store results
                for i, p in enumerate(params):
                    # copy both results and errors!
                    results[p] = [res[i*2], res[i*2+1]]

        # convert results dict into results list
        res = []
        for p in parameters:
            res.extend(results[p])
        return res


__all__ = ['MultiMain']
