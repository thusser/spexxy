import lmfit
import numpy as np
import scipy.linalg
from lmfit import Parameters
from lmfit.minimizer import MinimizerResult
from typing import List
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from spexxy.data import FitsSpectrum, Spectrum
from spexxy.component import Component, GridComponent
from spexxy.mask import Mask
from spexxy.weight import Weight
from spexxy.data import SpectrumFitsHDU
from .base import FilesRoutine
from spexxy.tools.plot import plot_spectrum
from .baseparamsfit import BaseParamsFit, Legendre


class BruteFit(BaseParamsFit):
    """ParamsFit is a fitting routine for spexxy that uses a brute force approach.
    """

    def __init__(self, *args, **kwargs):
        """Initialize a new BruteFit object"""
        BaseParamsFit.__init__(self, *args, **kwargs)

        # check components
        if len(self.components) != 1:
            raise ValueError('Only exactly one component is supported.')
        for cmp in self.components:
            if not isinstance(cmp, GridComponent):
                raise ValueError('All components must be GridComponents.')

    def __call__(self, filename: str) -> List[float]:
        """Start the fitting procedure on the given file.

        Args:
            filename: Name of file to fit.

        Returns:
            List of final values of parameters, ordered in the same way as the return value of parameters()
        """

        # fix any parameters?
        self._fix_params()

        # Load spectrum
        if not self._load_spectrum(filename):
            # something went wrong, return Nones as result
            return [None] * (len(self.columns()) - 2) + [False, 0]

        # get component
        cmp: GridComponent = self.components[0]

        # get parameters
        param_names = [cmp.prefix + name for name in cmp.grid.axis_names()]
        all_params = cmp.grid.all()

        # loop them
        best = None
        for params in all_params:
            # make parameters
            p = cmp.make_params(**dict(zip(param_names, params)))

            # get model
            model = self._get_model(p)

            # chi2
            chi2 = np.sum((self._spec.flux[self._valid] - model.flux[self._valid])**2 / model.flux[self._valid])
            print(params, chi2)

            if best is None or chi2 < best[0]:
                best = (chi2, params)

        print('BEST: ', best)
        # success?
        #results += [success, result.redchi]
        return []


__all__ = ['BruteFit']
