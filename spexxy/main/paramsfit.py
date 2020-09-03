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
from spexxy.component import Component
from spexxy.mask import Mask
from spexxy.weight import Weight
from spexxy.data import SpectrumFitsHDU
from .base import FilesRoutine
from spexxy.tools.plot import plot_spectrum
from .baseparamsfit import BaseParamsFit, Legendre


class ParamsFit(BaseParamsFit):
    """ParamsFit is a fitting routine for spexxy that uses a Levenberg-Marquardt optimization to fit a set of
    model spectra (components) to a given spectrum.
    """

    def __init__(self, maxfev: int = 500, ftol: float = 1.49012e-08, xtol: float = 1.49012e-08,
                 factor: float = 100.0, epsfcn: float = 1e-7, *args, **kwargs):
        """Initialize a new ParamsFit object

        Args:
            maxfev: The maximum number of calls to the function (see scipy documentation).
            ftol: Relative error desired in the sum of squares (see scipy documentation).
            xtol: Relative error desired in the approximate solution (see scipy documentation).
            factor: A parameter determining the initial step bound (factor * || diag * x||) (see scipy documentation).
            epsfcn: A variable used in determining a suitable step length for the forward- difference approximation of
                the Jacobian (see scipy documentation)
        """
        BaseParamsFit.__init__(self, *args, **kwargs)

        # remember variables
        self._max_fev = maxfev
        self._ftol = ftol
        self._xtol = xtol
        self._factor = factor
        self._epsfcn = epsfcn

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

        # get parameters
        params = Parameters()
        for cmp in self.components:
            params += cmp.make_params()

        # open PDF
        if self._plot_iterations:
            self._iterations_pdf = PdfPages(filename.replace('.fits', '.pdf'))

        # start minimization
        self.log.info('Starting fit...')
        minimizer = lmfit.Minimizer(self._fit_func, params,
                                    iter_cb=self._callback, max_nfev=self._max_fev, nan_policy='raise',
                                    xtol=self._xtol, ftol=self._ftol, epsfcn=self._epsfcn, factor=self._factor)
        result = minimizer.leastsq()
        self.log.info('Finished fit.')

        # close PDF file
        if self._plot_iterations:
            self._iterations_pdf.close()

        # get best fit
        best_fit = self._get_model(result.params)

        # estimate SNR
        snr = None if best_fit is None else self._spec.estimate_snr(best_fit)
        self.log.info('Estimated S/N of %.2f.', snr)

        # successful, if minimization was a success
        success = result.success

        # get message
        message = "" if result.lmdif_message is None else result.lmdif_message.replace("\n", " ")

        # if any of the parameters was fitted close to their edge, fit failed
        for pn in result.params:
            # ignore all sigma values
            if pn.lower().find("sig") != -1 or pn.lower().find("tellurics") != -1:
                continue

            # get param
            p = result.params[pn]

            # get position of value within range for parameter
            pos = (p.value - p.min) / (p.max - p.min)

            # if pos < 0.01 or pos > 0.99, i.e. closer than 1% to the edge, fit failes
            if p.vary and (pos < 0.01 or pos > 0.99):
                success = False
                message = "Parameter %s out of range: %.2f" % (p.name, p.value)
                break

        # fill statistics dict
        stats = {'success': success,
                 'errorbars': result.errorbars,
                 'nfev': result.nfev,
                 'chisqr': result.chisqr,
                 'redchi': result.redchi,
                 'nvarys': result.nvarys,
                 'ndata': result.ndata,
                 'nfree': result.nfree,
                 'msg': message,
                 'snr': snr}

        # write results back to file
        self._write_results_to_file(filename, result, best_fit, stats)

        # all components
        components = self._cmps
        if self._tellurics is not None:
            components.append(self._tellurics)

        # build list of results and return them
        results = []
        for cmp in self._cmps:
            # parse parameters
            cmp.parse_params(result.params)

            # loop params
            for n in cmp.param_names:
                p = '%s%s' % (cmp.prefix, n)
                results += [cmp.parameters[n]['value'], cmp.parameters[n]['stderr']]

        # success?
        results += [success, result.redchi]
        return results


__all__ = ['ParamsFit']
