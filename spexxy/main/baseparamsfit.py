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


class Legendre:
    """Calculate 'continuum' between a given spectrum and a model as a
    linear combination of Legendre polynomials."""

    def __init__(self, model, degree):
        """Initialize continuum fitter.

        :param model:   Model to use for deriving the continuum.
        :param degree:  Degree of Legendre polynomial to use.
        """

        # store variables
        self.model = model
        self.degree = degree
        self.coefficients = None
        self.mean = None
        self.values = np.ones((len(model)))

        # get wavelength array of model and normalize it to range -1..1
        # since we don't have mask here yet, this gets a little more complicated...
        wave = self.model.wave
        x = (wave - 0.5 * (wave[0] + wave[-1])) / 2.
        x /= np.max(np.abs(x[~np.isnan(x)]))

        # create Legendre polynomials
        # need to add 1 to polynomials to avoid zero points
        self._legendre = np.zeros((len(x), degree))
        for k in range(degree):
            self._legendre[~np.isnan(x), k] = scipy.special.legendre(k)(x[~np.isnan(x)]) + 1.

    def __call__(self, spec, valid=None, norm_to_mean=True):
        """Calculate the continuum.
        Effectively we're searching for the best coefficients c that
        fulfil S=c*A, where S is the flux of the spectrum and A is the
        matrix multiplication of all Legendre polynomials with the flux M of
        the model, i.e. A=L*M.t, where * denotes the matrix multiplication and
        M.t is the transposed of M.

        :param spec:    Spectrum to calculate continuum for.
        :param valid:   Array containing valid pixels in spectrum.
        :return:        numpy array containing continuum.
        """

        # check length
        if len(spec) != self._legendre.shape[0]:
            raise ValueError('Data length must be equal to initial data '
                             'length.')

        # valid
        if valid is None:
            valid = ~np.isnan(spec.flux)

        # add invalid model values to mask
        v = valid & ~np.isnan(self.model.flux)

        # fill matrix A as a matrix multiplication of L and the transposed
        # flux of the model
        A = np.empty((len(spec.flux[v]), self.degree))
        for k in range(self.degree):
            A[:, k] = spec.flux[v] * self._legendre[v, k]

        # solve, i.e. find best coefficients c for S=c*A
        self.coefficients = scipy.linalg.lstsq(A, self.model.flux[v])[0]

        # evaluate continuum
        self.values = np.dot(self._legendre, self.coefficients)

        # normalize to mean?
        if norm_to_mean:
            # divide evaluated polynomial by mean
            self.mean = self.values.mean()
            self.values /= self.mean

            # also divide first coefficient by mean
            self.coefficients[0] /= self.mean

        # return values
        return self.values


class BaseParamsFit(FilesRoutine):
    """ParamsFit is a fitting routine for spexxy that uses a Levenberg-Marquardt optimization to fit a set of
    model spectra (components) to a given spectrum.
    """

    def __init__(self, components: List[Component] = None, tellurics: Component = None, masks: List[Mask] = None,
                 weights: List[Weight] = None, fixparams: List[str] = None, poly_degree: int = 40,
                 min_valid_pixels: float = 0.5, plot_iterations: bool = False, *args, **kwargs):
        """Initialize a new ParamsFit object

        Args:
            components: List of components (or descriptions to create them) to fit to the spectrum
            tellurics: Tellurics component to add to the fit.
            masks:  List of mask objects to use for masking the spectrum.
            weights: List of weight objects to use for creating weights for the spectrum.
            fixparams: List of names of parameters to fix during the fit.
            poly_degree: Number of coefficients for the multiplicative polynomial.
            min_valid_pixels: Fraction of minimum number of required pixels to continue with fit.
            plot_iterations: Plot all iterations into a PDF file.
        """
        FilesRoutine.__init__(self, *args, **kwargs)

        # remember variables
        self._poly_degree = poly_degree
        self._fixparams = fixparams
        self._min_valid_pixels = min_valid_pixels

        # spectrum
        self._spec = None
        self._valid = None
        self._weight = None

        # model
        self._model = None

        # polynomial
        self._mult_poly = None

        # iterations PDF?
        self._plot_iterations = plot_iterations
        self._iterations_pdf = None

        # find components
        self._cmps = self.get_objects(components, Component, 'components')

        # find tellurics
        self._tellurics = self.get_objects(tellurics, Component, 'components', single=True)

        # masks
        self._masks = self.get_objects(masks, Mask, 'masks')

        # weights
        self._weights = self.get_objects(weights, Weight, 'weights')

    def parameters(self) -> List[str]:
        """Get list of parameters fitted by this routine.

        Returns:
            List of parameter names (including prefix) fitted by this routine.
        """

        # init
        params = []

        # loop components
        for cmp in self._cmps:
            # add parameters
            params.extend(['%s %s' % (cmp.prefix, p) for p in cmp.param_names])

        # add tellurics
        if self._tellurics is not None:
            params.extend(['%s %s' % (self._tellurics.prefix, p) for p in self._tellurics.param_names])

        # finished
        return params

    def fit_parameters(self) -> List[str]:
        """Get list of parameters fitted by this routine. Exclude fixed parameters.

        Returns:
            List of parameter names (including prefix) fitted by this routine.
        """

        # init
        params = []

        # loop components
        for cmp in self._cmps:
            # loop all parameters of this component
            for param_name in cmp.param_names:
                # check if this parameter is fixed
                if self._fixparams and cmp.name in self._fixparams and param_name in self._fixparams[cmp.name]:
                    continue

                # add parameter to list
                params.append('{} {}'.format(cmp.prefix, param_name))

        return params

    def columns(self) -> List[str]:
        """Get list of columns returned by __call__.

        The returned list shoud include the list from parameters().

        Returns:
            List of columns returned by __call__.
        """

        # call base and add columns Success
        return FilesRoutine.columns(self) + ['Success', 'RedChi2']

    @property
    def components(self) -> List[Component]:
        """Returns all components used in this fit.

        Returns:
            List of all components.
        """
        return self._cmps + ([] if self._tellurics is None else [self._tellurics])

    def _fix_params(self):
        """Fix parameters."""

        # loop components
        for cmp_name, cmp in self.objects['components'].items():
            # loop all parameters of this component
            for param_name in cmp.param_names:
                # do we have parameters to fix and is this one of them?
                if self._fixparams and cmp_name in self._fixparams and param_name in self._fixparams[cmp_name]:
                    self.log.info('Fixing "%s" of component "%s" to its initial value of %f.',
                                  param_name, cmp_name, cmp[param_name])
                    cmp.set(param_name, vary=False)
                else:
                    # otherwise make it a free parameter
                    cmp.set(param_name, vary=True)

    def __call__(self, filename: str) -> List[float]:
        """Start the fitting procedure on the given file.

        Args:
            filename: Name of file to fit.

        Returns:
            List of final values of parameters, ordered in the same way as the return value of parameters()
        """
        pass

    def _load_spectrum(self, filename: str) -> (Spectrum, np.ndarray):
        """Loads the given spectrum and its uncertainties and creates a mask.

        Args:
            filename: Name of file to load.

        Returns:
            Tuple of Spectrum and mask of valid pixels
        """

        # open file
        self.log.info("Loading file {0:s}.".format(filename))
        with FitsSpectrum(filename) as fs:
            # get spectrum
            self._spec = fs.spectrum

            # mask of good pixels
            self._valid = fs.good_pixels.astype(np.bool)

            # mask all NaNs
            self._valid &= ~np.isnan(self._spec.flux) & ~np.isnan(self._spec.wave)

            # add other masks
            for mask in self._masks:
                self._valid &= mask(fs.spectrum, filename=fs.filename)

            # create weight array
            self.log.info('Creating weights array...')
            self._weight = np.ones((len(self._spec)))
            if self._weights is not None:
                # loop all weights
                for w in self._weights:
                    # multiply weights array with new weights
                    self._weight *= w(self._spec, filename)

            # adjusting valid mask for weights
            self._valid &= ~np.isnan(self._weight)

            # less than 50% of pixels valid?
            if np.sum(self._valid) < self._min_valid_pixels * len(self._valid):
                self.log.warning('Less then %d percent of pixels valid, skipping...', self._min_valid_pixels * 100)
                return False

            # initialize multiplicative polynomial with ones
            self._mult_poly = Legendre(self._spec, self._poly_degree)

            # success
            return True

    def _callback(self, params: Parameters, iter: int, resid: np.ndarray, *args, **kws):
        """Callback function that is called at each iteration of the optimization.

        Args:
            params: Current parameter values.
            iter: Current iteration.
            resid: Current residual array
        """
        # init
        messages = []

        # loop components
        for cmp in self._cmps + ([self._tellurics] if self._tellurics else []):
            values = []
            for name in cmp.param_names:
                # get param
                param = params[cmp.prefix + name]

                # de-normalize?
                p = cmp.parameters[name]
                val, _ = cmp.denorm_param(name, param.value) if cmp.normalize else (param.value, None)

                # show
                if param.vary:
                    values.append('%s=%.3f' % (name, val))

            # print it
            if len(values) > 0:
                messages.append('%s(%s)' % (cmp.prefix, ', '.join(values)))

        # calculate chi2
        sumsq = np.sum(((self._spec.flux[self._valid] - self._model.flux[self._valid])
                        * self._weight[self._valid])**2)
        chi2 = np.sum(((self._spec.flux[self._valid] - self._model.flux[self._valid])**2
                       / self._model.flux[self._valid]))
        messages += ['sqsum=%.5g' % sumsq, 'chi2=%.5g' % chi2]

        # plot spectrum
        if self._iterations_pdf is not None:
            self._plot_spectrum(params, iter)

        # log it
        self.log.info('(%3d) %s' % (iter, ', '.join(messages)))

    def _plot_spectrum(self, params: Parameters, iter: int):
        """Plot the current iteration.

        Args:
            params: Current parameter values.
            iter: Current iteration.
        """

        # build text
        text = 'Iteration %d\n===============\n' % iter

        # loop components and add current values
        for cmp in self._cmps + ([self._tellurics] if self._tellurics else []):
            text += '\n%s:\n' % cmp.prefix
            for name in cmp.param_names:
                param = params[cmp.prefix + name]
                text += '%10s = %.2f\n' % (name, param.value)

        # calculate chi2
        chi2 = np.sum(np.power(self._spec.flux[self._valid] - self._model.flux[self._valid], 2)
                      * self._weight[self._valid])
        text += '\nchi2 = %g\n' % chi2

        # do the plot
        fig = plot_spectrum(self._spec, model=self._model, residuals=self._spec.flux - self._model.flux,
                            valid=self._valid, title='Iteration %d' % iter, text=text)

        # store and close it
        self._iterations_pdf.savefig(fig)
        plt.close()

    def _fit_func(self, params: Parameters) -> np.ndarray:
        """Fit function for LM optimization

        Args:
            params: Parameters to evaluate.

        Returns:
            Weights scaled residuals between spec and model.
        """


        try:
            # evaluate params
            self._model = self._get_model(params)

            # calc residuals
            return (self._model.flux[self._valid] - self._spec.flux[self._valid]) * self._weight[self._valid]

        except (KeyError, pd.core.indexing.IndexingError, ValueError):
            # could not interpolate
            self.log.exception('Could not interpolate model.')
            self._model = Spectrum(spec=self._spec)
            self._model.flux[:] = 0
            return np.ones((len(self._spec.flux[self._valid]))) * 1e100

    def _get_model(self, params: Parameters) -> Spectrum:
        """Returns the model for the given parameters.

        Args:
            params: Parameters to evaluate.
            spec: Spectrum to fit.

        Returns:
            Model for given parameters.

        Raises:
            KeyError: Forwarded from grid/interpolator access.
            pd.core.indexing.IndexingError: Forwarded from grid/interpolator access.
        """

        # parse params
        for cmp in self.components:
            cmp.parse_params(params)

        # get tellurics
        tell = None
        if self._tellurics is not None:
            # evaluate component
            tell = self._tellurics()

            # resample to same wavelength grid as spec
            tell.mode(self._spec.wave_mode)
            tell = tell.resample(spec=self._spec)

        # get models for all components
        models = []
        for cmp in self._cmps:
            # evaluate model
            m = cmp()

            # all invalid?
            if len(m.flux) == len(m.flux[np.isnan(m.flux)]):
                raise ValueError('All values NaN.')

            # resample to same wavelength grid as spec
            m.mode(self._spec.wave_mode)
            m = m.resample(spec=self._spec)

            # multiply tellurics, if necessary
            if tell is not None:
                m.flux *= tell.flux

            # append to list
            models.append(m)

        # weight components
        if len(self._cmps) == 1:
            # get points that are valid in both model and spectrum
            v = self._valid & ~np.isnan(models[0].flux)

            # if we only have one component, it's easy :-)
            #a = models[0].flux[v] * self._mult_poly[v]
            a = models[0].flux[v] * self._mult_poly.values[v]
            b = self._spec.flux[v] * self._weight[v]
            self._cmps[0].weight = (a * b).sum() / (a * a).sum()
        else:
            self._fit_component_weights(models)

        # add all models together weighted
        model = models[0]
        model.flux *= self._cmps[0].weight
        for i in range(1, len(self._cmps)):
            model.flux += models[i].flux * self._cmps[i].weight

        # multiplicative poly
        cont = self._mult_poly(model, self._valid)
        cont_mean = self._mult_poly.mean

        # multiply weights with mean
        for c in self._cmps:
            c.weight /= cont_mean

        # multiply continuum
        model.flux *= cont
        model.flux *= cont_mean

        # return model
        return model

    def _calculate_mult_poly(self, model):
        # create Legendre fitter
        # obviously the Legendre module automatically maps the x values to the range -1..1
        leg = np.polynomial.Legendre.fit(self._spec.wave[self._valid],
                                         self._spec.flux[self._valid] / model.flux[self._valid],
                                         deg=self._poly_degree)

        # return new polynomial
        return leg(self._spec.wave)

    def _fit_component_weights(self, models: List[Spectrum]):
        """In case we got more than one model, we need to weight them.

        Args:
            models: List of models.
        """

        # get first model
        m0 = models[0]

        # get valid points
        valid = np.ones((len(m0)), dtype=np.bool)
        for m in models:
            valid &= ~np.isnan(m.flux)

        # create matrix with models
        mat = np.empty((len(m0[valid]), len(models)))
        for k in range(len(models)):
            mat[:, k] = models[k].flux[valid]

        # fit
        coeffs = scipy.linalg.lstsq(mat, self._spec.flux[valid])[0]

        # set weights
        for i, cmp in enumerate(self._cmps):
            cmp.weight = coeffs[i]

    def _write_results_to_file(self, filename: str, result: MinimizerResult, best_fit: Spectrum, stats: dict):
        """Writes results of fit back to file.

        Args:
            filename: Name of file to write results into.
            result: Result from optimization.
            best_fit: Best fit model.
            stats: Fit statistics.
        """

        # Write fits results back to file
        self.log.info("Writing results to file.")
        with FitsSpectrum(filename, 'rw') as fs:
            # stats
            res = fs.results('SPEXXY')
            for x in stats:
                res[x] = stats[x]

            # loop all components
            for cmp in self._cmps:
                # write results
                cmp.write_results_to_file(fs)

            # tellurics
            if self._tellurics is not None:
                # molecular abundances
                self._tellurics.write_results_to_file(fs)

            # weights
            weights = fs.results("WEIGHTS")
            for cmp in self._cmps:
                weights[cmp.prefix] = cmp.weight

            # write spectra best fit, good pixels mask, residuals and
            # multiplicative polynomial
            if best_fit is not None:
                fs.best_fit = best_fit
                fs.residuals = self._spec.flux - best_fit.flux
            fs.good_pixels = self._valid
            fs.mult_poly = self._mult_poly.values

            # loop all components again to add spectra
            for cmp in self._cmps:
                # get spectrum
                tmp = cmp()
                tmp.mode(self._spec.wave_mode)
                tmp = tmp.resample(spec=self._spec)
                cmpspec = SpectrumFitsHDU(spec=tmp, primary=False)

                # set it
                fs['CMP_' + cmp.name] = cmpspec

            # tellurics spectrum
            if self._tellurics is not None:
                tmp = self._tellurics()
                tmp.mode(self._spec.wave_mode)
                tmp = tmp.resample(spec=self._spec)
                tell = SpectrumFitsHDU(spec=tmp, primary=False)

                # set it
                fs['TELLURICS'] = tell

            # covariance
            if hasattr(result, 'covar'):
                fs.covar = result.covar


__all__ = ['BaseParamsFit']
