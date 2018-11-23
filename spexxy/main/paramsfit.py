import sys
import lmfit
import numpy as np
import scipy.linalg
from lmfit import Parameters
from lmfit.minimizer import MinimizerResult
from typing import List

from spexxy.data import FitsSpectrum, Spectrum
from spexxy.component import Component
from spexxy.mask.mask import Mask
from spexxy.data import SpectrumFitsHDU
from spexxy.object import spexxyObject


class ParamsFit(spexxyObject):
    """ParamsFit is a fitting routine for spexxy that uses a Levenberg-Marquardt optimization to fit a set of
    model spectra (components) to a given spectrum.
    """

    def __init__(self, components: List[Component] = None, tellurics: Component = None, masks: List[Mask] = None,
                 fixparams: List[str] = None, poly_degree: int = 40,
                 maxfev: int = 500, ftol: float = 1.49012e-08, xtol: float = 1.49012e-08,
                 factor: float = 100.0, epsfcn: float = 1e-7, *args, **kwargs):
        """Initialize a new ParamsFit object

        Args:
            components: List of components (or descriptions to create them) to fit to the spectrum
            tellurics: Tellurics component to add to the fit.
            masks:  List of mask components to use for masking the spectrum.
            fixparams: List of names of parameters to fix during the fit.
            poly_degree: Number of coefficients for the multiplicative polynomial.
            maxfev: The maximum number of calls to the function (see scipy documentation).
            ftol: Relative error desired in the sum of squares (see scipy documentation).
            xtol: Relative error desired in the approximate solution (see scipy documentation).
            factor: A parameter determining the initial step bound (factor * || diag * x||) (see scipy documentation).
            epsfcn: A variable used in determining a suitable step length for the forward- difference approximation of
                the Jacobian (see scipy documentation)
        """
        spexxyObject.__init__(self, *args, **kwargs)

        # remember variables
        self._max_fev = maxfev
        self._ftol = ftol
        self._xtol = xtol
        self._factor = factor
        self._epsfcn = epsfcn
        self._poly_degree = poly_degree
        self._fixparams = fixparams

        # find components
        self._cmps = self.get_objects(components, Component, 'components')

        # find tellurics
        self._tellurics = self.get_objects(tellurics, Component, 'components', single=True)

        # masks
        self._masks = self.get_objects(masks, Mask, 'masks')

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

    @property
    def components(self) -> List[Component]:
        """Returns all components used in this fit.

        Returns:
            List of all components.
        """
        return self._cmps + ([] if self._tellurics is None else [self._tellurics])

    def __call__(self, filename: str) -> List[float]:
        """Start the fitting procedure on the given file.

        Args:
            filename: Name of file to fit.

        Returns:
            List of final values of parameters, ordered in the same way as the return value of parameters()
        """

        # fix any parameters?
        for cmp_name, cmp in self.objects['components'].items():
            # loop all parameters of this component
            for param_name in cmp.param_names:
                # do we have parameters to fix and is this one of them?
                if self._fixparams and cmp_name in self._fixparams and param_name in self._fixparams[cmp_name]:
                    self.log.info('Fixing "%s" of component "%s" to its initial value of %f.',
                                  param_name, cmp_name, cmp[param_name])
                    cmp.set(param_name, vary=False)

        # Load spectrum
        spec, sigma, valid = self._load_spectrum(filename)

        # init stats
        stats = {'success': False}

        # initialize multiplicative polynomial with ones
        mult_poly = np.ones((len(spec)))

        # get parameters
        params = Parameters()
        for cmp in self.components:
            params += cmp.make_params()

        # start minimization
        self.log.info('Starting fit...')
        minimizer = lmfit.Minimizer(self._fit_func, params,
                                    fcn_kws={'spec': spec, 'valid': valid, 'sigma': sigma, 'mult_poly': mult_poly},
                                    iter_cb=self._callback, maxfev=self._max_fev,
                                    xtol=self._xtol, ftol=self._ftol, epsfcn=self._epsfcn,
                                    factor=self._factor, nan_policy='raise')
        result = minimizer.leastsq()
        self.log.info('Finished fit.')

        # get best fit
        best_fit = self._model(result.params, spec=spec, sigma=sigma, valid=valid, mult_poly=mult_poly)

        # estimate SNR
        snr = None if best_fit is None else spec.estimate_snr(best_fit)
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
        self._write_results_to_file(filename, spec, valid, result, best_fit, mult_poly)

        # build list of results and return them
        results = []
        for cmp in self._cmps:
            for n in cmp.param_names:
                p = '%s%s' % (cmp.prefix, n)
                results.extend([result.params[p].value, result.params[p].stderr])
        if self._tellurics is not None:
            for n in self._tellurics.param_names:
                p = '%s%s' % (self._tellurics.prefix, n)
                results.extend([result.params[p].value, result.params[p].stderr])
        return results

    def _load_spectrum(self, filename: str) -> (Spectrum, np.ndarray, np.ndarray):
        """Loads the given spectrum and its uncertainties and creates a mask.

        Args:
            filename: Name of file to load.

        Returns:
            Tuple of Spectrum, sigma array and mask of valid pixels
        """

        # open file
        self.log.info("Loading file {0:s}.".format(filename))
        with FitsSpectrum(filename) as fs:
            # get spectrum
            spec = fs.spectrum

            # sigma values for spectrum
            sigma = fs.sigma

            # mask of good pixels
            valid = fs.good_pixels.astype(np.bool)

            # mask all NaNs
            valid &= ~np.isnan(spec.flux) & ~np.isnan(spec.wave)

            # and those in sigma
            valid &= ~np.isnan(sigma)

            # add other masks
            for mask in self._masks:
                valid &= mask(fs.spectrum, filename=fs.filename)

        # return them
        return spec, sigma, valid

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
                param = params[cmp.prefix + name]
                if param.vary:
                    values.append('%s=%.2f' % (name, param.value))
            if len(values) > 0:
                messages.append('%s(%s)' % (cmp.prefix, ', '.join(values)))

        # log it
        self.log.info('(%3d) %s' % (iter, ', '.join(messages)))

    def _fit_func(self, params: Parameters, spec: Spectrum, sigma: np.ndarray, valid: np.ndarray,
                  mult_poly: np.ndarray) -> np.ndarray:
        """Fit function for LM optimization

        Args:
            params: Parameters to evaluate.
            spec: Spectrum to fit.
            sigma: Uncertainties for spec.
            valid: Valid pixel mask for spec.
            mult_poly: Multiplicative polynomial.

        Returns:
            Sigma scaled residuals between spec and model.
        """

        # evaluate params
        try:
            model = self._model(params, spec, sigma, valid, mult_poly)
        except KeyError:
            return np.ones((len(spec[valid]))) * sys.float_info.max

        # calc residuals
        res = (model.flux - spec.flux) / sigma

        # replace nans by zeros
        res[np.isnan(res)] = 0

        # return residuals
        return res

    def _model(self, params: Parameters, spec: Spectrum, sigma: np.ndarray, valid: np.ndarray,
               mult_poly: np.ndarray) -> Spectrum:
        """Returns the model for the given parameters.

        Args:
            params: Parameters to evaluate.
            spec: Spectrum to fit.
            sigma: Uncertainties for spec.
            valid: Valid pixel mask for spec.
            mult_poly: Multiplicative polynomial.

        Returns:
            Model for given parameters.

        Raises:
            KeyError: Forwarded from grid/interpolator access.
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
            tell.mode(spec.wave_mode)
            tell = tell.resample(spec=spec)

        # get models for all components
        models = []
        for cmp in self._cmps:
            # evaluate model
            m = cmp()

            # resample to same wavelength grid as spec
            m.mode(spec.wave_mode)
            m = m.resample(spec=spec)

            # multiply tellurics, if necessary
            if tell is not None:
                m.flux *= tell.flux

            # append to list
            models.append(m)

        # weight components
        if len(self._cmps) == 1:
            # get points that are valid in both model and spectrum
            v = valid & ~np.isnan(models[0].flux)

            # if we only have one component, it's easy :-)
            a = models[0].flux[v] * mult_poly[v]
            b = spec.flux[v] / sigma[v]
            self._cmps[0].weight = (a * b).sum() / (a * a).sum()
        else:
            self._fit_component_weights(models)

        # add all models together weighted
        model = models[0]
        model.flux *= self._cmps[0].weight
        for i in range(1, len(self._cmps)):
            model.flux += models[i].flux * self._cmps[i].weight

        # calculate new multiplicative poly
        leg = np.polynomial.Legendre.fit(spec.wave[valid], spec.flux[valid] / model.flux[valid],
                                         deg=self._poly_degree)
        mult_poly = leg(spec.wave)

        # multiply continuum
        model.flux *= mult_poly

        # divide continuum by mean
        cont_mean = np.mean(mult_poly)
        mult_poly /= cont_mean

        # multiply weights with mean
        for cmp in self._cmps:
            cmp.weight /= cont_mean

        # return model
        return model

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
        coeffs = scipy.linalg.lstsq(mat, spec.flux[valid])[0]

        # set weights
        for i, cmp in enumerate(self._cmps):
            cmp.weight = coeffs[i]

    def _write_results_to_file(self, filename: str, spec: Spectrum, valid: np.ndarray, result: MinimizerResult,
                               best_fit: Spectrum, mult_poly: np.ndarray):
        """Writes results of fit back to file.

        Args:
            filename: Name of file to write results into.
            spec: Spectrum that has been fitted.
            valid: Mask of valid pixels.
            result: Result from optimization.
            best_fit: Best fit model.
            mult_poly: Multiplicative polynomial.
        """

        # Write fits results back to file
        self.log.info("Writing results to file.")
        with FitsSpectrum(filename, 'rw') as fs:
            # stats
            stats = fs.results('SPEXXY')
            for x in stats:
                stats[x] = stats[x]

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
                fs.residuals = spec.flux - best_fit.flux
            fs.good_pixels = valid
            fs.mult_poly = mult_poly

            # loop all components again to add spectra
            for cmp in self._cmps:
                # get spectrum
                tmp = cmp()
                tmp.mode(spec.wave_mode)
                tmp = tmp.resample(spec=spec)
                cmpspec = SpectrumFitsHDU(spec=tmp, primary=False)

                # set it
                fs['CMP_' + cmp.name] = cmpspec

            # tellurics spectrum
            if self._tellurics is not None:
                tmp = self._tellurics()
                tmp.mode(spec.wave_mode)
                tmp = tmp.resample(spec=spec)
                tell = SpectrumFitsHDU(spec=tmp, primary=False)

                # set it
                fs['TELLURICS'] = tell

            # covariance
            fs.covar = result.covar


__all__ = ['ParamsFit']
