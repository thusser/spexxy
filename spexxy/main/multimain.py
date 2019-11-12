import sys
import lmfit
import numpy as np
import scipy.linalg
from lmfit import Parameters
from lmfit.minimizer import MinimizerResult
from typing import List, Dict

from spexxy.data import FitsSpectrum, Spectrum
from spexxy.component import Component
from spexxy.mask import Mask
from spexxy.weight import Weight
from spexxy.data import SpectrumFitsHDU
from spexxy.object import spexxyObject
from .main import MainRoutine


class MultiMain(MainRoutine):
    """MultiRun iterates over a given set of main routines and runs them sequentially."""

    def __init__(self, routines: List = None, iterations: int = 1, max_iterations: int = None,
                 threshold: Dict[str, float] = None, poly_degree: int = 40, *args, **kwargs):
        """Initialize a new MultiRun object

        Args:
            routines: List of main routines to run.
            iterations: Number of iterations for the whole cycle.
            max_iterations: If set to a value >=2, the fit runs until it converges or until the number of iterations
                            reaches max_iterations.
            threshold: Dictionary that contains the absolute values for each fit parameter below which the fit is
                       considered as converged.
            poly_degree: Degree of Legendre polynomial used for the continuum fit.
        """
        spexxyObject.__init__(self, *args, **kwargs)

        # remember variables
        self._iterations = iterations
        self._max_iterations = max_iterations
        self._poly_degree = poly_degree
        self._threshold = threshold

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
            parameters += routine.parameters()

        # make unique and sort
        return sorted(list(set(parameters)))

    def columns(self) -> List[str]:
        """Get list of columns returned by __call__.

        The returned list shoud include the list from parameters().

        Returns:
            List of columns returned by __call__.
        """

        # call base and add columns Iterations, Success and Convergence
        if self._max_iterations is not None:
            return MainRoutine.columns(self) + ['Iterations', 'Success', 'Convergence', 'Damping']

        return MainRoutine.columns(self) + ['Iterations', 'Success']

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

        success = []

        # list with all fit parameters
        fit_params = []
        for routine in self._routines:
            fit_params.extend(routine.fit_parameters())

        fit_params = list(set(fit_params))

        # dictionary that contains the fit results of all iteration steps, used for convergence test
        results_total = {p: [] for p in fit_params}

        # if routine checks for convergence set the threshold now
        if self._max_iterations is not None:
            # if threshold is not given set it now
            if self._threshold is None:
                self._threshold = {}
                # loop over components
                for cmp_name, cmp in self.objects['components'].items():
                    # loop over all parameters of this component
                    for param_name in cmp.param_names:
                        # loop over all fit parameters
                        for p in fit_params:
                            # if component parameter is a fit parameter set its threshold
                            if param_name in p:
                                # set default thresholds for each parameter
                                if param_name.lower() == 'teff':
                                    self._threshold[p] = 25
                                elif param_name.lower() == 'feh':
                                    self._threshold[p] = 0.05
                                elif param_name.lower() == 'logg':
                                    self._threshold[p] = 0.05
                                elif param_name.lower() == 'alpha':
                                    self._threshold[p] = 0.05
                                elif param_name == 'v':
                                    self._threshold[p] = 1.
                                elif param_name == 'sig':
                                    self._threshold[p] = 1.
            else:
                # if threshold is given, check if dictionary keys correspond to parameter names, if not add the cmp prefix
                tmp = {}
                # loop over components
                for cmp_name in self.objects['components']:
                    for p in self._threshold:
                        # adjust keys in dictionary if necessary
                        if p in fit_params:
                            continue
                        elif cmp_name + ' ' + p in fit_params:
                            tmp[cmp_name + ' ' + p] = self._threshold[p]
                        else:
                            raise IndexError

                self._threshold = tmp

            maxiter = self._max_iterations
        else:
            maxiter = self._iterations

        # save initial values of components
        init_values = {}
        for cmp_name, cmp in self.objects['components'].items():
            for param_name in cmp.param_names:
                init_values['{} {}'.format(cmp.prefix, param_name)] = cmp[param_name]

        # loop iterations
        for it in range(maxiter):
            # loop main routines
            for routine in self._routines:
                # forward iteration step and initial values to routine
                routine.step = it + 1
                routine.init_values = init_values

                # set poly degree for this routine
                routine._poly_degree = self._poly_degree

                # get parameters for this routine
                params = routine.parameters()

                # run routine
                res = routine(filename)

                # store results
                for i, p in enumerate(params):
                    # if parameter is a fit parameter in another iteration step don't overwrite previous result
                    # otherwise the error will be set to zero
                    if p in fit_params and p not in routine.fit_parameters():
                        # initialize dictionary
                        if results[p] is None:
                            results[p] = [res[i * 2], res[i * 2 + 1]]

                        continue

                    # copy both results and errors!
                    results[p] = [res[i * 2], res[i * 2 + 1]]

                # was iteration a success?
                success.append(res[-1])

            # update initial values by most recent fit results
            init_values = {p: results[p][0] for p in init_values}

            # check for convergence?
            if self._max_iterations is not None:
                # save results of all steps
                for p in fit_params:
                    results_total[p].append(results[p][0])

                # run at least for 2 iterations
                if it == 0:
                    continue

                # check for convergence
                if self.convergence(results_total):
                    # fit is successful if each iteration was a success
                    success = np.all(success)

                    # fit converged
                    converged = True

                    # convert results dict into results list
                    res = []
                    for p in parameters:
                        res.extend(results[p])

                    # add iteration, success and convergence to results
                    res.append(it + 1)
                    res.append(success)
                    res.append(converged)
                    res.append(1.)

                    return res

        # if fit did not converge try again with damping factor
        if self._max_iterations is not None:
            for p in self._threshold:
                self._threshold[p] /= 2

            for damping in [0.9, 0.7]:
                # reset parameters to initial values
                for cmp_name, cmp in self.objects['components'].items():
                    cmp.init(filename)

                # save initial values of components
                init_values = {}
                for cmp_name, cmp in self.objects['components'].items():
                    for param_name in cmp.param_names:
                        init_values['{} {}'.format(cmp.prefix, param_name)] = cmp[param_name]

                results_damped = {p: None for p in parameters}
                results = {p: None for p in parameters}
                success = []
                results_total = {p: [] for p in fit_params}

                # loop over fit parameters
                for it in range(3 * maxiter):
                    for routine in self._routines:
                        # forward iteration step and initial values to routine
                        routine.step = it + 1
                        routine.init_values = init_values

                        # set poly degree for this routine
                        routine._poly_degree = self._poly_degree

                        # get parameters for this routine
                        params = routine.parameters()

                        # run routine
                        res = routine(filename)

                        # store results
                        for i, p in enumerate(params):
                            # if parameter is a fit parameter in another iteration step don't overwrite previous result
                            # otherwise the error will be set to zero
                            if p in fit_params and p not in routine.fit_parameters():
                                # initialize dictionary
                                if results[p] is None:
                                    results[p] = [res[i * 2], res[i * 2 + 1]]
                                    results_damped[p] = [res[i * 2], res[i * 2 + 1]]

                                continue

                            # copy both results and errors!
                            results[p] = [res[i * 2], res[i * 2 + 1]]
                            results_damped[p] = [res[i * 2], res[i * 2 + 1]]

                        # was iteration a success?
                        success.append(res[-1])

                        delta = {}
                        for p in routine.fit_parameters():
                            delta[p] = damping * (results[p][0] - init_values[p])

                        for p in routine.fit_parameters():
                            init_values[p] += delta[p]
                            results_damped[p][0] = init_values[p]

                        for cmp_name, cmp in self.objects['components'].items():
                            for param_name in cmp.param_names:
                                if '{} {}'.format(cmp.prefix, param_name) in routine.fit_parameters():
                                    cmp[param_name] = init_values['{} {}'.format(cmp.prefix, param_name)]

                    # update initial values by most recent fit results
                    init_values = {p: results_damped[p][0] for p in init_values}

                    # save results of all steps
                    for p in fit_params:
                        results_total[p].append(results[p][0])

                    # run at least for 2 iterations
                    if it == 0:
                        continue

                    # check for convergence
                    if self.convergence(results_total):
                        # fit is successful if each iteration was a success
                        success = np.all(success)

                        # fit converged
                        converged = True

                        # convert results dict into results list
                        res = []
                        for p in parameters:
                            res.extend(results[p])

                        # add iteration, success and convergence to results
                        res.append(it + 1)
                        res.append(success)
                        res.append(converged)
                        res.append(damping)

                        return res

            # fit is successful if each iteration was a success
            success = np.all(success)

            # fit did not converge
            converged = False

            # convert results dict into results list
            res = []
            for p in parameters:
                res.extend(results[p])

            # add iteration, success and convergence to results
            res.append(maxiter + 2 * 3 * maxiter)
            res.append(success)
            res.append(converged)
            res.append(damping)

            return res

        # fit is successful if each iteration was a success
        success = np.all(success)

        # convert results dict into results list
        res = []
        for p in parameters:
            res.extend(results[p])

        # add iteration and success to results
        res.append(self._iterations)
        res.append(success)

        return res

    def convergence(self, results):
        """Returns true if the fit satisfies the convergence criteria for all fit parameters."""

        c = []
        # loop over results
        for param, res in results.items():
            # is parameter fit parameter
            if param in self._threshold.keys():
                # test for convergence
                c.append(abs(res[-1] - res[-2]) <= self._threshold[param])

        # return True if all parameters satisfy their convergence criterion
        return np.all(c)


__all__ = ['MultiMain']
