import numpy as np
from typing import List, Dict

from spexxy.component import Component, TelluricsComponent
from .base import FilesRoutine


class MultiMain(FilesRoutine):
    """MultiRun iterates over a given set of main routines and runs them sequentially."""

    def __init__(self, routines: List = None, iterations: int = None, max_iterations: int = 8,
                 threshold: Dict[str, float] = None, poly_degree: int = 40, damped: bool = True,
                 factors: list = (0.7, 0.3), *args, **kwargs):
        """Initialize a new MultiMain object

        Args:
            routines: List of main routines to run.
            iterations: Number of iterations for the whole cycle.
            max_iterations: If set to a value >=2, the fit runs until it converges or until the number of iterations
                            reaches max_iterations.
            threshold: Dictionary that contains the absolute values for each fit parameter below which the fit is
                       considered as converged.
            poly_degree: Degree of Legendre polynomial used for the continuum fit.
            damped: If True the fit is repeated with a damping factor in case the fit does not converge.
            factors: List of damping factors used if the fit does not converge.
        """
        FilesRoutine.__init__(self, *args, **kwargs)

        # remember variables
        self._iterations = iterations
        self._max_iterations = max_iterations
        self._poly_degree = poly_degree
        self._threshold = threshold
        self._damped = damped
        self._factors = sorted(factors, reverse=True)

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

        The returned list should include the list from parameters().

        Returns:
            List of columns returned by __call__.
        """

        # call base and add columns Iterations, Success and Convergence
        if self._max_iterations is not None and self._damped:
            return FilesRoutine.columns(self) + ['Iterations', 'Success', 'Convergence', 'Damping Factor']
        elif self._max_iterations is not None:
            return FilesRoutine.columns(self) + ['Iterations', 'Success', 'Convergence']

        return FilesRoutine.columns(self) + ['Iterations', 'Success']

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
        if self._iterations is None and self._max_iterations > 1:
            # initialize thresholds
            tmp = {}
            for cmp_name, cmp in self.objects['components'].items():
                # loop over all parameters of this component
                for param_name in cmp.param_names:
                    if param_name.lower() == 'teff':
                        tmp['{} {}'.format(cmp.prefix, param_name)] = 25.
                    elif param_name.lower() == 'v' or param_name.lower() == 'sig':
                        tmp['{} {}'.format(cmp.prefix, param_name)] = 1.
                    else:
                        tmp['{} {}'.format(cmp.prefix, param_name)] = 0.05

            if self._threshold is not None:
                for cmp_name, values in self._threshold.items():
                    for param, threshold in values.items():
                        tmp['{} {}'.format(cmp_name, param)] = np.float(threshold)

            self._threshold = tmp

            maxiter = self._max_iterations
        else:
            if self._iterations is None and (self._max_iterations == 1 or self._max_iterations is None):
                maxiter = 1
            else:
                maxiter = self._iterations

            self._max_iterations = None

        # loop iterations
        for it in range(maxiter):
            self.objects['init_iter'] = {}
            for cmp_name, cmp in self.objects['components'].items():
                if isinstance(cmp, TelluricsComponent):
                    continue

                self.objects['init_iter'][cmp_name] = Component(name=cmp_name)
                for param_name in cmp.param_names:
                    self.objects['init_iter'][cmp_name].set(name=param_name, value=cmp[param_name])

            # loop main routines
            for routine in self._routines:
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

                    if self._damped:
                        res.append(1.)

                    return res
                elif it == self._max_iterations - 1 and not self._damped:
                    # fit is successful if each iteration was a success
                    success = np.all(success)

                    # fit did not converge
                    converged = False

                    # convert results dict into results list
                    res = []
                    for p in parameters:
                        res.extend(results[p])

                    # add iteration, success and convergence to results
                    res.append(it + 1)
                    res.append(success)
                    res.append(converged)

                    return res

        # if fit did not converge try again with damping factor
        if self._max_iterations is not None and self._damped:
            for p in self._threshold:
                self._threshold[p] /= 3

            iterations = self._max_iterations
            for damping_factor in self._factors:
                # reset parameters to initial values
                for cmp_name, cmp in self.objects['components'].items():
                    cmp.init(filename)

                results = {p: None for p in parameters}
                success = []
                results_total = {p: [] for p in fit_params}

                # loop over fit parameters
                for it in range(3 * maxiter):
                    self.objects['init_iter'] = {}
                    for cmp_name, cmp in self.objects['components'].items():
                        if isinstance(cmp, TelluricsComponent):
                            continue

                        self.objects['init_iter'][cmp_name] = Component(name=cmp_name)
                        for param_name in cmp.param_names:
                            self.objects['init_iter'][cmp_name].set(name=param_name, value=cmp[param_name])

                    # loop main routines
                    for routine in self._routines:
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

                        # calculate damped result
                        for p in params:
                            if p not in routine.fit_parameters():
                                continue

                            for cmp_name, cmp in self.objects['components'].items():
                                if not p.startswith(cmp.prefix):
                                    continue

                                name = p[len(cmp.prefix)+1:]

                                cmp[name] = (1 - damping_factor) * self.objects['init_iter'][cmp_name][
                                    name] + damping_factor * results[p][0]
                                results[p][0] = (1 - damping_factor) * self.objects['init_iter'][cmp_name][
                                    name] + damping_factor * results[p][0]

                        # was iteration a success?
                        success.append(res[-1])

                    # save results of all steps
                    for p in fit_params:
                        results_total[p].append(results[p][0])

                    # run at least for max_iterations // 2 iterations
                    if it < self._max_iterations // 2:
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
                        res.append(iterations + it + 1)
                        res.append(success)
                        res.append(converged)
                        res.append(damping_factor)

                        return res

                iterations += 3 * self._max_iterations

            # fit is successful if each iteration was a success
            success = np.all(success)

            # fit did not converge
            converged = False

            # convert results dict into results list
            res = []
            for p in parameters:
                res.extend(results[p])

            # add iteration, success and convergence to results
            res.append(iterations)
            res.append(success)
            res.append(converged)
            res.append(self._factors[-1])

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
            # test for convergence
            c.append(abs(res[-1] - res[-2]) <= self._threshold[param])

        # return True if all parameters satisfy their convergence criterion
        return np.all(c)


__all__ = ['MultiMain']
