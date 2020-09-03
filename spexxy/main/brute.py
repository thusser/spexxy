import numpy as np
from typing import List

from spexxy.component import GridComponent
from .baseparamsfit import BaseParamsFit


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
        for iter, params in enumerate(all_params, 1):
            # make parameters
            p = cmp.make_params(**dict(zip(param_names, params)))

            # get model
            self._model = self._get_model(p)

            # chi2
            chi2 = np.sum((self._spec.flux[self._valid] - self._model.flux[self._valid])**2 / self._model.flux[self._valid])

            # found a new best result?
            if best is None or chi2 < best[0]:
                # set it
                best = (chi2, params)

                # log new best result
                self._callback(p, iter, None)

        # return results
        return list(best[1]) + [True, best[0]]


__all__ = ['BruteFit']
