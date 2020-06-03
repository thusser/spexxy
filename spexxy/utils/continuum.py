from typing import List, Tuple

import scipy.special
import scipy.linalg
import scipy.interpolate
import numpy as np
from numpy import polynomial


class Continuum(object):
    """Base class for all continuum fitting classes."""

    def __init__(self, poly: str = "polynomial", poly_degree: int = 7):
        """Initialize a new Continuum

        Args:
            poly: Type of polynomial.
            poly_degree: Degree of polynomial.
        """

        # polynomial function or spline?
        if poly == 'spline':
            self._poly_func = None
        else:
            self._poly_func = {
                'polynomial': polynomial.Polynomial,
                'chebyshev': polynomial.Chebyshev,
                'legendre': polynomial.Legendre
            }[poly]

        # polynomial degree
        self._degree = poly_degree

    def _fit_poly(self, xin: np.ndarray, yin: np.ndarray, xout: np.ndarray = None) -> np.ndarray:
        """Fit a polynomial to the data and return it.

        Args:
            xin: Input x array for continuum to fit.
            yin: Input y array for continuum to fit.
            xout: Output x array to evaluate polynomial on, if None is given, xin is used.

        Returns:
            Array containing calculated polynomial.

        """

        # no xout?
        if xout is None:
            xout = xin

        # do actual fitting
        if self._poly_func is not None:
            # fit with a polynomial
            result = self._poly_func.fit(xin, yin, deg=self._degree)

            # evaluate polynomial at xout
            return result(xout)

        else:
            # fit with a spline
            ip = scipy.interpolate.interp1d(xin, yin, kind='cubic', bounds_error=False, fill_value='extrapolate')

            # evaluate spline at xout
            return ip(xout)

    def __call__(self, x: np.ndarray, y: np.ndarray, valid: np.ndarray = None) -> np.ndarray:
        """Calculate the continuum.

        Args:
            x: x array for continuum to fit.
            y: y array for continuum to fit.
            valid: Valid pixels mask.

        Returns:
            Array containing calculated continuum.
        """

        # no valid array?
        if valid is None:
            valid = np.ones((x.shape), dtype=np.bool)

        # fit polynomial
        return self._fit_poly(x[valid], y[valid], x)

    def remove(self, x: np.ndarray, y: np.ndarray, valid: np.ndarray = None) -> np.ndarray:
        """Remove continuum from a given array.

        Args:
            x: x array for continuum to fit.
            y: y array for continuum to fit.
            valid: Valid pixels mask.

        Returns:
            Array containing continuum corrected y array.
        """

        # calculate continuum
        cont = self(x, y, valid)

        # do we have negative values in the continuum?
        min_cont = np.min(cont)
        if min_cont < 1:
            # correct y by shifting it into positive range first
            return (y + min_cont + 1) / (cont + min_cont + 1)
        else:
            # otherwise remove continuum directly
            return y / cont


class MaximumBin(Continuum):
    """Derive continuum as spline to all points that are within a given fraction of largest points in a given number
    of bins."""

    def __init__(self, frac: float = 0.15, sbin: int = 100, *args, **kwargs):
        """Initialize a new MaximumBin continuum.

        Args:
            frac: Fraction of largest points in each bin to use for fitting the continuum.
            sbin: Number of bins.
        """
        Continuum.__init__(self, *args, **kwargs)

        # remember values
        self.frac = frac
        self.sbin = sbin

    def __call__(self, x: np.ndarray, y: np.ndarray, valid: np.ndarray = None) -> np.ndarray:
        """Calculate the continuum.

        Args:
            x: x array for continuum to fit.
            y: y array for continuum to fit.
            valid: Valid pixels mask.

        Returns:
            Array containing calculated continuum.
        """

        # no valid array?
        if valid is None:
            valid = np.ones((x.shape), dtype=np.bool)

        # find continuum for every of the sbin bins
        bins_x = np.zeros((self.sbin))
        bins_y = np.zeros((self.sbin))
        w1 = 0
        w2 = float(len(x)) / self.sbin
        for i in range(self.sbin):
            # get range
            bindata = y[int(w1):int(w2)].copy()

            # sort it
            bindata.sort()

            # calculate median
            bins_y[i] = np.median(bindata[int(-self.frac * len(bindata)):-1])

            # set wavelength
            bins_x[i] = np.mean(x[int(w1):int(w2)])

            # reset ranges
            w1 = w2
            w2 += float(len(x)) / self.sbin

            # check for last bin
            if i == self.sbin - 1:
                w2 = len(x)

        # fit polynomial
        return self._fit_poly(bins_x, bins_y, x)


class SigmaClipping(Continuum):
    """Derives continuum from kappa-sigma clipping with polynomial."""

    def __init__(self, kappa_low: float = 3, kappa_high: float = 3, maxiter: int = 20, *args, **kwargs):
        """Initialize a new SigmaClipping continuum

        Args:
            kappa_low: Lower kappa.
            kappa_high: Upper kappa.
            maxiter: Maximum number of iterations.
        """
        Continuum.__init__(self, *args, **kwargs)

        # store values
        self.kappa_low = kappa_low
        self.kappa_high = kappa_high
        self.maxiter = maxiter

    def __call__(self, x: np.ndarray, y: np.ndarray, valid: np.ndarray = None) -> np.ndarray:
        """Calculate the continuum.

        Args:
            x: x array for continuum to fit.
            y: y array for continuum to fit.
            valid: Valid pixels mask.

        Returns:
            Array containing calculated continuum.
        """

        # no valid array?
        if valid is None:
            valid = np.ones((x.shape), dtype=np.bool)

        # copy x and y
        xx = x[valid].copy()
        yy = y[valid].copy()

        # fit
        poly = self._fit_poly(xx, yy)

        # iterate
        i = 0
        while i < self.maxiter:
            # calc residuals and do sigma clipping
            residuals = yy - poly

            # calculate current median and sigma
            median = np.nanmedian(residuals)
            sigma = np.nanstd(residuals)

            # create mask
            mask = (residuals < (median - self.kappa_low * sigma)) | (residuals > (median + self.kappa_high * sigma))

            # no new pixels masked?
            if np.sum(mask) == 0:
                break

            # apply mask
            xx = xx[~mask]
            yy = yy[~mask]

            # fit new poly
            poly = self._fit_poly(xx, yy)

        # return continuum
        return self._fit_poly(xx, yy, xout=x)


class Regions(Continuum):
    """Derives continuum from given regions."""

    def __init__(self, regions: List[Tuple[float, float]], average: bool = True, *args, **kwargs):
        """Initializes new Regions continuum

        Args:
            regions: List of tuples with start end end x values for regions.
            average: Use average wave/flux in regions instead of all pixels.
        """

        Continuum.__init__(self, *args, **kwargs)
        self.regions = regions
        self.average = average

    def __call__(self, x: np.ndarray, y: np.ndarray, valid: np.ndarray = None) -> np.ndarray:
        """Calculate the continuum.

        Args:
            x: x array for continuum to fit.
            y: y array for continuum to fit.
            valid: Valid pixels mask.

        Returns:
            Array containing calculated continuum.
        """

        # no valid array?
        if valid is None:
            valid = np.ones(x.shape, dtype=np.bool)

        # evaluate regions
        xx = []
        yy = []
        for r in self.regions:
            # create mask
            mask = valid & (x >= r[0]) & (x <= r[1])

            # add (mean) of points within mask
            if self.average:
                xx.append(np.mean(x[mask]))
                yy.append(np.mean(y[mask]))
            else:
                xx.extend(x[mask])
                yy.extend(y[mask])

        # fit continuum
        return self._fit_poly(xx, yy, xout=x)


__all__ = ['Continuum', 'MaximumBin', 'Regions', 'SigmaClipping']
