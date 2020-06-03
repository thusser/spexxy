from typing import List, Tuple

import scipy.special
import scipy.linalg
import scipy.interpolate
import numpy as np
from numpy import polynomial


class Continuum(object):
    """Base class for all continuum fitting classes."""

    def __init__(self, poly: str = "polynomial", poly_degree: int = 7, spline_type: str = 'cubic'):
        """Initialize a new Continuum

        Args:
            poly: Type of polynomial ('polynomial', 'chebyshev', or 'legendre') or 'spline'.
            poly_degree: Degree of polynomial.
            spline_type: Type of spline ('quadratic', 'cubic')
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

        # store
        self._poly_degree = poly_degree
        self._spline_type = spline_type

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
            result = self._poly_func.fit(xin, yin, deg=self._poly_degree)

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


class PeakToPeak(Continuum):
    """Derives continuum from a ptp optimization.

    from "Spectroscopy of Binaries in Globular Clusters"
    by Benjamin Giesers
    https://ediss.uni-goettingen.de/handle/21.11130/00-1735-0000-0005-13B4-A
    """

    def __init__(self, model_x: np.ndarray, model_y: np.ndarray, min_bunch_size: int = 3, max_ptp: float = 0.001,
                 threshold: float = 0.9995, iterations: int = 4, bin_size: float = 100,
                 continuum_points: np.ndarray = None, to_mask: List[Tuple[float, float]] = None,
                 standard_mask: bool = True, *args, **kwargs):
        """Normalize spectrum with the help of a fitted model (the model could also be the spectrum itself!)


        Args:
            model_x: Array of model wavelength points
            model_y: Array of model flux points.
            min_bunch_size: The minimum number of coherent flux points within a continuum bunch.
            max_ptp: The maximum peak to peak deviation of flux points within a continuum bunch.
            threshold: Threshold to filter wrong (local minima) continuum bunches (comparison with neighbor bunches).
            iterations: Filter iterations to find wrong continuum bunches.
            bin_size: Wavelength bins to calculate (spline fit) interpolated continuum.
            continuum_points: Definitive mask of continuum points in spectrum (wavelength). Will be ignored if None.
            to_mask: Wavelength regions [min, max] to mask, where continuum will not be detected. If None standard set is taken.
            standard_mask: Should a standard mask for broad lines be applied?
        """
        Continuum.__init__(self, *args, **kwargs)

        # store
        self._min_bunch_size = min_bunch_size
        self._max_ptp = max_ptp
        self._threshold = threshold
        self._iterations = iterations
        self._bin_size = bin_size

        # mask tellurics and wings of deep absorptions
        if to_mask is None:
            to_mask = []
        if standard_mask is True:
            to_mask += [[4000, 4768], [4810, 4930], [5763, 5775], [5877, 5898],
                        [6266, 6277], [6515, 6640], [9333, 10000]]
        self._mask = np.ones(len(model_x), dtype=bool)
        for m in to_mask:
            self._mask &= (model_x < m[0]) | (model_x > m[1])

        # get continuum points and continuum in model spectrum
        self._cont = self._get_continuum_in_model(model_x, model_y)

        # if continuum_points is set, use them definitely
        if continuum_points is not None:
            self._cont |= continuum_points

    def __call__(self, x: np.ndarray, y: np.ndarray, valid: np.ndarray = None) -> np.ndarray:
        """Calculate the continuum.

        Args:
            x: x array for continuum to fit.
            y: y array for continuum to fit.
            valid: Valid pixels mask.

        Returns:
            Array containing calculated continuum.
        """

        # create continuum spectrum
        cont_x = x[self._cont & self._mask & ~np.isnan(y)]
        cont_y = y[self._cont & self._mask & ~np.isnan(y)]

        # bin continuum points in spectrum
        bin_wave = []
        bin_flux = []
        for ind in range(int((np.max(x) - np.min(x)) / self._bin_size) + 1):
            slc = (cont_x > np.min(x) + ind * self._bin_size) & (cont_x < np.min(x) + (ind + 1) * self._bin_size)
            if len(cont_x[slc]) > 2:
                bin_wave.append(np.mean(cont_x[slc]))
                bin_flux.append(np.median(cont_y[slc]))

        # deal with boundary conditions
        bin_wave.append(np.max(x))
        bin_flux.append(bin_flux[-1])
        bin_wave.insert(0, np.min(x))
        bin_flux.insert(0, bin_flux[0])

        # fit continuum
        return self._fit_poly(bin_wave, bin_flux, xout=x)

    def _get_continuum_in_model(self, model_x: np.ndarray, model_y: np.ndarray) -> np.ndarray:
        """Find continuum points in model spectrum

        Args:
            model_x: Array of model wavelength points
            model_y: Array of model flux points.

        Returns:
            Boolean mask with continuum areas.
        """

        # first guess of continuum in model
        cont, bunches, bunch_index = self._determine_continuum_bunches(model_y)
        cont = self._get_filtered_continuum(cont, model_y, bunches, bunch_index)

        # normalize model
        fit = np.poly1d(np.polyfit(model_x[cont & self._mask], model_y[cont & self._mask], 4))
        model_cont = fit(model_x)
        model_norm = model_y / model_cont

        # find continuum points in normalized model
        cont, bunches, bunch_index = self._determine_continuum_bunches(model_norm)
        return self._get_filtered_continuum(cont, model_norm, bunches, bunch_index)

    def _determine_continuum_bunches(self, y) -> Tuple[np.ndarray, List[int], np.ndarray]:
        """Determine bunches of continuum in given spectrum.

        Args:
            y: Array of spectrum flux points

        Returns:
            Tuple containing:
                - Mask of continuum points
                - List of bunch names
                - Mask with bunch names
        """

        continuum = []
        bunches = []
        bunch_index = []
        start = 0
        # go through spectrum
        while start < len(y):
            # reset peak to peak value and size of bunch
            ptp = 0
            size = 0
            # create bunch until max_ptp exceeded or end of spectrum reached
            while ptp < self._max_ptp and start + size < len(y):
                size += 1
                ptp = np.ptp(y[start:start + size] / np.median(y))

            # reduce size by 1, cause last point is not continuum
            size -= 1

            # fulfilling continuum criteria?
            if size >= self._min_bunch_size:
                continuum += list(np.ones(size, dtype=bool))

                # store new bunch
                new_bunch = len(bunches) + 1
                bunches.append(new_bunch)

                bunch_index += list(np.ones(size, dtype=int) * new_bunch)
                start += size
            else:
                continuum += [False]

                bunch_index += [0]
                start += 1

        return np.array(continuum), bunches, np.array(bunch_index)

    def _get_filtered_continuum_bunches(self, y, bunches, bunch_index):
        # store first bunch, cause we can only start filtering with second
        filtered_bunches = [bunches[0]]

        # calculate median of all bunches spectrum flux
        median = {}
        for bunch in bunches:
            median[bunch] = np.median(y[bunch_index == bunch])

        # compare each bunch with neighbors, starting with second and ending with second last
        for i in range(1, len(bunches) - 1):
            if median[bunches[i]] > self._threshold * median[bunches[i - 1]] or \
                    median[bunches[i]] > self._threshold * median[bunches[i + 1]]:
                filtered_bunches.append(bunches[i])

        # store last bunch
        filtered_bunches.append(bunches[-1])

        return filtered_bunches

    def _get_filtered_continuum(self, continuum, spectrum_flux, bunches, bunch_index):
        # iterate to identify wrong continuum bunches
        for _ in range(self._iterations):
            bunches = self._get_filtered_continuum_bunches(spectrum_flux, bunches, bunch_index)

        # remove wrong continuum bunches from continuum
        for i, bunch in enumerate(bunch_index):
            if bunch not in bunches:
                continuum[i] = False

        return continuum


__all__ = ['Continuum', 'MaximumBin', 'Regions', 'SigmaClipping']
