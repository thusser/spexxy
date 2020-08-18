import math
import os
from fnmatch import fnmatch
from typing import List

import numpy as np
from scipy.integrate import trapz
from spexxy.data.spectrum import Spectrum


class Filter(object):
    """ A filter class than can be applied to spectra """

    def __init__(self, filter_name: str = None, file_extension: str = '.txt', path: str = '$SPEXXYPATH/filters'):
        """Initializes a filter object with the given filter.

        Names for files that contain filter data are build as <path>/filters/<filter_name>.<extension>.

        Args:
            filter_name: Name of filter data to load without extension
            file_extension: Extension for file.
            path: Path the file is located in.
        """

        # init
        self.filter_name = filter_name
        self._throughput = None
        self._vega = None
        self._vega_throughput = None

        # if no filter name is given, we're finished here
        if filter_name is None:
            return

        # is filter_name a filename?
        if os.path.exists(filter_name):
            # just set it
            self._filter_filename = filter_name
        else:
            # build filename
            directory = os.path.expandvars(path)
            self._filter_filename = os.path.join(directory, filter_name + file_extension)

        # load filter
        self._throughput = Spectrum.load(self._filter_filename)

    @property
    def filename(self):
        """Filename for filter"""
        return self._filter_filename

    @property
    def wave(self):
        """Wavelength array the filter is defined on"""
        return self._throughput.wave

    @property
    def throughput(self):
        """Filter response curve"""
        return self._throughput.flux

    def copy(self):
        """Copy filter

        Returns:
            Copy of this filter
        """

        # create new filter
        f = Filter()

        # copy data
        f.filter_name = self.filter_name
        f._filter_filename = self._filter_filename
        f._throughput = self._throughput.copy() if self.throughput is not None else None
        f._vega = self._vega.copy() if self._vega is not None else None
        f._vega_throughput = self._vega_throughput.copy() if self._vega_throughput is not None else None

        # return it
        return f

    @staticmethod
    def list(pattern: str = '*', file_extension: str = '.txt', path: str = '$SPEXXYPATH/filters') -> List['Filter']:
        """Load all filters in a given directory that match the pattern and file extension.

        Args:
            pattern: Pattern for filters, e.g. 'Johnson/*' or 'hst/acs/*'.
            file_extension: File extension of filter files in given directory.
            path: Root path to start search in.

        Returns:
            List of filters that match the criteria.
        """

        # expand path
        p = os.path.expandvars(path)

        # walk path
        filters = []
        for root, dirs, files in os.walk(p):
            for file in sorted(files):
                filename = os.path.splitext(os.path.join(root, file)[len(p) + 1:])[0]
                if fnmatch(filename, pattern):
                    filters.append(Filter(filename, file_extension=file_extension, path=path))
        return filters
        
    def resample(self, inplace: bool = True, *args, **kwargs):
        """Resamples filter to other wavelength grid.

        Args:
            inplace: If True, this filter is changed, otherwise a new one is returned.
            args, kwargs: Forwarded to Spectrum's resample method.

        Returns:
            If inplace is True, a new Filter with the resampled response curve.
        """

        # do linear interpolation by default
        if 'linear' not in kwargs:
            kwargs['linear'] = True

        # fill missing values with zeros
        kwargs['fill_value'] = 0.

        # delegate resampling to Spectrum class
        if inplace:
            self._throughput = self._throughput.resample(*args, **kwargs)
        else:
            # create new filter
            f = self.copy()
            f._throughput = f._throughput.resample(*args, **kwargs)
            return f

    def integrate(self, spec: Spectrum) -> float:
        """Apply the filter to a given spectrum and integrate it.

        Args:
            spec: The spectrum to integrate together with the filter.

        Returns:
            Integrated flux of the given spectrum after applying the filter.
        """

        # lambda mode
        if spec.wave_mode != Spectrum.Mode.LAMBDA:
            raise ValueError('Spectrum must be in LAMBDA mode.')

        # wavelength grid okay?
        if not np.array_equal(spec.wave, self._throughput.wave):
            raise ValueError('Please resample filter before applying it.')

        # apply filter
        fs = spec.flux * self._throughput.flux

        # integrate
        return trapz(fs, spec.wave) / trapz(self._throughput.flux, spec.wave)

    def stmag(self, spec: Spectrum) -> float:
        """Calculate ST magnitude of given spectrum.

        Args:
            spec: The spectrum to get ST magnitude for.

        Returns:
            ST magnitude of the given spectrum in this filter.
        """

        # lambda mode
        if spec.wave_mode != Spectrum.Mode.LAMBDA:
            raise ValueError('Spectrum must be in LAMBDA mode.')

        # convolve with filter and integrate
        integral = self.integrate(spec)

        # ST magnitude
        return -2.5 * math.log10(integral) - 21.1

        # calculate stmag
        return self.stmag(spec.wave, spec.flux)

    def _load_vega(self):
        """Load VEGA spectrum."""
        if self._vega is None or self._vega_throughput is None:
            # get path to vega.sed
            filename = os.path.join(os.path.dirname(__file__), 'vega.sed')

            # no vega given? use default
            self._vega = Spectrum.load(filename)

            # create filter for vega
            tmp = Spectrum.load(self._filter_filename)
            self._vega_throughput = tmp.resample(spec=self._vega, fill_value=0., linear=True)

    def vegamag(self, spec: Spectrum) -> float:
        """Calculate Vega magnitude of given spectrum.

        Args:
            spec: The spectrum to get Vega magnitude for.

        Returns:
            Vega magnitude of the given spectrum in this filter.
        """

        # load vega spectrum
        self._load_vega()

        # lambda mode
        if spec.wave_mode != Spectrum.Mode.LAMBDA:
            raise ValueError('Spectrum must be in LAMBDA mode.')

        # no vega given?
        if self._vega is None:
            raise ValueError('No Vega spectrum given.')

        # wavelength grid okay?
        if not np.array_equal(spec.wave, self._throughput.wave):
            raise ValueError('Please resample filter before applying it.')

        # if no filter is given, use full spectrum
        if self.filter_name is None:
            # whole spectrum
            flux1 = trapz(spec.flux, spec.wave)
            flux2 = trapz(self._vega.flux, self._vega.wave)

        else:
            # multiply spectra with filter
            filter_spec = spec.flux * self._throughput.flux
            filter_vega = self._vega.flux * self._vega_throughput.flux

            # integrate
            w = ~np.isnan(filter_spec)
            flux1 = trapz(filter_spec[w], spec.wave[w]) / trapz(self._throughput.flux[w], spec.wave[w])
            w = ~np.isnan(filter_vega)
            flux2 = trapz(filter_vega[w], self._vega.wave[w]) / trapz(self._vega_throughput.flux[w], self._vega.wave[w])

        # check
        if flux2 == 0:
            raise ValueError('Integrated flux for Vega spectrum is zero.')

        # calculate Vega magnitude
        mag = -2.5 * math.log10(flux1 / flux2)
        return mag


__all__ = ['Filter']
