import copy
import math
import os
from enum import Enum
import numpy as np
import scipy
import scipy.linalg
import scipy.ndimage.filters
import scipy.optimize
import scipy.signal
from astropy.io import fits
from scipy.interpolate import UnivariateSpline, interp1d
from typing import Tuple, Union
import h5py

from ..utils.exception import *


class Spectrum(object):
    """A class representing a single spectrum."""

    """Data type for spectra."""
    dtype = np.float64

    class Mode(Enum):
        """Spectrum wavelength mode."""
        LAMBDA = 0
        LOGLAMBDA = 1
        LOG10LAMBDA = 2

    def __init__(self, spec: 'Spectrum' = None, copy_flux: bool = True, wave: np.ndarray = None,
                 wave_start: float = None, wave_count: int = None, wave_step: float = None, wave_end: float = None,
                 wave_mode: Mode = Mode.LAMBDA, flux: np.ndarray = None, valid: np.ndarray = None):
        """Initialize a new spectrum.

        Initialization follows the following pattern:
        - Is spec given?
            - Yes, copy everything from that Spectrum. Copy flux only, if copy_flux is True.
            - No, start with empty spectrum:
                - Is wave given?
                    - Yes, just copy wavelength array.
                    - No, create new wavelength grid:
                        - If no wave_count is given, calculate it from wave_start, wave_step, and wave_end. Therefore
                          those MUST be provided!
                        - Store wave_start, wave_step, wave_count, and wave_mode
                - Do we now have a valid wavelength grid?
                    - Yes:
                        - Copy flux array, if given, otherwise create empty one
                        - Copy valid array, if given, otherwise create new one (all valid, if flux was given,
                          otherwise all invalid)

        Args:
            spec: If given, initialize this spectrum from given one.
            copy_flux: Only used in combination with spec: If False, do not copy flux array.
            wave: If given, initialize wavelength with this array.
            wave_start: If given, defines first wavelength point.
            wave_count: If given, defines number of points in spectrum.
            wave_step: If given, defines wavelength step.
            wave_end: If given, defines last wavelength point.
            wave_mode: Wavelength mode (lambda or log).
            flux: If given, defines flux array.
            valid: If given, defines valid pixel mask.
        """
        
        # start empty
        self.flux = None
        self._wave_start = None
        self._wave_step = None
        self._wavelength = None
        self._wave_mode = None
        self._valid = None

        # got a ref_spec?
        if spec is not None:
            # initialize from ref_spec
            self.flux = spec.flux.astype(Spectrum.dtype) if copy_flux else None
            self._wave_start = spec.wave_start
            self._wave_step = spec.wave_step
            self._wavelength = None if spec._wavelength is None else spec._wavelength.copy()
            self._wave_mode = spec.wave_mode
            self._valid = None if spec._valid is None else spec._valid.copy()

        # wavelength array given?
        if wave is not None:
            # init from wavelength
            self._wave_start = wave[0]
            self._wave_step = 0.
            self._wavelength = wave
            self._wave_mode = wave_mode
            wave_count = len(wave)

        else:
            # calculate wave_count
            if wave_count is None and wave_end is not None:
                wave_count = int((wave_end - wave_start) / wave_step)

        # init
        if wave_start is not None and wave_step is not None and wave_mode is not None:
            self._wave_start = wave_start
            self._wave_step = wave_step
            self._wavelength = None
            self._wave_mode = wave_mode

        # initialize flux and valid only if we got a valid wavelength array
        if wave_step is not None:
            # if no flux is given, initialize it empty
            if flux is None:
                # create flux array and fill with NaNs
                flux = np.empty((wave_count), dtype=Spectrum.dtype)
                flux[:] = np.nan

                # if also no valid array is given, it's all invalid now
                if valid is None:
                    # fill valid with zeros
                    valid = np.zeros(flux.shape, dtype=bool)

            else:
                # if we got some flux, but no valid array, we assume all valid
                valid = np.ones(flux.shape, dtype=bool)

        # finally set flux and valid
        if flux is not None:
            self.flux = flux
        if valid is not None:
            self._valid = valid

    @property
    def wave(self) -> np.ndarray:
        """Returns the wavelength array for this spectrum and creates it first, if necessary.

        Returns:
            Wavelength array.
        """

        if self._wavelength is None:
            # create new wavelength array
            length = len(self.flux) * self._wave_step - 0.5 * self._wave_step
            self._wavelength = np.arange(self._wave_start,
                                         self._wave_start + length,
                                         self._wave_step)
        return self._wavelength

    def pixel_borders(self) -> list:
        """Returns a list containing the borders of the pixels, always one entry more than number of wavelength points.

        Returns:
            List of pixel borders.
        """

        # get wave array
        wave = self.wave

        # get central positions between pixels
        centres = 0.5 * (wave[1:] + wave[:-1])

        # add first and last border
        centres = [2. * wave[0] - centres[0]] + list(centres)
        centres = list(centres) + [2. * wave[-1] - centres[-1]]

        # return
        return centres

    def pixel_sizes(self, scale: float = 1.) -> list:
        """Returns list of (left,right) tuples for each pixel in spectrum containing its left and right border.
        The size of the pixels can be scales using the scale parameter.

        Args:
            scale: Scale the pixel sizes with this factor.

        Returns:
            List of pixel sizes.
        """

        # get borders and calculate sizes
        borders = self.pixel_borders()
        sizes = list(zip(borders[:-1], borders[1:]))

        # scale?
        if scale == 1.:
            return sizes
        else:
            # scale left and right by (1-scale)/2
            s = (1. - scale) / 2.
            return [(l + (r - l) * s, r - (r - l) * s) for l, r in sizes]

    @property
    def valid(self) -> np.ndarray:
        """Returns the mask of valid pixels."""
        return self._valid

    @valid.setter
    def valid(self, v: np.ndarray):
        """Sets the mask of valid pixels. Must be of same length as flux."""

        # check length
        if len(v) != len(self.flux):
            raise ValueError("Length of valid array must be equal to "
                             "length of flux array.")

        # set it
        self._valid = v

    def __getitem__(self, idx: int):
        """Returns a single value from the flux array.

        Args:
            idx: Index in flux array.

        Returns:
            Flux at given index.
        """
        return self.flux[idx]

    def __setitem__(self, idx: int, value: float):
        """Sets a single value in the flux array,

        Args:
            idx: Index in flux array.
            value: Value to set.

        Returns:
            Flux at given index.
        """
        self.flux[idx] = value

    @property
    def wave_start(self) -> float:
        """Wavelength of first point in spectrum."""
        return self._wave_start

    @wave_start.setter
    def wave_start(self, v: float):
        """Sets a new start wavelength and invalidates wavelength array.

        Args:
            v: New start wavelength.
        """
        self._wave_start = v
        self._wavelength = None

    @property
    def wave_step(self) -> float:
        """Wavelength step."""
        return self._wave_step

    def __len__(self) -> int:
        """Number of pixels in spectrum."""
        return len(self.flux)

    def __iter__(self):
        """Return an iterator over the flux."""
        return iter(self.flux)

    @property
    def wave_mode(self) -> Mode:
        """Mode of spectrum."""
        return self._wave_mode

    def redshift(self, vrad: float):
        """Shift the spectrum by a given radial velocity.

        Args:
            vrad: Radial velocity in km/s to shift by.
        """

        # which mode?
        if self._wave_mode == Spectrum.Mode.LAMBDA:
            # wavelength in AA means a multiplication
            self._wavelength = self.wave * (1. + vrad / 299792.458)
            # and non-constant step size
            self._wave_step = 0
        elif self._wave_mode == Spectrum.Mode.LOGLAMBDA:
            # wavelength in log domain is simpler
            self._wavelength = self.wave + math.log(1. + vrad / 299792.458)
            self._wave_start = self._wavelength[0]
        elif self._wave_mode == Spectrum.Mode.LOG10LAMBDA:
            self._wavelength = self.wave + math.log10(1. + vrad / 299792.458)
            self._wave_start = self._wavelength[0]
        else:
            raise NotImplementedError('Unsupported wave mode: {}'.format(self._wave_mode))

    def resample(self, spec: 'Spectrum' = None, wave: np.ndarray = None, wave_start: float = None,
                 wave_count: int = None, wave_step: float = None, wave_end: float = None, vrad: float = 0.,
                 linear: bool = False, flux_conserve: bool = False, fill_value: float = np.nan) -> 'Spectrum':
        """Resamples the spectrum to a new grid.

        Parameters defining the wavelength grid are just passed to the constructor of Spectrum.

        Args:
            spec: If given, initialize this spectrum from given one.
            wave: If given, initialize wavelength with this array.
            wave_start: If given, defines first wavelength point.
            wave_count: If given, defines number of points in spectrum.
            wave_step: If given, defines wavelength step.
            wave_end: If given, defines last wavelength point.
            vrad: In the process, do a radial velocity shift.
            linear: Do a linear interpolation instead of using a cubic spline (faster!).
            flux_conserve: Conserve flux instead of intensity.
            fill_value: Value to use for extrapolations.

        Returns:
            Resampled spectrum.
        """

        # no flux?
        if len(self.flux) < 2:
            return

        # get old wavelength array
        cur_wave = self.wave.copy()

        # radial velocity?
        if self._wave_mode == Spectrum.Mode.LAMBDA:
            cur_wave[~np.isnan(cur_wave)] *= 1. + vrad / 299792.458
        else:
            cur_wave[~np.isnan(cur_wave)] += math.log(1. + vrad / 299792.458)

        # create output spectrum
        output = self.__class__(spec=spec, wave=wave, wave_start=wave_start, wave_count=wave_count,
                                wave_step=wave_step, wave_end=wave_end, wave_mode=self.wave_mode)

        # set flux to NaN
        output.flux[:] = np.nan

        # do we have any NaNs in wave or flux?
        parts = []
        if np.any(np.isnan(cur_wave)) or np.any(np.isnan(self.flux)):
            # split into several sub spectra
            start_index = None
            for i in range(len(self.flux)):
                if np.isnan(cur_wave[i]) or np.isnan(self.flux[i]):
                    # found a NaN value
                    if start_index is not None:
                        # extract part
                        parts.append((start_index, self.extract_index(start_index, i)))
                        start_index = None
                else:
                    # start a new part
                    if start_index is None:
                        start_index = i

            # add last part?
            if start_index is not None:
                parts.append((start_index, self.extract_index(start_index, len(self.flux))))

        else:
            # single part
            parts.append((0, self))

        # loop all parts
        for start_index, part in parts:
            # interpolate
            ip = Spectrum._resample(part, output.wave, flux_conserve=flux_conserve,
                                    linear=linear, fill_value=fill_value)

            # copy all values in range of this part
            output.flux[start_index:start_index+len(ip)] = ip[start_index:start_index+len(ip)]
            #i_min, i_max = output.indices_of_wave_range(part.wave.min(), part.wave.max())
            #output.flux[i_min:i_max] = ip[i_min:i_max]

        # return result
        return output
    
    @staticmethod
    def _resample(spec: 'Spectrum', wave: np.ndarray, flux_conserve: bool = False, linear: bool = False,
                  fill_value: float = np.nan) -> np.ndarray:
        """Does the actual resampling of a spectrum.

        Args:
            spec: Spectrum to resample.
            wave: Output wavelength array.
            flux_conserve: If True, conserves flux instead of intensity.
            linear: If True, uses linear interpolation instead of a cubic spline.
            fill_value: Fill value for extrapolation.

        Returns:
            Resampled flux
        """

        # interpolate
        if not flux_conserve:
            # do interpolation, force linear if less than 3 points, return Nones if less than 2 points
            if len(spec) < 2 or len(wave) < 2:
                return [None] * len(wave)
            elif linear or len(spec) < 3:
                # interpolate linearly
                interpol_flux = interp1d(x=spec.wave, y=spec.flux, kind='linear',
                                         bounds_error=False, fill_value=fill_value)
                return interpol_flux(wave)

            else:
                # interpolate using a spline
                interpol_flux = UnivariateSpline(spec.wave, spec.flux, k=2, s=0, ext=1)
                new_flux = interpol_flux(wave)

                # since UnivariateSpline doesn't support a fill value, do a workaround
                new_flux[new_flux == 0] = fill_value

                #return it
                return new_flux

        else:
            # old borders
            old_borders = np.empty((len(spec.flux) + 1))
            old_borders[1:-1] = (spec.wave[1:] + spec.wave[0:-1]) * 0.5
            old_borders[0] = spec.wave[0] + spec.wave[1] - old_borders[1]
            old_borders[-1] = spec.wave[-2] - spec.wave[-1] + old_borders[-2]

            # new borders
            new_borders = np.empty((len(wave) + 1))
            new_borders[1:-1] = (wave[1:] + wave[0:-1]) * 0.5
            new_borders[0] = wave[0] + wave[1] - new_borders[1]
            new_borders[-1] = wave[-2] - wave[-1] + new_borders[-2]

            # integrate
            integr = np.concatenate([[0], np.cumsum(spec.flux)])

            # interpolate
            interpol_flux = scipy.interpolate.interp1d(x=old_borders, y=integr, kind='linear',
                                                       bounds_error=False, fill_value=1.)
            interp = interpol_flux(new_borders)

            # differentiate
            return interp[1:] - interp[0:-1]

    def resample_const(self, step: float = 0, flux_conserve: bool = False) -> 'Spectrum':
        """Resample spectrum to constant wavelength step size. If none is given, the lowest step size in
        existing grid is used.

        Args:
            step: New step size.
            flux_conserve: Whether or not to conserve flux. Otherwise intensity is conserved.

        Returns:
            Resampled spectrum
        """

        # nothing to do?
        if step == 0 and self._wave_step > 0:
            return self.copy(copy_flux=True)

        # filter wavelength array for NaNs
        w = np.sort(self.wave)
        w = w[~np.isnan(w)]

        # resample
        if step == 0:
            # get step sizes and discard zero values
            steps = np.array(w[1:] - w[:-1])
            steps = steps[~np.isnan(steps)]
            steps = steps[steps > 0]

            # get minimum and scale by two
            step = 0.5 * np.min(steps)

            # get number of steps
            n = math.ceil((w[-1] - w[0]) / step)

            # rescale step
            step = (w[-1] - w[0]) / n
        else:
            n = math.ceil((w[-1] - w[0]) / step)

        # finally resample
        return self.resample(wave_start=w[0], wave_count=n, wave_step=step, flux_conserve=flux_conserve)

    def add_noise(self, sigma: float):
        """Add gaussian noise with given sigma.

        Args:
            sigma: Sigma of noise to add.
        """
        if sigma == 0.:
            return
        for i in range(len(self.flux)):
            self.flux[i] += np.random.normal(0., self.flux[i] * sigma)

    def add_noise_snr(self, snr: float):
        """Add gaussian noise with a given S/N ratio.

        Args:
            snr: S/N ratio of noise to add.
        """
        if snr == 0.:
            return
        self.add_noise(1. / snr)

    def smooth(self, fwhm: float = None, sigma: float = None):
        """Smooth spectrum with a Gaussian kernel. Either FWHM or sigma should be given.

        Args:
            fwhm: FWHM of gaussian kernel.
            sigma: Sigma of gaussian kernel.
        """

        # smoothing not supported on irregular grids
        if self._wave_step == 0.:
            raise ValueError("Smoothing not supported for unbinned spectra.")

        # calculate sigma
        if sigma is None:
            # 2.35...=(2*math.sqrt(2*math.log(2)))
            sigma = fwhm / 2.3548200450309493

        # sigma is now in Angstroms, change it to pixels
        sigma /= self._wave_step

        # smooth
        self.flux = scipy.ndimage.filters.gaussian_filter(self.flux, sigma)

    def index_of_wave(self, wave: float) -> int:
        """Returns index of given wavelength in wavelength array.

        Args:
            wave: Wavelength to return index for.

        Returns:
            Index in wavelength array or None if outside range.
        """

        # get wave array
        w = self.wave

        # too small/large?
        if wave < 0.5 * (w[1] + w[0]):
            raise spexxyValueTooLowException()
        if wave >= 0.5 * (w[-1] + w[-2]):
            raise spexxyValueTooHighException()

        # find it
        for i in range(1, len(w) - 1):
            if 0.5 * (w[i] + w[i - 1]) <= wave < 0.5 * (w[i + 1] + w[i]):
                return i
        return None

    def indices_of_wave_range(self, w1: float, w2: float) -> Tuple[int, int]:
        """Same as index_of_wave, but optimized for ranges.

        Args:
            w1: Start wavelength for range.
            w2: End wavelength for range.

        Returns:
            Start and end index of range
        """

        # cover whole range?
        if w1 <= self.wave[0] and w2 >= self.wave[-1]:
            return 0, len(self.wave) - 1

        # get indices for w1 and w2
        try:
            i1 = self.index_of_wave(w1)
        except spexxyValueTooLowException:
            i1 = 0
        except spexxyValueTooHighException:
            return None
        try:
            i2 = self.index_of_wave(w2) + 1   # +1 in for including this pixel
        except spexxyValueTooLowException:
            return None
        except spexxyValueTooHighException:
            i2 = len(self.wave) - 1

        # mask at least 1px
        if i2 <= i1 < len(self.wave) - 1:
            i2 = i1 + 1

        # return indices
        return i1, i2

    def extract_index(self, i1: int, i2: int) -> 'Spectrum':
        """Extract spectrum in given index range.

        Args:
            i1: Start index to extract.
            i2: End index (not included) to extract.

        Returns:
            Extracted spectrum
        """
        if self._wave_step != 0.:
            spec = self.__class__(spec=self, flux=np.copy(self.flux[i1:i2]), wave_start=self.wave[i1],
                                  wave_step=self._wave_step, wave_mode=self._wave_mode)
        else:
            spec = self.__class__(spec=self, flux=np.copy(self.flux[i1:i2]), wave=np.copy(self.wave[i1:i2]),
                                  wave_mode=self._wave_mode)

        if self.valid is not None:
            spec.valid = np.copy(self.valid[i1:i2])

        return spec

    def extract(self, w1: float, w2: float) -> 'Spectrum':
        """Extract spectrum in given wavelength range.

        Args:
            w1: Start of wavelength range to extract.
            w2: End of wavelength range to extract.

        Returns:
            Extracted spectrum.
        """

        # get start and end indices
        s1 = self.index_of_wave(w1) or 0
        s2 = self.index_of_wave(w2) or 0
        if s1 < s2 + 1:
            s1 += 1

        # extract
        return self.extract_index(s1, s2)

    def vac_to_air(self):
        """Transforms spectrum from vacuum wavelengths to air wavelengths."""

        # copy wave
        wave = self.wave.copy()

        # if log, convert to lambda
        if self._wave_mode == Spectrum.Mode.LOGLAMBDA:
            wave = np.exp(wave)

        # convert vac to air
        sigma2 = (1e4 / wave) * (1e4 / wave)
        fact = 1. + 5.792105e-2 / (238.0185 - sigma2) + \
               1.67917e-3 / (57.362 - sigma2)

        # only set for lambda>2000A
        self._wavelength = wave.copy()
        self._wavelength[wave > 2000] = wave[wave > 2000] / fact[wave > 2000]
        self._wave_step = 0.

        # convert back to log, if necessary
        if self._wave_mode == Spectrum.Mode.LOGLAMBDA:
            self._wavelength = np.log(self._wavelength)

        # set wStart
        self._wave_start = self._wavelength[0]

    def mode(self, m: Mode):
        """Change mode of spectrum.

        Args:
            m: New mode of spectrum.
        """

        # convert
        if self._wave_mode == Spectrum.Mode.LAMBDA and m == Spectrum.Mode.LOGLAMBDA:
            # LAMBDA to LOG
            self._wavelength = np.log(self.wave)
        elif self._wave_mode == Spectrum.Mode.LAMBDA and m == Spectrum.Mode.LOG10LAMBDA:
            # LAMBDA to LOG10
            self._wavelength = np.log10(self.wave)
        elif self._wave_mode == Spectrum.Mode.LOGLAMBDA and m == Spectrum.Mode.LAMBDA:
            # LOG to LAMBDA
            self._wavelength = np.exp(self.wave)
        elif self._wave_mode == Spectrum.Mode.LOGLAMBDA and m == Spectrum.Mode.LOG10LAMBDA:
            # LOG to LOG10
            self._wavelength = np.log10(np.exp(self.wave))
        elif self._wave_mode == Spectrum.Mode.LOG10LAMBDA and m == Spectrum.Mode.LAMBDA:
            # LOG10 to LAMBDA
            self._wavelength = np.power(10., self.wave)
        elif self._wave_mode == Spectrum.Mode.LOG10LAMBDA and m == Spectrum.Mode.LOGLAMBDA:
            # LOG10 to LOG
            self._wavelength = np.log(np.power(10., self.wave))
        else:
            # do nothing
            return

        # set values
        self._wave_start = self._wavelength[0]
        self._wave_step = 0.
        self._wave_mode = m

    def norm_to_mean(self) -> float:
        """Normalize spectrum to mean flux.

        Returns:
            Factor the spectrum has been divided by.
        """
        norm = np.mean(self.flux)
        self.flux /= norm
        return norm

    def norm_to_one(self) -> float:
        """Normalize spectrum to an integrated value of one.

        Returns:
            Factor the spectrum has been divided by.
        """
        norm = scipy.integrate.trapz(self.flux, self.wave)
        self.flux /= norm
        return norm

    def norm_at_wavelength(self, wave: float = 5000, width: float = None) -> float:
        """Normalize spectrum so that flux at given wavelength is one. If width>0 and not None, use average of region.

        Args:
            wave: Wavelength to norm to.
            width: Width of region to average.

        Returns:
            Factor the spectrum has been divided by.
        """

        if width is None:
            # get flux at wavelength
            norm = self.flux[self.index_of_wave(wave)]
        else:
            # get average in range
            idx = self.indices_of_wave_range(wave - 0.5 * width, wave + 0.5 * width)
            norm = np.mean(self.flux[idx[0]:idx[1]])

        # divide and return
        self.flux /= norm
        return norm

    def __truediv__(self, other: Union['Spectrum', np.ndarray, float, int]) -> 'Spectrum':
        """Divide my flux with something else and return result.

        Args:
            other: Something to divide by.

        Returns:
            New spectrum with the result of the division.
        """

        # make a copy
        copy = self.copy(copy_flux=False)

        # check type
        if isinstance(other, Spectrum):
            copy.flux = self.flux / other.flux
        else:
            copy.flux = self.flux / other

        # return result
        return copy

    def __itruediv__(self, other: Union['Spectrum', np.ndarray, float, int]) -> 'Spectrum':
        """Divide my flux with something else.

        Args:
            other: Something to divide by.

        Returns:
            Input spectrum spec divided by other.
        """

        # check type
        if isinstance(other, Spectrum):
            self.flux /= other.flux
        else:
            self.flux /= other

        # return result
        return self

    def __mul__(self, other: Union['Spectrum', np.ndarray, float, int]) -> 'Spectrum':
        """Multiply my flux with something else and return result.

        Args:
            other: Something to multiply with.

        Returns:
            New spectrum with the result of the multiplication.
        """

        # make a copy
        copy = self.copy(copy_flux=False)

        # check type
        if isinstance(other, Spectrum):
            copy.flux = self.flux * other.flux
        else:
            copy.flux = self.flux * other

        # return result
        return copy

    def __imul__(self, other: Union['Spectrum', np.ndarray, float, int]) -> 'Spectrum':
        """Multiply my flux with something else.

        Args:
            other: Something to multiply with.

        Returns:
            Input spectrum spec multiplied with other.
        """

        # check type
        if isinstance(other, Spectrum):
            self.flux *= other.flux
        else:
            self.flux *= other

        # return result
        return self

    def __add__(self, other: Union['Spectrum', np.ndarray, float, int]) -> 'Spectrum':
        """Add something else to my flux and return result.

        Args:
            other: Something to add to my flux.

        Returns:
            New spectrum with the result of the addition.
        """

        # make a copy
        copy = self.copy(copy_flux=False)

        # check type
        if isinstance(other, Spectrum):
            copy.flux = self.flux + other.flux
        else:
            copy.flux = self.flux + other

        # return result
        return copy

    def __iadd__(self, other: Union['Spectrum', np.ndarray, float, int]) -> 'Spectrum':
        """Add something else to my flux.

        Args:
            other: Something to add to my flux.

        Returns:
            Input spectrum spec multiplied with other.
        """

        # check type
        if isinstance(other, Spectrum):
            self.flux += other.flux
        else:
            self.flux += other

        # return result
        return self

    def __sub__(self, other: Union['Spectrum', np.ndarray, float, int]) -> 'Spectrum':
        """Subtract something else from my flux and return result.

        Args:
            other: Something to subtract from my flux.

        Returns:
            New spectrum with the result of the subtraction.
        """

        # make a copy
        copy = self.copy(copy_flux=False)

        # check type
        if isinstance(other, Spectrum):
            copy.flux = self.flux - other.flux
        else:
            copy.flux = self.flux - other

        # return result
        return copy

    def __isub__(self, other: Union['Spectrum', np.ndarray, float, int]) -> 'Spectrum':
        """Subtract something else from my flux.

        Args:
            other: Something to subtract from my flux.

        Returns:
            Input spectrum spec with other subtracted.
        """

        # check type
        if isinstance(other, Spectrum):
            self.flux -= other.flux
        else:
            self.flux -= other

        # return result
        return self

    def estimate_snr(self, model: 'Spectrum') -> float:
        """Estimate S/N ratio for spectrum using a given model.

        Args:
            model: Model to compare with.

        Returns:
            Estimated S/N ratio
        """

        # get valid points
        v = ~np.isnan(model.flux) & ~np.isnan(self.flux)

        # get normalized residuals
        res = (self.flux[v] - model.flux[v]) / model.flux[v]

        # do some kappa-sigma-clipping
        for i in range(5):
            # get mean and standard deviation
            m = np.mean(res)
            s = np.std(res)

            # do clipping
            res = res[np.abs(res-m) < 4.*s]

        # return 1/s
        s = np.std(res)
        return 1 / s if s > 0 else None

    def copy(self, copy_flux: bool = True) -> 'Spectrum':
        """Create a copy of this spectrum.

        Args:
            copy_flux: If False, flux is not copied.

        Returns:
            Copy of this spectrum.
        """
        return copy.deepcopy(self) if copy_flux else copy.copy(self)

    def __copy__(self):
        """Create a shallow copy of this spectrum.

        Returns:
            Copy of this spectrum.
        """
        return self.__class__(spec=self, copy_flux=False)

    def __deepcopy__(self, memodict={}):
        """Create a deep copy of this spectrum.

        Returns:
            Copy of this spectrum.
        """
        return self.__class__(spec=self, copy_flux=True)

    @staticmethod
    def load(filename: str):
        """Loads a spectrum from file and guesses type.

        Args:
            filename: Name of file to load spectrum from.

        Returns:
            Loaded spectrum.
        """

        # expand vars on filename
        filename = os.path.expandvars(filename)

        # is it a FITS file?
        if filename.endswith(".fit") or filename.endswith(".FIT") or \
                filename.endswith(".fits") or filename.endswith(".FITS"):

            # get header
            hdr = fits.getheader(filename)

            # WAVE keyword?
            if "WAVE" in hdr.keys():
                # normal fits spectrum with WAVE extension
                return SpectrumFits(filename)

            elif hdr["NAXIS"] == 0:
                # no axis given? bin tabke!
                return SpectrumBinTableFITS(filename)

            else:
                # normal fits spectrum
                return SpectrumFits(filename)

        elif filename.endswith(".h5"):
            return SpectrumH5(filename)

        else:
            # no fits? try to load ascii...
            return SpectrumAscii(filename)


class SpectrumFitsHDU(Spectrum):
    """A Spectrum in a FITS HDU

    This class allows to load a spectrum from a FITS HDU and save it into one.
    """

    def __init__(self, hdu=None, hdu_list=None, primary=True, dtype=np.float32, filename: str = None, *args, **kwargs):
        """Initialize a new Spectrum from a FITS HDU

        Args:
            hdu: If given, HDU to load spectrum from.
            hdu_list: Probably necessary for loading wavelength array.
            primary: Whether the HDU should be a primary HDU. If HDU is given, this is ignored and derived automatically.
            dtype: Data type of array.
            filename: Name of file this HDU is in.
        """
        Spectrum.__init__(self, *args, **kwargs)

        # got a spec?
        if 'spec' in kwargs and isinstance(kwargs['spec'], SpectrumFitsHDU):
            # init from given spec
            spec = kwargs['spec']
            self._primary = primary
            self._hdu = spec._hdu.copy() if hdu is None and spec._hdu is not None else hdu
            self._dtype = spec._dtype if dtype is None else dtype

        else:
            # store
            self._hdu = hdu
            self._primary = primary
            self._dtype = dtype

            # init HDU
            if hdu:
                # overwrite primary, according to HDU
                self._primary = isinstance(hdu, fits.PrimaryHDU)

                # get data from HDU
                self.flux = hdu.data.astype(Spectrum.dtype)

                # get header
                hdr = hdu.header

                # do we have an extra HDU for wavelength array?
                if 'WAVE' in hdr:
                    # is there a HDU with this name?
                    if hdu_list is not None and hdr['WAVE'] in hdu_list:
                        # no file, must be HDU, check hdu_list
                        if hdu_list is None:
                            raise ValueError('No HDU list given for loading '
                                             'wavelength array.')

                        # get HDU and data
                        wave_hdu = hdu_list[hdr['WAVE']]

                    else:
                        # try to buld filename
                        wave_filename = os.path.join(os.path.dirname(filename), hdr['WAVE'])

                        # is it a file?
                        if os.path.exists(wave_filename):
                            # load from file
                            f = fits.open(wave_filename, memmap=False)
                            wave_hdu = f[0]
                            tmp = wave_hdu.data
                            f.close()

                        else:
                            # something else
                            raise ValueError('Could not load wavelength array')

                    # set data
                    self._wavelength = wave_hdu.data
                    self._wave_start = self._wavelength[0]
                    self._wave_step = 0

                    # type
                    if "CTYPE1" in wave_hdu.header.keys() and \
                            (wave_hdu.header["CTYPE1"] == "WAVE-LOG" or
                             wave_hdu.header["CTYPE1"] == "AWAV-LOG"):
                        self._wave_mode = Spectrum.Mode.LOGLAMBDA
                    else:
                        self._wave_mode = Spectrum.Mode.LAMBDA

                    # convert to Angstrom, if necessary
                    units = wave_hdu.header['CUNIT1'] if 'CUNIT1' in wave_hdu.header else 'Angstrom'
                    self._wavelength_to_angstrom(units)

                else:
                    # no, get wavelength array info
                    self._wave_start = hdr["START1"] if "START1" in hdr else hdr[
                        "CRVAL1"]
                    self._wave_step = hdr["STEP1"] if "STEP1" in hdr else hdr["CDELT1"]
                    if 'CRPIX1' in hdr and hdr['CRPIX1'] > 1:
                        self._wave_start -= (hdr['CRPIX1'] - 1) * self._wave_step
                    self._wavelength = None
                    if "CTYPE1" in hdr.keys() and \
                            (hdr["CTYPE1"] == "WAVE-LOG" or
                             hdr["CTYPE1"] == "AWAV-LOG"):
                        self._wave_mode = Spectrum.Mode.LOGLAMBDA
                    else:
                        self._wave_mode = Spectrum.Mode.LAMBDA

                    # convert to Angstrom, if necessary
                    units = hdr['CUNIT1'] if 'CUNIT1' in hdr else 'Angstrom'
                    self._wavelength_to_angstrom(units)

                # is it valid?
                self._valid = np.zeros(self.flux.shape, dtype=bool)

    def _wavelength_to_angstrom(self, units: str):
        """Convert wavelength units to Angstrom

        Args:
            units: Current unit of wavelength.
        """

        # everything okay?
        if units == 'Angstrom':
            return

        # get conversion factor
        factor = 1.
        if units == 'm':
            factor = 1e10
        else:
            raise ValueError('Unknown unit %s.' % units)

        # convert
        self._wave_start *= factor
        if self._wavelength:
            self._wavelength *= factor
        self._wave_step *= factor

    @property
    def header(self):
        """FITS header object"""
        hdu, _ = self.hdu()
        return hdu.header

    @property
    def primary(self):
        """Returns whether this HDU is supposed to be a primary HDU"""
        return self._primary

    def hdu(self):
        """Returns the updated HDU for this spectrum"""

        # get flux
        flux = self.flux.astype(self._dtype)

        # need to create HDU?
        if not self._hdu:
            # create HDU
            self._hdu = fits.PrimaryHDU(flux) if self._primary else fits.ImageHDU(flux)
        else:
            # just set data
            self._hdu.data = flux

        # do we need an extra HDU for the wavelength array?
        if self._wave_step != 0:
            # no
            wave_hdu = None

            #  set wavelength array values
            self._hdu.header["CRVAL1"] = (self._wave_start,
                                          "Wavelength of reference pixel")
            self._hdu.header["CDELT1"] = (self._wave_step,
                                          "Wavelength grid step size")
            self._hdu.header["CRPIX1"] = (1, "Reference pixel coordinates")
            self._hdu.header["WSTART"] = (self._wave_start,
                                          "Wavelength of reference pixel")
            self._hdu.header["WSTEP"] = (self._wave_step,
                                         "Wavelength grid step size")
            self._hdu.header["CUNIT1"] = "Angstrom"

            # type of array
            t = "AWAV" if self._wave_mode == Spectrum.Mode.LAMBDA \
                else "AWAV-LOG"
            self._hdu.header["CTYPE1"] = (t, "Type of wavelength grid")

        else:
            # yes, create it
            wave_hdu = fits.ImageHDU(self.wave)
            wave_hdu.name = 'WAVE'

            # type of array
            t = "AWAV" if self._wave_mode == Spectrum.Mode.LAMBDA \
                else "AWAV-LOG"
            wave_hdu.header["CTYPE1"] = (t, "Type of wavelength grid")

            # reference it
            self._hdu.header['WAVE'] = 'WAVE'

        # return new hdu
        return self._hdu, wave_hdu


class SpectrumFits(SpectrumFitsHDU):
    """A spectrum in a FITS file.

    The class extends SpectrumFitsHDU for a easier file handling.
    """

    def __init__(self, filename: str = None, extension: int = 0, *args, **kwargs):
        """Initialize a spectrum from a FITS file.

        Args:
            filename: Name of FITS file to load spectrum from.
            extension: Extension in FITS file to load from.
        """

        # store
        self._filename = filename

        # load it
        if filename:
            f = fits.open(filename, memmap=False)
            SpectrumFitsHDU.__init__(self, hdu=f[extension], hdu_list=f, filename=filename, *args, **kwargs)
            f.close()
        else:
            SpectrumFitsHDU.__init__(self, *args, **kwargs)

    @property
    def filename(self):
        return self._filename

    def save(self, filename: str = None, headers: dict = None):
        """Save spectrum to FITS file

        Args:
            filename: Name of file to save spectrum into.
            headers: Additional FITS headers.
        """

        # get HDU
        spec_hdu, wave_hdu = self.hdu()

        # add more headers?
        if headers:
            for key, value in headers.items():
                spec_hdu.header[key] = value

        # filename
        filename = filename or self._filename
        if not filename:
            raise ValueError('No filename given for saving the FITS file.')

        # create HDU list and save it
        hdu_list = fits.HDUList([spec_hdu])
        if wave_hdu:
            hdu_list.append(wave_hdu)
        hdu_list.writeto(filename, overwrite=True)

        # store filename
        self._filename = filename


class SpectrumHiResFITS(Spectrum):
    """Handles HiRes spectra from the PHOENIX library."""

    def __init__(self, filename: str = None, *args, **kwargs):
        """Loads a HiRes spectrum from file.

        Args:
            filename: Name of file to load spectrum from.
        """
        Spectrum.__init__(self, *args, **kwargs)

        # file given?
        if filename is not None:
            # get flux
            self.flux = fits.getdata(filename, 0)
            if len(self.flux) == 1 and len(self.flux[0]) > 1:
                self.flux = self.flux[0]

            # get header
            hdr = fits.getheader(filename, 0)

            # get path of filename
            path = os.path.dirname(filename)

            # get filename of wavelength file
            wave_file = path + ('/' if len(path) > 0 else "") + hdr["WAVE"]

            # get wavelength array
            self._wave_step = 0.
            self._wavelength = fits.getdata(wave_file)
            if "CTYPE1" in hdr.keys() and \
                    (hdr["CTYPE1"] == "WAVE-LOG" or hdr["CTYPE1"] == "AWAV-LOG"):
                self._wave_mode = Spectrum.Mode.LOGLAMBDA
            else:
                self._wave_mode = Spectrum.Mode.LAMBDA

            # is it valid?
            self._valid = np.zeros(self.flux.shape, dtype=bool)


class SpectrumBinTableFITS(Spectrum):
    """Handles spectra stored in FITS binary tables."""

    def __init__(self, filename: str, *args, **kwargs):
        """Loads a spectrum from a FITS binary table.

        Args:
            filename: Name of file to load spectrum from.
        """
        Spectrum.__init__(self, *args, **kwargs)

        # store
        self._filename = filename

        # load data
        if filename:
            try:
                data = fits.getdata(filename, "SPECTRUM")
                self._wavelength = data.WAVE
                self.flux = data.FLUX
            except KeyError:
                data = fits.getdata(filename, "SCI")
                self._wavelength = data.WAVELENGTH
                self.flux = data.FLUX

            # set wavelength array
            self._wave_start = self._wavelength[0]
            self._wave_step = 0

            # get array and decide on wave mode
            hdr = fits.getheader(filename)
            if "CTYPE1" in hdr.keys() and \
                    (hdr["CTYPE1"] == "WAVE-LOG" or
                     hdr["CTYPE1"] == "AWAV-LOG"):
                self._wave_mode = Spectrum.Mode.LOGLAMBDA
            else:
                self._wave_mode = Spectrum.Mode.LAMBDA

            # valid?
            self._valid = np.zeros(self.flux.shape, dtype=bool)

    def save(self, filename: str):
        """Save spectrum to FITS file

        Args:
            filename: If given, save to this filename.
        """

        # filename
        filename = filename or self._filename
        if not filename:
            raise ValueError('No filename given for saving the FITS file.')

        # create binary table
        c1 = fits.Column(name='WAVE', format='1D',
                         unit='Angstrom', array=self._wavelength)
        c2 = fits.Column(name='FLUX', format='1E',
                         unit='erg/s/cm2/Angstrom', array=self.flux)
        hdu = fits.new_table([c1, c2])
        hdu.header["EXTNAME"] = "SPECTRUM"

        # create HDU list and save it
        hdu_list = fits.HDUList([hdu])
        hdu_list.writeto(filename)

        # store filename
        self._filename = filename


class SpectrumAscii(Spectrum):
    """Handles spectra stored in ASCII files."""

    def __init__(self, filename: str = None, separator: str = ',', skip_lines: int = 0, comment: str = '#',
                 wave_column: int = 0, flux_column: int = 1, header: bool = True, *args, **kwargs):
        """Reads spectrum from ASCII file.

        Args:
            filename: Filename to load spectrum from.
            separator: Separator in file.
            skip_lines: Number of lines to skip while reading file.
            comment: Character indicating a comment line.
            header: Whether to write a header.
        """
        Spectrum.__init__(self, *args, **kwargs)

        # store
        self._filename = filename
        self._separator = separator
        self._header = header

        # load
        if filename:
            # init arrays
            wave = []
            flux = []

            # open file
            with open(filename, "r") as txt:
                # skip lines
                for i in range(skip_lines):
                    txt.readline()

                # read lines
                for line in txt:
                    if line[0] == comment:
                        continue
                    s = line.split(separator)
                    wave.append(float(s[wave_column]))
                    flux.append(float(s[flux_column]))

            # set everything
            self._wavelength = np.array(wave)
            self.flux = np.array(flux)
            self._wave_start = self._wavelength[0]
            self._wave_step = 0
            self._wave_mode = Spectrum.Mode.LOGLAMBDA \
                if self._wave_start < 10 else Spectrum.Mode.LAMBDA
            self._valid = np.zeros(self.flux.shape, dtype=bool)

    def save(self, filename: str):
        """Save spectrum to ASCII file

        Args:
            filename: If given, save to this filename.
        """

        # filename
        filename = filename or self._filename
        if not filename:
            raise ValueError('No filename given for saving the ASCII file.')

        # save it
        with open(filename, 'w') as f:
            # write header?
            if self._header:
                f.write('wavelength,flux\n')

            # write flux
            wave = self.wave
            for i in range(len(self.flux)):
                f.write("{0:f}{1:s}{2:f}\n".format(wave[i], self._separator, self.flux[i]))

        # store filename
        self._filename = filename


class SpectrumH5(Spectrum):
    """Handles spectra stored in H5 files."""

    def __init__(self, filename: str = None, *args, **kwargs):
        """Reads spectrum from H5 file.

        Args:
            filename: Filename to load spectrum from.
        """
        Spectrum.__init__(self, *args, **kwargs)

        # store
        self._filename = filename

        # load
        if filename:
            # read from H5
            fh5 = h5py.File(filename, 'r')
            wave = fh5['PHOENIX_SPECTRUM/wl'][()]
            flux = 10. ** fh5['PHOENIX_SPECTRUM/flux'][()]
            fh5.close()

            # set everything
            self._wavelength = np.array(wave)
            self.flux = np.array(flux)
            self._wave_start = self._wavelength[0]
            self._wave_step = 0
            self._wave_mode = Spectrum.Mode.LOGLAMBDA \
                if self._wave_start < 10 else Spectrum.Mode.LAMBDA
            self._valid = np.zeros(self.flux.shape, dtype=bool)

    def save(self, filename: str):
        """Save spectrum to H5 file

        Args:
            filename: If given, save to this filename.
        """
        pass

    @property
    def filename(self):
        return self._filename


__all__ = ['Spectrum', 'SpectrumAscii', 'SpectrumBinTableFITS', 'SpectrumFits', 'SpectrumFitsHDU', 'SpectrumHiResFITS',
           'SpectrumH5']