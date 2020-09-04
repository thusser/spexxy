import logging
import math
import os
from typing import List, Union
import astropy.io.fits as pyfits
import numpy as np
import pandas as pd
import scipy
import scipy.optimize
import scipy.stats
from astropy.convolution import convolve

from spexxy.data.losvd import LOSVD
from spexxy.data.spectrum import Spectrum


class LSF(object):
    """Base class for classes dealing with Line Spread Functions."""

    def wavelength_points(self, log: bool = False) -> List[float]:
        """List of wavelength points at which the LSF is defined.

        Args:
            log: Return the wavelength points as log(wavelength).

        Returns:
            List of wavelength points
        """
        raise NotImplementedError

    def __call__(self, spec: Spectrum) -> Spectrum:
        """Apply LSF to given spectrum and return result.

        Args:
            spec: Spectrum to apply LSF to.

        Returns:
            Final spectrum.
        """
        raise NotImplementedError

    def wave_mode(self, mode: Spectrum.Mode):
        """Change the wavelength mode of the LSF (LAMBDA or LOGLAMBDA)

        Args:
            mode: New wavelength mode for LSF.
        """
        raise NotImplementedError

    @staticmethod
    def load(filename: str) -> Union['EmpiricalLSF', 'AnalyticalLSF']:
        """Loads a LSF from file

        Args:
            filename: Name of LSF file.

        Returns:
            The LSF object.
        """

        # expand vars on filename
        filename = os.path.expandvars(filename)

        # if it's a FITS file, we assume it to be an empirical LSF
        if filename.upper().endswith('.FIT') or filename.upper().endswith('.FITS'):
            return EmpiricalLSF(filename)

        # if it's a TXT or CSV, it must be an analytical LSF
        if filename.upper().endswith('.CSV') or filename.upper().endswith('.TXT'):
            return AnalyticalLSF(filename)

        # otherwise we don't know
        raise ValueError("Unknown LSF format.")


class AnalyticalLSF(LSF):
    """Describes an analytical LSF, i.e. one that is defined as LOSVDs at different wavelengths."""

    def __init__(self, filename: str = None):
        """Initialize a new analutical LSF

        Args:
            filename: Name of file to load LSF from. If none, empty LSF is created.
        """
        LSF.__init__(self)

        # init
        self.data = None
        self._wave_mode = None

        # load from file?
        if filename:
            # read data file
            self.data = pd.read_csv(filename, index_col=False)

            # crude check for wavelength mode:
            # if first wavelength is <100, it's assumed to be on a log scale
            self._wave_mode = Spectrum.Mode.LAMBDA if self.data['wave'].iloc[0] > 100. else Spectrum.Mode.LOGLAMBDA

    def save(self, filename: str):
        """Save LSF to file.

        Args:
            filename: Name of file to save LSF into.
        """
        if self.data is not None:
            self.data.to_csv(filename, index=False)

    def wavelength_points(self, log: bool = False) -> list:
        """List of wavelength points at which the LSF is defined.

        Args:
            log: Return the wavelength points as log(wavelength).

        Returns:
            List of wavelength points.
        """

        # get wavelength array
        wave = sorted(self.data['wave'].values)

        # check wave mode and convert, if necessary
        if log and self.ctype2 == Spectrum.Mode.LAMBDA:
            return np.log(wave)
        elif not log and self.ctype2 == Spectrum.Mode.LOGLAMBDA:
            return np.exp(wave)
        else:
            return wave

    def wave_mode(self, mode: Spectrum.Mode):
        """Change the wavelength mode of the LSF (LAMBDA or LOGLAMBDA)

        Args:
            mode: New wavelength mode for LSF.
        """

        # no need to change anything?
        if mode == self._wave_mode:
            return

        # do the actual conversion
        if mode == Spectrum.Mode.LAMBDA:
            self.data['wave'] = np.exp(self.data['wave'])
        else:
            self.data['wave'] = np.log(self.data['wave'])

    def __call__(self, spec: Spectrum) -> Spectrum:
        """Apply LSF to given spectrum and return result.

        Args:
            spec: Spectrum to apply LSF to.

        Returns:
            Final spectrum.
        """

        # check
        if spec.wave_mode != Spectrum.Mode.LOGLAMBDA:
            raise ValueError("Spectrum must be sampled on log scale.")

        # create all losvds
        losvd = []
        for row in self.data[['wave', 'v', 'sig', 'h3', 'h4', 'h5', 'h6']].itertuples():
            l = LOSVD(row[2:])
            losvd.append(l.kernel(l.x(spec.wave_step)))

        # get maximum size of losvd
        losvd_npix = np.max([len(l) for l in losvd])

        # get half size of LSF
        npix = int(losvd_npix / 2.)

        # array for new flux
        new_flux = np.empty((len(spec.wave)))

        # calculate new flux
        for w in range(npix+1, len(new_flux)-npix):
            # find best lsf point
            lsf_idx = np.argmin(np.abs(spec.wave[w] - self.wavelength_points()))

            # half size of this specific lsf
            npix_lsf = int(len(losvd[lsf_idx]) / 2.)

            # calculate new flux
            new_flux[w] = np.sum(spec.flux[w-npix_lsf:w+npix_lsf+1] * losvd[lsf_idx]) * spec.wave_step

            # normalize
            new_flux[w] /= np.sum(losvd[lsf_idx]) * spec.wave_step

        # create new spec
        return spec.__class__(flux=new_flux[npix + 1:-npix], wave_start=spec.wave_start + (npix + 1) * spec.wave_step,
                              wave_step=spec.wave_step, wave_mode=spec.wave_mode)

    @staticmethod
    def from_empirical(lsf: 'EmpiricalLSF'):
        """Fits LOSVD profiles to a given empirical LSF and creates an analytical LSF from it.

        Args:
            lsf: LSF to create analytical LSF from.

        Returns:
            New analytical LSF.
        """

        # check
        if not isinstance(lsf, EmpiricalLSF):
            raise ValueError("LSF must be of type EmpiricalLSF.")

        # loop all wavelength point
        data = []
        for i, lam in enumerate(lsf.wavelength_points()):
            # get LSF for this wavelength and normalize it
            lsf_data = lsf.data[i, :]
            lsf_data /= np.sum(lsf_data)

            # wave step in km/s
            wave = 299792.458 * (np.exp(lsf.wave_lsf()) - 1.)

            # fit
            p0 = [0., 100., 0., 0., 0., 0.]
            p = scipy.optimize.leastsq(lambda p, x, y: LOSVD(p).kernel(x) - y,
                                       p0, args=(wave, lsf_data))

            # store results
            data.append([lam] + list(p[0]))

        # to pandas dataframe
        df = pd.DataFrame(data, columns=['wave', 'v', 'sig', 'h3', 'h4', 'h5', 'h6'])

        # create new LSF
        lsf_new = AnalyticalLSF()
        lsf_new.data = df
        return lsf_new

    def convolve(self, fwhm: float):
        """Convolve the LSF with a Gaussian of given FWHM.

        Args:
            fwhm: FWHM of Gaussian to convolve with.
        """

        # loop all wavelength points
        for i in range(len(self.data)):
            # get sigma
            sig = self.data['sig'].iloc[i]

            # (de)convolve, which is just a quadratic addition/subtraction for a Gaussian
            sig = np.sqrt(sig**2. + np.sign(fwhm)*fwhm**2.)

            # and set again
            self.data['sig'].iloc[i] = sig


class EmpiricalLSF(LSF):
    """Describes an empirical LSF, i.e. a measured one, where actual values are provided."""

    def __init__(self, filename: str = None):
        """Initialize a new empirical LSF.

        Args:
            filename: If given, load LSF from this file.
        """
        LSF.__init__(self)

        # init
        self.data = None
        self.crval1 = None
        self.cdelt1 = None
        self.crval2 = None
        self.cdelt2 = None
        self.ctype1 = None
        self.ctype2 = None

        # load from file?
        if filename is not None:
            # load LSF
            self.data, hdr = pyfits.getdata(filename, header=True)

            # get header info
            self.crval1 = hdr["CRVAL1"]
            self.cdelt1 = hdr["CDELT1"]
            self.crval2 = hdr["CRVAL2"]
            self.cdelt2 = hdr["CDELT2"]

            # wave mode
            self.ctype1 = Spectrum.Mode.LOGLAMBDA \
                if "CTYPE1" in hdr.keys() and hdr["CTYPE1"] == "WAVE-LOG" else Spectrum.Mode.LAMBDA
            self.ctype2 = Spectrum.Mode.LOGLAMBDA \
                if "CTYPE2" in hdr.keys() and hdr["CTYPE2"] == "WAVE-LOG" else Spectrum.Mode.LAMBDA

    def save(self, filename: str):
        """Save LSF to file.

        Args:
            filename: Name of file to write LSF into.
        """

        # create HDU and fill it
        hdu = pyfits.PrimaryHDU(self.data)
        hdu.header["CRVAL1"] = self.crval1
        hdu.header["CDELT1"] = self.cdelt1
        hdu.header["CD1_1"] = self.cdelt1
        hdu.header["CRPIX1"] = 1
        hdu.header["CRVAL2"] = self.crval2
        hdu.header["CDELT2"] = self.cdelt2
        hdu.header["CD2_2"] = self.cdelt2
        hdu.header["CRPIX2"] = 1
        hdu.header["CD1_2"] = 0.
        hdu.header["CD2_1"] = 0.
        hdu.header["CTYPE1"] = "WAVE" if self.ctype1 == Spectrum.Mode.LAMBDA else "WAVE-LOG"
        hdu.header["CTYPE2"] = "WAVE" if self.ctype2 == Spectrum.Mode.LAMBDA else "WAVE-LOG"

        # write it
        hdulist = pyfits.HDUList([hdu])
        if os.path.exists(filename):
            os.remove(filename)
        hdulist.writeto(filename)

    def wave_lsf(self):
        """Return wavelength array for the individual LSFs at different wavelengths."""
        return np.arange(self.crval1, self.crval1 + self.cdelt1 * self.data.shape[1], self.cdelt1)

    def wavelength_points(self, log: bool = False) -> List[float]:
        """List of wavelength points at which the LSF is defined.

        Args:
            log: Return the wavelength points as log(wavelength).

        Returns:
            List of wavelength points.
        """

        # build wave array
        wave = np.arange(self.crval2, self.crval2 + self.cdelt2 * self.data.shape[0], self.cdelt2)

        # check wave mode and convert, if necessary
        if log and self.ctype2 == Spectrum.Mode.LAMBDA:
            return np.log(wave)
        elif not log and self.ctype2 == Spectrum.Mode.LOGLAMBDA:
            return np.exp(wave)
        else:
            return wave

    def get_lsf_closest_to_wave(self, wave: float, log: bool = False) -> np.ndarray:
        """Return LSF defined closest to the given wavelength.

        Args:
            wave: Wavelength to return LSF for.
            log: Whether the given wavelength is on a log scale.

        Returns:
            LSF closest to given wavelength.
        """

        # get wavelength points
        points = np.array(self.wavelength_points(log=log))

        # get closest and return it
        p = np.argmin(abs(points - wave))
        return self.data[p, :]

    def extract_at_wave(self, wave: float) -> 'EmpiricalLSF':
        """Extract LSF at given wavelength

        Args:
            wave: Wavelength to extract at.

        Returns:
            New LSF
        """

        # create new lsf
        lsf = EmpiricalLSF()

        # copy values
        lsf.crval1 = self.crval1
        lsf.cdelt1 = self.cdelt1
        lsf.ctype1 = self.ctype1

        # ignore 2nd dimension
        lsf.crval2 = 0
        lsf.cdelt2 = 1
        lsf.ctype2 = Spectrum.Mode.LAMBDA

        # extract data and set it
        data = self.get_lsf_closest_to_wave(wave)
        lsf.data = data.reshape((1, len(data)))
        return lsf

    def wave_mode(self, mode: Spectrum.Mode):
        """Change the wavelength mode of the LSF (LAMBDA or LOGLAMBDA)

        Args:
            mode: New wavelength mode for LSF.
        """

        # no change required?
        if mode == self.ctype1:
            return

        # get wavelength array
        wave_lsf = self.wave_lsf()
        wave_spec = self.wavelength_points()

        # get largest extent of new array
        new_range = None
        for lam in wave_spec:
            # calculate new wavelength array for this lambda
            tmp = wave_lsf / lam if mode == Spectrum.Mode.LOGLAMBDA else wave_lsf * lam
            # larger?
            if new_range is None or tmp[-1]-tmp[0] > new_range[1]-new_range[0]:
                new_range = (tmp[0], tmp[-1])

        # create new wavelength array
        new_step = (new_range[1] - new_range[0]) / self.data.shape[1]
        new_wave_lsf = np.arange(new_range[0], new_range[1], new_step)

        # resample and rescale lsf
        for i, lam in enumerate(wave_spec):
            # get data
            data = self.data[i, :]

            # get wavelength array for this
            wave = wave_lsf / lam if mode == Spectrum.Mode.LOGLAMBDA else wave_lsf * lam

            # resample to new grid
            ip = scipy.interpolate.interp1d(x=wave, y=data, kind='linear', bounds_error=False, fill_value=0.)
            tmp = ip(new_wave_lsf)

            # normalize
            self.data[i, :] = tmp / np.sum(tmp) # np.trapz(tmp, new_wave_lsf)

        # set wave parameters
        self.crval1 = new_range[0]
        self.cdelt1 = new_step
        self.ctype1 = mode

    def resample(self, spec: Spectrum):
        """Resample LSF to match given spectrum.

        Args:
            spec: Spectrum to resample to.
        """

        # do wave modes match?
        if self.ctype1 != spec.wave_mode:
            raise ValueError("Wave modes of LSF and spectrum do not match.")

        # alreay good?
        if self.wave_lsf() == spec.wave:
            return

        # get number of pixels required for LSF and obtain new sampling
        npix = math.ceil(abs(self.crval1) / spec.wave_step)
        wave_lsf = np.arange(-spec.wave_step * npix, spec.wave_step * npix + spec.wave_step * 0.5, spec.wave_step)

        # blow up data to full spectrum's range and sampling
        if self.crval2 > 0:
            w = self.wavelength_points(log=(spec.wave_mode == Spectrum.Mode.LOGLAMBDA))
            ip = scipy.interpolate.interp2d(self.wave_lsf(), w, self.data)
            self.data = ip(wave_lsf, w)
        else:
            ip = scipy.interpolate.interp1d(self.wave_lsf(), self.data[0, :], fill_value='extrapolate')
            self.data = ip(wave_lsf).reshape(1, len(wave_lsf))

        # set wave parameters
        self.crval1 = wave_lsf[0]
        self.cdelt1 = spec.wave_step
        self.ctype1 = spec.wave_mode

    def __call__(self, spec: Spectrum) -> Spectrum:
        """Apply LSF to given spectrum and return result.

        Args:
            spec: Spectrum to apply LSF to.

        Returns:
            Final spectrum.
        """

        # check
        if self.cdelt1 != spec.wave_step:
            raise ValueError("Samplings of LSF and spectrum do not match.")

        # get lsf wave points
        wave_spec = self.wavelength_points(spec.wave_mode == Spectrum.Mode.LOGLAMBDA)

        # get half size of LSF
        npix = int(self.data.shape[1] / 2.)

        # LSF at multiple wavelengths?
        if self.crval1 > 0:
            # array for new flux
            new_flux = np.empty((len(spec.wave)))

            # calculate new flux
            for w in range(npix+1, len(new_flux)-npix):
                # find best lsf point
                lsf_idx = np.argmin(np.abs(spec.wave[w] - wave_spec))
                # calculate new flux
                new_flux[w] = np.sum(spec.flux[w-npix:w+npix+1] * self.data[lsf_idx, :]) # * spec.wave_step
                new_flux[w] /= np.sum(self.data[lsf_idx, :])
        else:
            # just convolve
            new_flux = convolve(spec.flux, self.data[0, :], boundary='extend')

        # create new spec
        return spec.__class__(flux=new_flux[npix + 1:-npix], wave_start=spec.wave_start + (npix + 1) * spec.wave_step,
                              wave_step=spec.wave_step, wave_mode=spec.wave_mode)


__all__ = ['LSF', 'AnalyticalLSF', 'EmpiricalLSF']
