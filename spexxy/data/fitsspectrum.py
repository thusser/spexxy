import os
import numpy as np
from astropy.io import fits
from spexxy.data.resultsfits import ResultsFITS
from spexxy.data.spectrum import SpectrumFitsHDU


class FitsSpectrum(object):
    """A class for handling both spectra and results in a FITS file."""

    def __init__(self, filename: str, mode: str = 'r'):
        """Open an existing or create a new FITS file.

        Args:
            filename: Name of file to open.
            mode: Open mode (w=write, r=read, rw=read/write).
        """

        # store it
        self._filename = filename
        self._mode = mode
        self._hdu_as_spectrum = {}

        # check mode
        if mode == 'r':
            # open fits file
            self._fits = fits.open(filename, memmap=False)

        elif mode == 'w':
            # create an empty HDU list
            self._fits = fits.HDUList()

        elif mode == 'rw':
            # open fits file in update mode
            self._fits = fits.open(filename, memmap=False, mode='update')

        else:
            raise ValueError('Invalid mode for opening FITS file.')

    def __enter__(self):
        """Returns just itself, when class is used in with statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """When used in with statement, leaving it closes the file."""
        self.close()

    def close(self):
        """Close the file."""

        # check mode
        if self._mode == 'r':
            # just close file
            self._fits.close()

        elif self._mode == 'w':
            # save file
            if os.path.exists(self._filename):
                os.remove(self._filename)
            self._fits.writeto(self._filename)
            self._fits.close()

        elif self._mode == 'rw':
            # flush and close
            self._fits.flush()
            self._fits.close()

    def save(self, filename: str = None):
        """Save spectrum to file.

        Args:
            filename: If filename is given, save to new file, otherwise use existing one (if exists).
        """

        # update HDUs
        for hdu, spec in self._hdu_as_spectrum.items():
            spec.hdu()

        # no filename given?
        filename = filename or self._filename
        if not filename:
            raise ValueError('No filename given for saving spectrum.')

        # save to new filename
        if os.path.exists(filename):
            os.remove(filename)
        self._fits.writeto(filename)

        # in any case, we change mode and filename
        self._filename = filename

    @property
    def filename(self):
        """Returns filename of spectrum, or None."""
        return self._filename

    def hdu_names(self):
        """Returns names of HDUs in FITS file."""
        return [hdu.header['EXTNAME'] if 'EXTNAME' in hdu.header else ''
                for hdu in self._fits]

    def hdu(self, hdu: int):
        """Returns a HDU, either by number or by name.

        Args:
            hdu: Number or name of HDU to return.

        Returns:
            Selected HDU.
        """

        return self._fits[hdu]

    def __len__(self):
        """Return number of HDUs in FITS file."""
        return len(self._fits)

    def __contains__(self, hdu):
        """Whether a HDU exists."""
        return hdu in self._fits

    def __getitem__(self, hdu):
        """Returns spectrum for given HDU.

        Args:
            param:  HDU to return spectrum for.

        Returns:
            A SpectrumFitsHDU object.
        """

        # get hdu
        fits_hdu = self.hdu(hdu)

        # stored already?
        if fits_hdu in self._hdu_as_spectrum:
            return self._hdu_as_spectrum[fits_hdu]

        # workaround: if CRVAL1, CDELT1 do not exist in HDU, take from primary
        """
        if 'CRVAL1' not in fits_hdu.header:
            # copy, if possible
            if 'CRVAL1' in self._fits[0].header:
                fits_hdu.header['CRVAL1'] = self._fits[0].header['CRVAL1']
                fits_hdu.header['CDELT1'] = self._fits[0].header['CDELT1']
            elif 'WAVE' in self._fits[0].header:
                fits_hdu.header['WAVE'] = self._fits[0].header['WAVE']
        """

        # match data?
        if 'CRVAL1' not in fits_hdu.header and 'WAVE' not in fits_hdu.header:
            raise ValueError('Not a spectrum extension.')

        # create spectrum and return it
        self._hdu_as_spectrum[fits_hdu] = SpectrumFitsHDU(fits_hdu, hdu_list=self._fits, filename=self._filename)
        return self._hdu_as_spectrum[fits_hdu]

    def __setitem__(self, hdu_name, spec):
        """Sets a new spectrum for a HDU.

        Args:
            hdu_name: Name of HDU.

        Returns:
            Spectrum to set for that HDU.
        """

        # spec must be a SpectrumFitsHDU instance
        if not isinstance(spec, SpectrumFitsHDU):
            raise ValueError('Spectrum must be instance of SpectrumFitsHDU.')

        # check type
        if hdu_name == 0 and not spec.primary:
            raise ValueError('Cannot store non-primary HDU as 0.')
        if hdu_name != 0 and spec.primary:
            raise ValueError('Cannot store primary HDU with other name than 0.')

        # get hdu
        spec_hdu, wave_hdu = spec.hdu()

        # is it a primary HDU?
        if spec.primary:
            # yes, find and remove existing one
            prim = list(filter(lambda hdu: isinstance(hdu, fits.PrimaryHDU), self._fits))
            if len(prim) > 0:
                # remove from list of HDUs
                self._fits.remove(prim[0])
                # remove from mapping
                if prim[0] in self._hdu_as_spectrum:
                    del self._hdu_as_spectrum[prim[0]]

            # add it
            self._fits.insert(0, spec_hdu)
            self._hdu_as_spectrum[spec_hdu] = spec

        else:
            # extension HDU
            # name taken already?
            if hdu_name in self._fits:
                # get hdu of that name
                exist_hdu = self.hdu(hdu_name)

                # and the owner is not the given spectrum?
                if spec_hdu != exist_hdu:
                    # get index and delete HDU
                    del self._fits[hdu_name]
                    if exist_hdu in self._hdu_as_spectrum:
                        del self._hdu_as_spectrum[exist_hdu]

            # rename it
            spec_hdu.name = hdu_name

            # add to lists
            self._fits.append(spec_hdu)
            self._hdu_as_spectrum[spec_hdu] = spec

    def __delitem__(self, hdu_name):
        if hdu_name in self._fits:
            del self._fits[hdu_name]

    def append(self, hdu):
        self._fits.append(hdu)

    def wave_count(self):
        """Returns number of elements in spectrum."""
        spec = self.spectrum
        return len(spec)

    @property
    def spectrum(self):
        """Returns main spectrum, short vor FitsSpectrum[0]"""
        return self[0]

    @spectrum.setter
    def spectrum(self, spec):
        """Sets main spectrum, short vor FitsSpectrum[0]=spec"""
        self[0] = spec

    def results(self, namespace):
        """Returns a ResultsFITS object for the given namespace.

        Args:
            namespace: Namespace for the results.

        Returns:
            ResultsFITS object.
        """
        return ResultsFITS(self.hdu(0), namespace)

    @property
    def sigma(self):
        """Returns uncertainties for spectrum, or array of ones if they don't exist."""
        if 'SIGMA' in self.hdu_names():
            return self['SIGMA'].flux
        else:
            return np.ones((self.wave_count()))

    @property
    def good_pixels(self):
        """Returns mask of valid for spectrum, or array of ones if they don't exist."""
        if 'GOODPIXELS' in self.hdu_names():
            return self['GOODPIXELS'].flux.astype(np.bool)
        else:
            return np.ones((self.wave_count()), dtype=np.bool)

    @good_pixels.setter
    def good_pixels(self, arr: np.ndarray):
        """Sets mask of valid pixels in spectrum.

        Args:
            arr: New mask of valid pixels.
        """

        # create new Spectrum and set mask as flux
        spec = SpectrumFitsHDU(spec=self.spectrum, dtype=np.int16, primary=False)
        spec.flux = arr.astype(np.bool)

        # write into HDU
        self['GOODPIXELS'] = spec

    @property
    def residuals(self):
        """Returns residuals of spectrum fit or None if there are none."""
        return self['RESIDUALS'].flux if 'RESIDUALS' in self.hdu_names() else None

    @residuals.setter
    def residuals(self, arr: np.ndarray):
        """Sets residuals.

        Args:
            arr: New residuals for spectrum.
        """

        # create new Spectrum and set residuals as flux
        spec = SpectrumFitsHDU(spec=self.spectrum, primary=False)
        spec.flux = arr

        # write into HDU
        self['RESIDUALS'] = spec

    @property
    def mult_poly(self):
        """Returns multiplicative polynomial from spectrum fitting, or array of ones if they don't exist."""
        if 'MULTPOLY' in self.hdu_names():
            return self['MULTPOLY'].flux
        else:
            return np.ones((self.wave_count()))

    @mult_poly.setter
    def mult_poly(self, arr: np.ndarray):
        """Sets new multiplicative polynomial.

        Args:
            arr: New multiplicative polynomial for spectrum.
        """

        # create new Spectrum and set residuals as flux
        spec = SpectrumFitsHDU(spec=self.spectrum, primary=False)
        spec.flux = arr

        # write into HDU
        self['MULTPOLY'] = spec

    @property
    def best_fit(self):
        """Returns best fit of spectrum fit or None if there is none."""
        return self['BESTFIT'] if 'BESTFIT' in self.hdu_names() else None

    @best_fit.setter
    def best_fit(self, best: np.ndarray):
        """Sets new best fit.

        Args:
            best:  New best fit for spectrum.
        """

        # create new Spectrum and write it into HDU
        spec = SpectrumFitsHDU(spec=best, primary=False)
        self['BESTFIT'] = spec

    @property
    def tellurics(self):
        """Returns tellurics of spectrum fit or None if there is none."""
        return self['TELLURICS'] if 'TELLURICS' in self.hdu_names() else None

    @property
    def header(self):
        """Returns FITS header."""
        return self._fits[0].header

    @property
    def covar(self):
        """Returns covariance matrix."""
        return

    @covar.setter
    def covar(self, cov: np.ndarray):
        """Sets new covariance matrix.

        Args:
            cov: New covariance matrix for fit.
        """
        # name taken already?
        if 'COVAR' in self._fits:
            del self._fits['COVAR']

        # create HDU
        hdu = fits.ImageHDU(cov)
        hdu.name = 'COVAR'

        # add to list
        self._fits.append(hdu)


__all__ = ['FitsSpectrum']
