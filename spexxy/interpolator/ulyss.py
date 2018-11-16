import os
import astropy.io.fits as pyfits
import numpy as np
import scipy.linalg
from typing import List, Tuple

from . import Interpolator
from ..data import Spectrum
from ..grid import GridAxis


class UlyssInterpolator(Interpolator):
    """Interpolator that works with Ulyss input files."""

    def __init__(self, filename: str, *args, **kwargs):
        """Initializes a new Ulyss interpolator.

        Args:
            filename: Name of file containing ulyss interpolator.
        """
        Interpolator.__init__(self, *args, **kwargs)

        # store it
        self.filename = os.path.expandvars(filename)

        # get info
        hdr = pyfits.getheader(self.filename, 0)
        self.wStart = hdr["CRVAL1"]
        self.wStep = hdr["CDELT1"]
        self.wCount = hdr["NAXIS1"]
        self.waveMode = Spectrum.Mode.LAMBDA
        if hdr["CTYPE1"] == "AWAV-LOG":
            self.waveMode = Spectrum.Mode.LOGLAMBDA

        # load coefficients
        self._coeffsWarm = pyfits.getdata(self.filename, 0)[:23]
        self._coeffsHot = pyfits.getdata(self.filename, 1)[:19]
        self._coeffsCold = pyfits.getdata(self.filename, 2)[:23]

    def axes(self) -> List[GridAxis]:
        """Returns information about the axes.

        Returns:
            List of GridAxis objects describing the grid's axes.
        """
        return [
            GridAxis(name='Teff', initial=6000., min=2300., max=15000.),
            GridAxis(name='logg', initial=4.5, min=0., max=6.),
            GridAxis(name='FeH', initial=-1.5, min=-4., max=1.)
        ]

    @staticmethod
    def _poly_matrix(params: Tuple) -> np.ndarray:
        """Create the matrix used in the polynomial evaluation for the given params.

        Args:
            params: Parameters to interpolate at

        Returns:
            Matrix that can be used for the actual interpolation
        """
        # primary scaling
        Teff = np.log10(params[0]) - 3.7617
        logg = params[1] - 4.44
        FeH = params[2]

        # matrix
        mat = np.zeros((3, 23))
        for i in [0, 1, 2]:
            # secondary scaling
            teffc = (Teff + 0.1) if i == 2 else Teff
            tt = teffc / 0.2
            tt2 = tt * tt - 1.

            # set it
            mat[i, 0] = 1
            mat[i, 1] = tt
            mat[i, 2] = FeH
            mat[i, 3] = logg
            mat[i, 4] = tt * tt
            mat[i, 5] = tt * tt2
            mat[i, 6] = tt2 * tt2
            mat[i, 7] = tt * FeH
            mat[i, 8] = tt * logg
            mat[i, 9] = tt2 * logg
            mat[i, 10] = tt2 * FeH
            mat[i, 11] = logg * logg
            mat[i, 12] = FeH * FeH
            mat[i, 13] = tt * tt2 * tt2
            mat[i, 14] = tt * logg * logg
            mat[i, 15] = logg * logg * logg
            mat[i, 16] = FeH * FeH * FeH
            mat[i, 17] = tt * FeH * FeH
            mat[i, 18] = logg * FeH
            mat[i, 19] = logg * logg * FeH
            mat[i, 20] = logg * FeH * FeH
            mat[i, 21] = np.exp(tt) - 1. - tt * (1. + tt / 2. +
                                                 tt * tt / 6. + tt * tt * tt / 24. + tt * tt * tt * tt / 120.)
            mat[i, 22] = np.exp(tt * 2.) - 1. - 2. * tt * (1. + tt +
                                                           2. / 3. * tt * tt + tt * tt * tt / 3. +
                                                           tt * tt * tt * tt * 2. / 15.)

        # return matrix
        return mat

    def __call__(self, params: Tuple) -> Spectrum:
        """Interpolates at the given parameter set.

        Args:
            params: Parameter set to interpolate at.

        Returns:
            Interpolated spectrum at given position.
        """

        # parameters
        teff = np.log10(params[0]) - 3.7617
        grav = params[1] - 4.44

        # create matrix
        mat = UlyssInterpolator._poly_matrix(params)

        # read coefficients
        if teff <= np.log10(9000.) - 3.7617:
            t1 = np.dot(mat[0, :], self._coeffsWarm)
        if teff >= np.log10(7000.) - 3.7617:
            t2 = np.dot(mat[1, :19], self._coeffsHot)
        if teff <= np.log10(4550.) - 3.7617:
            t3 = np.dot(mat[2, :], self._coeffsCold)

        # interpolate spectrum
        flux = None
        if teff <= np.log10(7000.) - 3.7617:
            if teff > np.log10(4550.) - 3.7617:
                flux = t1
            elif teff > np.log10(4000.) - 3.7617:
                q = (teff - np.log10(4000.) + 3.7617) / (np.log10(4550.) -
                                                         np.log10(4000.))
                flux = q * t1 + (1. - q) * t3
            else:
                flux = t3
        elif teff >= np.log10(9000.) - 3.7617:
            flux = t2
        else:
            q = (teff - np.log10(7000.) + 3.7617) / (np.log10(9000.) -
                                                     np.log10(7000.))
            flux = q * t2 + (1. - q) * t1

        # create spectrum and return it
        return Spectrum(flux=flux, wave_start=self.wStart, wave_step=self.wStep, wave_mode=self.waveMode)

    @staticmethod
    def create(files: List[str], output: str):
        """Create a new ulyss interpolator from a set of spectrum files.

        Args:
            files: List of spectrum files.
            output: Output file name.
        """

        # prepare lists for data
        teffs = [[], [], []]
        loggs = [[], [], []]
        fehs = [[], [], []]
        fluxes = [[], [], []]

        # load all spectra
        for i, f in enumerate(files):
            print("({0:d}/{1:d}) {2:s}".format(i, len(files), f))
            # open file
            fits = pyfits.open(f, memmap=False)

            # get params
            hdr = fits[0].header
            teff = hdr["PHXTEFF"]
            logg = hdr["PHXLOGG"]
            feh = hdr["PHXM_H"]

            # spectrum flux
            flux = fits[0].data
            flux /= np.mean(flux)

            # store it
            if teff < 4550:
                teffs[2].append(teff)
                loggs[2].append(logg)
                fehs[2].append(feh)
                fluxes[2].append(flux)
            if 4000 < teff < 9000:
                teffs[0].append(teff)
                loggs[0].append(logg)
                fehs[0].append(feh)
                fluxes[0].append(flux)
            if teff > 7000:
                teffs[1].append(teff)
                loggs[1].append(logg)
                fehs[1].append(feh)
                fluxes[1].append(flux)

            # close file
            fits.close()

        # our output HDUs
        hdus = []

        # loop regimes
        for regime in [0, 1, 2]:
            # create matrix with polynomial evaluations
            num = len(fluxes[regime])
            c = 19 if regime == 1 else 23
            polys = np.zeros((num, c))
            for i in range(num):
                # eval poly
                p = UlyssInterpolator._poly_matrix([teffs[regime][i], loggs[regime][i], fehs[regime][i]])
                # store it
                polys[i, :c] = p[regime, :c]

            # now loop wavelength bins
            coeffs = np.zeros((c, len(fluxes[regime][0])))
            for w in range(len(fluxes[regime][0])):
                # get all fluxes for this wavelength bin
                flx = [fluxes[regime][i][w] for i in range(len(fluxes[regime]))]
                # fit
                coeffs[:, w] = scipy.linalg.lstsq(polys, flx)[0]
                #print regime, w, coeffs[:, w]

            # store as ImageHDU
            if regime == 0:
                hdu = pyfits.PrimaryHDU(coeffs.astype(np.float32))
                hdr = pyfits.getheader(files[0])
                hdu.header["CRPIX1"] = 1
                hdu.header["CRVAL1"] = hdr["CRVAL1"]
                hdu.header["CDELT1"] = hdr["CDELT1"]
                hdu.header["CD1_1"] = hdr["CDELT1"]
                hdu.header["CTYPE1"] = hdr["CTYPE1"]
            else:
                hdu = pyfits.ImageHDU(coeffs.astype(np.float32))
                name = {1: "hot", 2: "cold"}
                hdu.header["EXTNAME"] = name[regime]
            hdus.append(hdu)

        # create FITS file
        hdulist = pyfits.HDUList(hdus)
        hdulist.writeto(output)


__all__ = ['UlyssInterpolator']
