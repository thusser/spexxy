import logging
import numpy as np
import sys
import os
from astropy.io import fits

from spexxy.data import Spectrum


def add_parser(subparsers):
    """
    Adds 'spexxy tellurics create' command

    :param subparsers:  Subparser to attach new one to.
    """

    # init parser
    parser = subparsers.add_parser('create', help='Creates a 1D interpolator for tellurics.')
    parser.set_defaults(func=run)

    # grid
    parser.add_argument('grid', help='Grid to use for creating interpolator', type=str)

    # name of molecule
    parser.add_argument('molec', help='Name of molecule', type=str)

    # degree of polynomial
    parser.add_argument('-x', '--deg', help='Degree for polynomial in the fit', type=int, default=5)

    # output file
    parser.add_argument('-o', '--output', help='Output file', type=str, default='TelluricsInterpolator.fits')


def run(args):
    """
    Takes a list of spectra and calculates the mean of the fitted tellurics.

    :param args:    argparse namespace with
                    .spectra:   List of files containing spectra.
                    .output:    Output file for mean tellurics
                                (default: tellurics.fits)
                    .snlimit:   Only calculate mean tellurics for spectra
                                with a S/N higher than the given number.
                    .weight:    If set, tellurics are weightes by the S/N
                                of their spectra.
    """
    from spexxy.grid import create_grid

    # create grid
    grid = create_grid(args.grid)

    # load first
    av = np.array(grid.axis_values(0))
    first = grid.spectrum([av[0]])

    # wave mode
    ctype = "AWAV"
    if first.wave_mode == Spectrum.Mode.LOGLAMBDA:
        ctype = "AWAV-LOG"

    # load all spectra
    spectra = np.empty((len(av), len(first)))
    for i in range(len(av)):
        logging.info("Loading spectrum %d/%d...", i + 1, len(av))
        s = grid.spectrum([av[i]])
        spectra[i, :] = s.flux

    # create coefficients matrix
    coeffs = np.empty((args.deg + 1, len(first)))

    # loop all wavelength points
    for i in range(len(first)):
        # fit
        pc = 100. * (i + 1.) / len(first)
        logging.info("Fitting wavelength %d/%d (%.2f%%)...", i + 1, len(first), pc)
        sys.stdout.flush()
        c = np.polyfit(av, spectra[:, i], args.deg)
        coeffs[:, i] = c[::-1]

    # save to file
    hdu = fits.PrimaryHDU(coeffs.astype(np.float32))
    hdu.header["FILENAME"] = args.output
    hdu.header["CRPIX1"] = (1., "Reference pixel")
    hdu.header["CRVAL1"] = (first.wave_start, "Coordinate at reference pixel")
    hdu.header["CD1_1"] = (first.wave_step, "Coordinate increment per pixel")
    hdu.header["CDELT1"] = (first.wave_step, "Coordinate increment (prefer CD)")
    hdu.header["CREATOR"] = ("spexxy", "Written by spexxy")
    hdu.header["H_IDENT"] = ("Tellurics interpolator model", "Frame identifier")
    hdu.header["H_AXIS1"] = (2, "Wavelength")
    hdu.header["CTYPE1"] = (ctype, "Axis type")
    hdu.header["INTRP_T"] = ("A", "Interpolator-type: Relative")
    hdu.header["INTRP_C"] = ("C", "Interpolator-calibration: Flux calibrated")
    hdu.header["INTRP_V"] = ("2", "Interpolator version number")
    hdu.header["ULY_TYPE"] = ("TGM", "Type of model component (for ULySS)")
    hdu.header["MOLECULE"] = (args.molec, "Name of molecule")
    hdu.header["VALIDMIN"] = (av[0], "Mininum valid value for interpolator")
    hdu.header["VALIDMAX"] = (av[-1], "Maxinum valid value for interpolator")

    # delete existing
    if os.path.exists(args.output):
        os.remove(args.output)

    # write
    logging.info("Writing interpolator...")
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(args.output)
    logging.info("Finished.")
